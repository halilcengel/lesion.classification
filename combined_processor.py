from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from skimage.color import rgb2lab
from skfuzzy import cmeans
import tempfile
import os
import io
from typing import Dict
import logging


class DermoscopyImageProcessor:
    """
    Combined processor for dermoscopic images that handles both
    hair removal and lesion segmentation.
    """

    def __init__(self, n_clusters: int = 2, fuzzy_m: float = 2,
                 error: float = 0.005, max_iter: int = 1000):
        """Initialize processor with FCM parameters."""
        self.n_clusters = n_clusters
        self.fuzzy_m = fuzzy_m
        self.error = error
        self.max_iter = max_iter
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the processor."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        """Safely normalize array to [0,1] range."""
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min == 0:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    @staticmethod
    def _remove_hair(image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Remove hair artifacts from dermoscopic image.

        Args:
            image: Input image array

        Returns:
            Dictionary containing processing stages
        """
        try:
            # Convert to grayscale
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Create kernel size based on image dimensions
            kernel_size = max(9, min(image.shape) // 50)
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1

            # Apply blackhat filter with adaptive kernel size
            kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))
            blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

            # Apply Gaussian blur with adaptive kernel size
            blur_size = max(3, kernel_size // 3)
            if blur_size % 2 == 0:
                blur_size += 1
            bhg = cv2.GaussianBlur(blackhat, (blur_size, blur_size), cv2.BORDER_DEFAULT)

            # Create binary mask with adaptive threshold
            thresh_value = np.mean(bhg) + np.std(bhg)
            _, mask = cv2.threshold(bhg, thresh_value, 255, cv2.THRESH_BINARY)

            # Inpaint to remove hair with adaptive radius
            radius = max(6, kernel_size // 2)
            clean = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)

            return {
                'original': image,
                'grayscale': grayscale,
                'blackhat': blackhat,
                'mask': mask,
                'clean': clean
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in hair removal: {str(e)}")

    async def _segment_lesion(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment lesion from hair-removed image.

        Args:
            image: Clean image after hair removal

        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Convert to LAB and extract L channel
            lab_img = rgb2lab(image)
            l_channel = self._normalize_array(lab_img[:, :, 0])

            # Apply FCM clustering
            pixels = l_channel.reshape((-1, 1))
            cntr, u, *_ = cmeans(
                pixels.T,
                self.n_clusters,
                self.fuzzy_m,
                self.error,
                self.max_iter
            )

            # Create segmentation mask
            segmentation = np.argmax(u, axis=0).reshape(l_channel.shape)

            # Create kernel size based on image dimensions
            kernel_size = max(5, min(image.shape) // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Post-process segmentation
            refined = cv2.morphologyEx(
                segmentation.astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel
            )
            refined = cv2.morphologyEx(
                refined,
                cv2.MORPH_OPEN,
                kernel
            )

            # Create final segmented image
            mask = (refined == 1).astype(np.uint8)
            segmented = cv2.bitwise_and(image, image, mask=mask)

            return {
                'preprocessed': l_channel,
                'mask': refined * 255,
                'segmented': segmented
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in lesion segmentation: {str(e)}")

    @classmethod
    async def process_image(cls, file: UploadFile) -> Dict[str, bytes]:
        """
        Process image through complete pipeline (hair removal + segmentation).

        Args:
            file: Uploaded image file

        Returns:
            Dictionary containing all processing stages as PNG-encoded bytes
        """
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        processor = cls()

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        try:
            # Read image
            image = cv2.imread(temp_file_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Could not read uploaded image")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Step 1: Remove hair
            hair_removal_results = cls._remove_hair(image)

            # Step 2: Segment lesion from clean image
            segmentation_results = await processor._segment_lesion(hair_removal_results['clean'])

            # Combine results
            all_results = {
                'original': cv2.cvtColor(image, cv2.COLOR_RGB2BGR),  # Convert back to BGR for OpenCV
                'hair_removed': cv2.cvtColor(hair_removal_results['clean'], cv2.COLOR_RGB2BGR),
                'hair_mask': hair_removal_results['mask'],
                'lesion_mask': segmentation_results['mask'],
                'final_segmented': cv2.cvtColor(segmentation_results['segmented'], cv2.COLOR_RGB2BGR)
            }

            # Encode results
            encoded_results = {}
            for key, img in all_results.items():
                if img is not None:
                    _, buffer = cv2.imencode('.png', img)
                    encoded_results[key] = buffer.tobytes()

            return encoded_results

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)


# FastAPI endpoint
async def process_dermoscopy_image(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Endpoint to process dermoscopic images (hair removal + lesion segmentation).

    Args:
        file: The uploaded image file

    Returns:
        StreamingResponse containing the final processed image
    """
    try:
        # Process image
        results = await DermoscopyImageProcessor.process_image(file)

        # Return the final segmented image
        return StreamingResponse(
            io.BytesIO(results['final_segmented']),
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))