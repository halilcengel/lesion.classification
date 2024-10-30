from fastapi import HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from skimage.color import rgb2lab
from skfuzzy import cmeans
import io
import logging
from typing import Dict, Union
import tempfile
import os


class LesionSegmentationProcessor:
    """
    A class for processing and segmenting skin lesion images using Fuzzy C-means clustering,
    adapted for FastAPI usage.
    """

    def __init__(self, n_clusters: int = 2, fuzzy_m: float = 2,
                 error: float = 0.005, max_iter: int = 1000):
        """Initialize the processor with FCM parameters."""
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
    async def _bytes_to_image(contents: bytes) -> np.ndarray:
        """
        Convert bytes to OpenCV image.

        Args:
            contents: Image file contents as bytes

        Returns:
            numpy.ndarray: The decoded image
        """
        # Create temporary file to store image bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        try:
            # Read image with OpenCV
            img = cv2.imread(temp_file_path)
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode image")

            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    @staticmethod
    async def _image_to_bytes(image: np.ndarray, format: str = '.png') -> bytes:
        """
        Convert numpy image array to bytes.

        Args:
            image: Image array to encode
            format: Output image format

        Returns:
            bytes: Encoded image
        """
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Encode image
        _, buffer = cv2.imencode(format, image)
        return buffer.tobytes()

    async def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.

        Args:
            img: Input RGB image array

        Returns:
            numpy.ndarray: Preprocessed image (L channel)
        """
        try:
            # Convert to LAB color space
            lab_img = rgb2lab(img)

            # Extract and normalize L channel
            l_channel = lab_img[:, :, 0]
            l_channel = self._normalize_array(l_channel)

            self.logger.info("Image preprocessing completed successfully")
            return l_channel

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            raise HTTPException(status_code=500, detail="Error preprocessing image")

    async def apply_fcm(self, image: np.ndarray) -> np.ndarray:
        """Apply Fuzzy C-means clustering to segment the image."""
        try:
            # Reshape for FCM
            pixels = image.reshape((-1, 1))

            # Apply FCM clustering
            cntr, u, *_ = cmeans(
                data=pixels.T,
                c=self.n_clusters,
                m=self.fuzzy_m,
                error=self.error,
                maxiter=self.max_iter
            )

            # Create segmentation mask
            segmentation = np.argmax(u, axis=0).reshape(image.shape)

            self.logger.info("FCM segmentation completed successfully")
            return segmentation

        except Exception as e:
            self.logger.error(f"Error during FCM application: {str(e)}")
            raise HTTPException(status_code=500, detail="Error during image segmentation")

    async def post_process(self, segmentation: np.ndarray) -> np.ndarray:
        """Apply morphological operations to refine the segmentation."""
        try:
            # Create kernel for morphological operations
            kernel = np.ones((5, 5), np.uint8)

            # Apply closing and opening operations
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

            self.logger.info("Post-processing completed successfully")
            return refined

        except Exception as e:
            self.logger.error(f"Error during post-processing: {str(e)}")
            raise HTTPException(status_code=500, detail="Error during image post-processing")

    @classmethod
    async def process_image(cls, contents: bytes) -> Dict[str, bytes]:
        """
        Process an image through the complete pipeline.

        Args:
            contents: Image file contents as bytes

        Returns:
            Dictionary containing original, mask, and segmented images as bytes
        """
        processor = cls()

        try:
            # Convert bytes to image
            original_img = await cls._bytes_to_image(contents)

            # Process image
            preprocessed_img = await processor.preprocess_image(original_img)
            segmentation = await processor.apply_fcm(preprocessed_img)
            refined_segmentation = await processor.post_process(segmentation)

            # Create segmented image
            mask = (refined_segmentation == 1).astype(np.uint8)
            segmented_img = cv2.bitwise_and(original_img, original_img, mask=mask)

            # Convert results to bytes
            results = {
                'original': await cls._image_to_bytes(original_img),
                'mask': await cls._image_to_bytes(refined_segmentation * 255),
                'segmented': await cls._image_to_bytes(segmented_img)
            }

            processor.logger.info("Complete image processing pipeline completed successfully")
            return results

        except HTTPException:
            raise
        except Exception as e:
            processor.logger.error(f"Error in processing pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image")


# FastAPI endpoint
async def segment_lesion(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Endpoint to segment skin lesions in dermoscopic images.

    Args:
        file: The uploaded image file

    Returns:
        StreamingResponse containing the segmented image
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read file contents
        contents = await file.read()

        # Process image
        results = await LesionSegmentationProcessor.process_image(contents)

        # Return the segmented image
        return StreamingResponse(
            io.BytesIO(results['segmented']),
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))