from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import tempfile
import os
import io
import logging
from typing import Dict


class HairRemovalProcessor:
    """Class for handling hair removal from dermoscopic images with quality assessment."""

    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_quality_metrics(original_img: np.ndarray, processed_img: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics between original and processed images.

        Args:
            original_img: Original image
            processed_img: Processed image

        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Ensure images are same size
            if original_img.shape != processed_img.shape:
                processed_img = cv2.resize(processed_img, (original_img.shape[1], original_img.shape[0]))

            # Convert to grayscale if needed
            if len(original_img.shape) == 3:
                orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original_img

            if len(processed_img.shape) == 3:
                proc_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            else:
                proc_gray = processed_img

            # Calculate SSIM
            ssim_score = ssim(orig_gray, proc_gray)

            # Calculate PSNR
            psnr_score = psnr(orig_gray, proc_gray)

            # Calculate MSE
            mse_score = mse(orig_gray, proc_gray)

            # Calculate Normalized Cross-Correlation
            norm_orig = (orig_gray - np.mean(orig_gray)) / (np.std(orig_gray) + 1e-10)
            norm_proc = (proc_gray - np.mean(proc_gray)) / (np.std(proc_gray) + 1e-10)
            ncc = np.mean(norm_orig * norm_proc)

            # Calculate histogram similarity
            hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
            hist_proc = cv2.calcHist([proc_gray], [0], None, [256], [0, 256])
            hist_similarity = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CORREL)

            return {
                'ssim': float(ssim_score),
                'psnr': float(psnr_score),
                'mse': float(mse_score),
                'ncc': float(ncc),
                'histogram_similarity': float(hist_similarity)
            }

        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return None

    @staticmethod
    async def process_image(file: UploadFile, include_metrics: bool = True) -> Dict[str, bytes]:
        """
        Process uploaded image file and remove hair artifacts.

        Args:
            file: Uploaded image file
            include_metrics: Whether to include quality metrics

        Returns:
            Dictionary containing processed image data and quality metrics
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        try:
            # Read image
            image = cv2.imread(temp_file_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Could not read uploaded image")

            # Store original for metrics calculation
            original = image.copy()

            # Hair removal processing
            # Convert to grayscale
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Create kernel size based on image dimensions
            kernel_size = max(9, min(image.shape) // 50)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Apply blackhat filter
            kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))
            blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

            # Apply Gaussian blur
            blur_size = max(3, kernel_size // 3)
            if blur_size % 2 == 0:
                blur_size += 1
            bhg = cv2.GaussianBlur(blackhat, (blur_size, blur_size), cv2.BORDER_DEFAULT)

            # Create binary mask
            thresh_value = np.mean(bhg) + np.std(bhg)
            _, mask = cv2.threshold(bhg, thresh_value, 255, cv2.THRESH_BINARY)

            # Inpaint to remove hair
            radius = max(6, kernel_size // 2)
            clean = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)

            # Prepare results
            results = {}
            for key, img in {
                'original': image,
                'grayscale': cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR),
                'blackhat': cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR),
                'mask': cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                'clean': clean
            }.items():
                _, buffer = cv2.imencode('.png', img)
                results[key] = buffer.tobytes()

            # Calculate quality metrics if requested
            if include_metrics:
                metrics = HairRemovalProcessor.calculate_quality_metrics(original, clean)
                if metrics:
                    results['quality_metrics'] = metrics

            return results

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            os.unlink(temp_file_path)


# FastAPI endpoint
async def remove_hair(
        file: UploadFile = File(...),
        include_metrics: bool = True
) -> StreamingResponse:
    """
    Endpoint to remove hair from dermoscopic images.

    Args:
        file: The uploaded image file
        include_metrics: Whether to include quality metrics in response

    Returns:
        StreamingResponse containing the processed image
    """
    try:
        # Process image
        results = await HairRemovalProcessor.process_image(file, include_metrics)

        # Prepare headers with quality metrics if available
        headers = {}
        if 'quality_metrics' in results:
            headers['X-Quality-Metrics'] = str(results['quality_metrics'])

        # Return the clean image
        return StreamingResponse(
            io.BytesIO(results['clean']),
            media_type="image/png",
            headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FastAPI route example
"""
@app.post("/remove-hair/")
async def remove_hair_endpoint(
    file: UploadFile = File(...),
    include_metrics: bool = True
):
    return await remove_hair(file, include_metrics)
"""