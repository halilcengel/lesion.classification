import tempfile
import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import logging
from typing import Dict, Tuple, Union, Any
from combined_processor import DermoscopyImageProcessor

class ImageQualityAssessor:
    """
    Class for assessing image quality using multiple metrics with reference images.
    """

    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_metrics(reference_img: np.ndarray, processed_img: np.ndarray) -> Dict[str, float]:
        """
        Calculate multiple image quality metrics.

        Args:
            reference_img: Reference image
            processed_img: Processed image to compare

        Returns:
            Dictionary containing various quality metrics
        """
        try:
            # Ensure images are same size
            if reference_img.shape != processed_img.shape:
                processed_img = cv2.resize(processed_img, (reference_img.shape[1], reference_img.shape[0]))

            # Convert to grayscale if needed
            if len(reference_img.shape) == 3:
                ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = reference_img

            if len(processed_img.shape) == 3:
                proc_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            else:
                proc_gray = processed_img

            # Calculate SSIM
            ssim_score = ssim(ref_gray, proc_gray)

            # Calculate PSNR
            psnr_score = psnr(ref_gray, proc_gray)

            # Calculate MSE
            mse_score = mse(ref_gray, proc_gray)

            # Calculate Normalized Cross-Correlation
            norm_ref = (ref_gray - np.mean(ref_gray)) / np.std(ref_gray)
            norm_proc = (proc_gray - np.mean(proc_gray)) / np.std(proc_gray)
            ncc = np.mean(norm_ref * norm_proc)

            # Calculate histogram similarity
            hist_ref = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            hist_proc = cv2.calcHist([proc_gray], [0], None, [256], [0, 256])
            hist_similarity = cv2.compareHist(hist_ref, hist_proc, cv2.HISTCMP_CORREL)

            metrics = {
                'ssim': float(ssim_score),
                'psnr': float(psnr_score),
                'mse': float(mse_score),
                'ncc': float(ncc),
                'histogram_similarity': float(hist_similarity)
            }

            return metrics

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

    @staticmethod
    def assess_segmentation_quality(reference_mask: np.ndarray,
                                    predicted_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate segmentation-specific metrics.

        Args:
            reference_mask: Ground truth segmentation mask
            predicted_mask: Predicted segmentation mask

        Returns:
            Dictionary containing segmentation quality metrics
        """
        try:
            # Ensure binary masks
            ref_mask = (reference_mask > 0).astype(np.uint8)
            pred_mask = (predicted_mask > 0).astype(np.uint8)

            # Ensure same size
            if ref_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask, (ref_mask.shape[1], ref_mask.shape[0]))

            # Calculate intersection and union
            intersection = np.logical_and(ref_mask, pred_mask).sum()
            union = np.logical_or(ref_mask, pred_mask).sum()

            # Calculate metrics
            iou = intersection / union if union > 0 else 0
            dice = 2 * intersection / (ref_mask.sum() + pred_mask.sum()) if (
                                                                                        ref_mask.sum() + pred_mask.sum()) > 0 else 0

            # Calculate precision and recall
            true_positives = intersection
            false_positives = pred_mask.sum() - true_positives
            false_negatives = ref_mask.sum() - true_positives

            precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0

            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics = {
                'iou': float(iou),
                'dice': float(dice),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }

            return metrics

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating segmentation metrics: {str(e)}")


# Example usage with DermoscopyImageProcessor
class EnhancedDermoscopyImageProcessor(DermoscopyImageProcessor):
    """
    Enhanced processor with quality assessment capabilities.
    """

    @classmethod
    async def process_and_assess(cls,
                                 input_file: UploadFile,
                                 reference_file: UploadFile = None) -> Dict[str, any]:
        """
        Process image and assess quality if reference is provided.

        Args:
            input_file: Input image to process
            reference_file: Optional reference image for quality assessment

        Returns:
            Dictionary containing processed images and quality metrics
        """
        # Process the input image
        results = await cls.process_image(input_file)

        # If reference image is provided, calculate quality metrics
        if reference_file:
            try:
                # Read reference image
                reference_contents = await reference_file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(reference_contents)
                    reference_path = temp_file.name

                try:
                    reference_img = cv2.imread(reference_path)
                    if reference_img is None:
                        raise HTTPException(status_code=400, detail="Could not read reference image")

                    # Initialize quality assessor
                    assessor = ImageQualityAssessor()

                    # Calculate quality metrics for hair removal
                    hair_removal_metrics = assessor.calculate_metrics(
                        reference_img,
                        cv2.imdecode(np.frombuffer(results['hair_removed'], np.uint8), cv2.IMREAD_COLOR)
                    )

                    # Calculate segmentation quality metrics if reference mask is available
                    segmentation_metrics = None
                    if 'reference_mask' in results:
                        segmentation_metrics = assessor.assess_segmentation_quality(
                            cv2.imdecode(np.frombuffer(results['reference_mask'], np.uint8), cv2.IMREAD_GRAYSCALE),
                            cv2.imdecode(np.frombuffer(results['lesion_mask'], np.uint8), cv2.IMREAD_GRAYSCALE)
                        )

                    # Add metrics to results
                    results['quality_metrics'] = {
                        'hair_removal': hair_removal_metrics,
                        'segmentation': segmentation_metrics
                    }

                finally:
                    os.unlink(reference_path)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error in quality assessment: {str(e)}")

        return results


# FastAPI endpoint
async def process_and_assess_image(
        input_file: UploadFile = File(...),
        reference_file: UploadFile = None
) -> StreamingResponse:
    """
    Process dermoscopic image and assess quality if reference is provided.
    """
    try:
        results = await EnhancedDermoscopyImageProcessor.process_and_assess(
            input_file,
            reference_file
        )

        # If quality metrics were calculated, include them in response headers
        headers = {}
        if 'quality_metrics' in results:
            headers['X-Quality-Metrics'] = str(results['quality_metrics'])

        # Return the final segmented image with quality metrics in headers
        return StreamingResponse(
            io.BytesIO(results['final_segmented']),
            media_type="image/png",
            headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))