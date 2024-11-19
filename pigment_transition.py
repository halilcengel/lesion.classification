import numpy as np
from typing import Dict
import cv2


def calculate_pigment_transition(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate pigment transition features from image

    Args:
        image: Original RGB image
        mask: Binary mask of the lesion

    Returns:
        Dictionary containing pigment transition measurements
    """
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Calculate gradient using Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Apply mask to get only lesion area
        lesion_area = gray[mask > 0]
        gradient_area = gradient_magnitude[mask > 0]

        # Calculate statistics
        mean_intensity = np.mean(lesion_area)
        std_intensity = np.std(lesion_area)
        mean_gradient = np.mean(gradient_area)
        std_gradient = np.std(gradient_area)

        # Create visualization
        vis_mask = np.zeros_like(gray, dtype=np.uint8)

        # Normalize gradient for visualization
        norm_gradient = (gradient_magnitude - gradient_magnitude.min()) / (
                gradient_magnitude.max() - gradient_magnitude.min()) * 255
        norm_gradient = norm_gradient.astype(np.uint8)

        # Apply mask to gradient visualization
        vis_mask[mask > 0] = norm_gradient[mask > 0]

        # Create color visualization
        vis_color = cv2.applyColorMap(vis_mask, cv2.COLORMAP_JET)

        # Add contour to visualization
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_color, contours, -1, (255, 255, 255), 2)

        # Save visualization
        temp_path = "images/pigment_transition.png"
        cv2.imwrite(temp_path, vis_color)

        return {
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity),
            "mean_gradient": float(mean_gradient),
            "std_gradient": float(std_gradient),
            "visualization_path": temp_path
        }
    except Exception as e:
        return {
            "mean_intensity": 0,
            "std_intensity": 0,
            "mean_gradient": 0,
            "std_gradient": 0,
        }
