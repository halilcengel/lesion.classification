import numpy as np
import cv2
from typing import Dict
import matplotlib.pyplot as plt


def calculate_color_variation(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate color variation in CIE-LAB color space

    Args:
        image: RGB image
        mask: Binary mask of the lesion

    Returns:
        Dictionary containing color variation measurements
    """
    try:
        # Convert RGB to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split LAB channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply mask to get lesion area only
        l_lesion = l_channel[mask > 0]
        a_lesion = a_channel[mask > 0]
        b_lesion = b_channel[mask > 0]

        # Calculate mean values for each channel
        l_mean = np.mean(l_lesion)
        a_mean = np.mean(a_lesion)
        b_mean = np.mean(b_lesion)

        # Calculate standard deviations
        l_std = np.std(l_lesion)
        a_std = np.std(a_lesion)
        b_std = np.std(b_lesion)

        # Calculate color differences for each pixel
        delta_l = l_lesion - l_mean
        delta_a = a_lesion - a_mean
        delta_b = b_lesion - b_mean

        # Calculate total color difference (ΔE)
        delta_e = np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2)
        mean_delta_e = np.mean(delta_e)
        max_delta_e = np.max(delta_e)

        # Create visualization
        # Create a figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

        # Original image with mask
        masked_image = image.copy()
        masked_image[mask == 0] = 0
        ax1.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Lesion')
        ax1.axis('off')

        # L channel variation
        ax2.hist(l_lesion, bins=50, color='gray')
        ax2.axvline(l_mean, color='r', linestyle='dashed', linewidth=2)
        ax2.set_title('L Channel Distribution')

        # a channel variation
        ax3.hist(a_lesion, bins=50, color='green')
        ax3.axvline(a_mean, color='r', linestyle='dashed', linewidth=2)
        ax3.set_title('a Channel Distribution')

        # b channel variation
        ax4.hist(b_lesion, bins=50, color='blue')
        ax4.axvline(b_mean, color='r', linestyle='dashed', linewidth=2)
        ax4.set_title('b Channel Distribution')

        # Save visualization
        temp_path = "images/color_variation.png"
        plt.tight_layout()
        plt.savefig(temp_path)
        plt.close()

        # Create color map visualization
        color_map = np.zeros_like(image)
        color_map[mask > 0] = cv2.applyColorMap(
            cv2.normalize(
                delta_e.reshape(-1, 1),
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8),
            cv2.COLORMAP_JET
        )[0]

        # Save color map
        color_map_path = "images/color_map.png"
        cv2.imwrite(color_map_path, color_map)

        return {
            "l_mean": float(l_mean),
            "a_mean": float(a_mean),
            "b_mean": float(b_mean),
            "l_std": float(l_std),
            "a_std": float(a_std),
            "b_std": float(b_std),
            "mean_color_difference": float(mean_delta_e),
            "max_color_difference": float(max_delta_e),
            "visualization_path": temp_path,
            "color_map_path": color_map_path
        }

    except Exception as e:
        raise ValueError(f"Error calculating color variation: {str(e)}")