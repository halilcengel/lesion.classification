import cv2
import numpy as np
# use


def detect_mask_border(mask, border_thickness=1):
    """
    Detect borders in a binary segmentation mask.

    Parameters:
        mask: numpy.ndarray
            Binary segmentation mask (2D array with 0s and 1s or 0s and 255s)
        border_thickness: int
            Thickness of the border to detect (default: 1)

    Returns:
        numpy.ndarray: Binary image with only the borders
    """
    # Ensure mask is binary
    if mask.max() > 1:
        mask = mask / 255.0

    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)

    # Method 1: Using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=border_thickness)
    border = cv2.subtract(mask, erosion)

    # Alternative Method 2: Using contours
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # border = np.zeros_like(mask)
    # cv2.drawContours(border, contours, -1, (255,255,255), border_thickness)

    return border


# Example usage
if __name__ == "__main__":
    # Read the mask image
    mask = cv2.imread('segmentation_v2_masks/ISIC_0000042_segmented.png', cv2.IMREAD_GRAYSCALE)

    # Detect borders
    border = detect_mask_border(mask, border_thickness=2)

    # Save or display results
    cv2.imwrite('mask_border.png', border)

    # Optional: Overlay border on original image in color
    colored_border = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_border[border > 0] = [0, 0, 255]  # Red border

    # Blend with original image if needed
    # original = cv2.imread('original_image.png')
    # result = cv2.addWeighted(original, 1.0, colored_border, 0.5, 0)