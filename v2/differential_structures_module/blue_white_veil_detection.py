import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float


def detect_blue_white_veil(image_path, reference_blue=(0, 0, 255), reference_white=(255, 255, 255)):
    """
    Detects blue-white veil regions in a segmented dermoscopic image.

    :param image_path: Path to the segmented dermoscopic image
    :param reference_blue: Reference color in BGR (OpenCV format) for the "blue" range
    :param reference_white: Reference color in BGR (OpenCV format) for the "white" range
    :return: A mask highlighting regions identified as "blue-white veil"
    """
    # Read the segmented image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("Could not read the image at the specified path.")

    # Convert to float image for SLIC
    float_img = img_as_float(original_img)

    # Apply SLIC superpixel segmentation
    superpixels = slic(
        float_img,
        n_segments=300,  # Number of superpixels
        compactness=10,  # Balances color proximity and space proximity
        start_label=1
    )

    # Create an array to store the "blue-white veil" detection
    blue_white_veil_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)

    # For each superpixel, check if it's part of the lesion and measure color distances
    for label_val in np.unique(superpixels):
        # Get the region corresponding to this superpixel
        region_mask = (superpixels == label_val)

        # Calculate the mean color in BGR in this superpixel region
        region_pixels = original_img[region_mask]

        # Skip if the region is empty (black background)
        if len(region_pixels) == 0 or np.all(region_pixels == 0):
            continue

        mean_b = np.mean(region_pixels[:, 0])
        mean_g = np.mean(region_pixels[:, 1])
        mean_r = np.mean(region_pixels[:, 2])

        # Compare the mean color to reference values
        dist_blue = np.sqrt(
            (mean_b - reference_blue[0]) ** 2 +
            (mean_g - reference_blue[1]) ** 2 +
            (mean_r - reference_blue[2]) ** 2
        )
        dist_white = np.sqrt(
            (mean_b - reference_white[0]) ** 2 +
            (mean_g - reference_white[1]) ** 2 +
            (mean_r - reference_white[2]) ** 2
        )

        # Mark regions that are close to either "blue" or "white" reference
        if dist_blue < 150 or dist_white < 150:  # Adjust threshold as needed
            blue_white_veil_mask[region_mask] = 255

    return blue_white_veil_mask


# Example usage
if __name__ == "__main__":
    # Path to the segmented image
    segmented_image = '../segmentation_v2_masked_images/ISIC_0000042_masked.png'

    bw_veil_mask = detect_blue_white_veil(
        image_path=segmented_image,
        reference_blue=(80, 50, 200),  # Example reference for "blue" in BGR
        reference_white=(220, 220, 220)  # Example reference for "white"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(bw_veil_mask, cmap='gray')
    axes[0].set_title('Blue White Veil Mask')
    axes[0].axis('off')

    plt.tight_layout()
    plt.show()