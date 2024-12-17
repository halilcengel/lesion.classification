import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import moments_central, moments
from scipy.ndimage import rotate
import cv2


def preprocess_segmented_image(segmented_image):
    """
    Convert a segmented image to a binary mask.
    """
    if len(segmented_image.shape) == 3:
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = segmented_image.copy()

    binary_mask = (gray > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return binary_mask


def calculate_shape_asymmetry(segmented_image):
    """
    Calculate shape asymmetry and return detailed intermediate results
    """
    binary_mask = preprocess_segmented_image(segmented_image)
    M = moments_central(binary_mask)
    theta = 0.5 * np.arctan2(2 * M[1, 1], (M[2, 0] - M[0, 2]))
    angle_degrees = np.degrees(theta)

    rotated_mask = rotate(binary_mask, angle_degrees, reshape=False)

    height, width = rotated_mask.shape
    center_y = height // 2

    # Get halves for visualization
    top_half = rotated_mask[:center_y, :].copy()
    bottom_half = rotated_mask[center_y:, :].copy()

    # Flip bottom half for comparison
    bottom_half_flipped = np.flip(bottom_half, axis=0)

    # Pad if necessary to match sizes
    if top_half.shape != bottom_half_flipped.shape:
        target_height = max(top_half.shape[0], bottom_half_flipped.shape[0])
        if top_half.shape[0] < target_height:
            pad_height = target_height - top_half.shape[0]
            top_half = np.pad(top_half, ((0, pad_height), (0, 0)))
        if bottom_half_flipped.shape[0] < target_height:
            pad_height = target_height - bottom_half_flipped.shape[0]
            bottom_half_flipped = np.pad(bottom_half_flipped, ((0, pad_height), (0, 0)))

    # Calculate difference image
    difference = np.abs(top_half - bottom_half_flipped)

    # Calculate asymmetry scores
    horizontal_diff = abs(np.sum(top_half) - np.sum(bottom_half))
    total_area = np.sum(rotated_mask)
    horizontal_asymmetry = (horizontal_diff / total_area) * 100

    is_asymmetric = horizontal_asymmetry > 2.0

    return {
        'original': binary_mask,
        'rotated': rotated_mask,
        'top_half': top_half,
        'bottom_half': bottom_half,
        'bottom_half_flipped': bottom_half_flipped,
        'difference': difference,
        'asymmetry_score': horizontal_asymmetry,
        'is_asymmetric': is_asymmetric
    }


def visualize_detailed_asymmetry(image_path):
    """
    Create detailed visualization of asymmetry calculation steps
    """
    # Read and process image
    segmented_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = calculate_shape_asymmetry(segmented_image)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    fig.suptitle(f"Asymmetry Score: {results['asymmetry_score']:.2f}%")

    # Plot each step
    axes[0].imshow(results['original'], cmap='gray')
    axes[0].set_title('(a) Original')

    axes[1].imshow(results['rotated'], cmap='gray')
    axes[1].set_title('(b) Rotated')

    axes[2].imshow(results['top_half'], cmap='gray')
    axes[2].set_title('(c) Upper Half')

    axes[3].imshow(results['bottom_half'], cmap='gray')
    axes[3].set_title('(d) Lower Half')

    axes[4].imshow(results['bottom_half_flipped'], cmap='gray')
    axes[4].set_title('(e) Folded Lower Half')

    axes[5].imshow(results['difference'], cmap='gray')
    axes[5].set_title('(f) Difference')

    # Remove axes for cleaner visualization
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = visualize_detailed_asymmetry('segmentation_v2_masks/ISIC_0000042_segmented.png')
    plt.show()