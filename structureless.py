import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, filters, util


def detect_structureless_area(img_gray):
    """
    Detect structureless area from a dermoscopic grayscale image.

    Parameters:
    -----------
    img_gray : ndarray
        Grayscale dermoscopic image.

    Returns:
    --------
    structureless_mask : ndarray (boolean)
        Binary mask for detected structureless area.
    """
    # Step 1: Mask the grayscale image to focus only on the lesion
    img_lesion = img_gray

    # Step 2: Perform morphological closing using a circular structuring element
    se_radius = 2
    se = morphology.disk(se_radius)
    img_closed = morphology.closing(img_lesion, se)

    # Step 3: Subtract the complement of original lesion image from the closed image
    img_complement = util.invert(img_lesion)
    img_diff = img_closed - img_complement

    # Add debugging information
    print(f"Min value in img_diff: {np.min(img_diff)}")
    print(f"Max value in img_diff: {np.max(img_diff)}")
    print(f"Number of positive values: {np.sum(img_diff > 0)}")

    # Step 4: Modified thresholding approach
    positive_diff = img_diff[img_diff > 0]
    if len(positive_diff) > 0:
        threshold_val = filters.threshold_otsu(positive_diff)
    else:
        # Fallback: use a simple statistical threshold if no positive values
        threshold_val = np.mean(img_diff) + np.std(img_diff)

    structureless_mask = (img_diff >= threshold_val)

    # Optional: Clean up the mask using morphological operations
    structureless_mask = morphology.remove_small_objects(structureless_mask, min_size=50)
    structureless_mask = morphology.remove_small_holes(structureless_mask, area_threshold=50)

    return structureless_mask


if __name__ == "__main__":
    # Load the image
    img_gray = io.imread('segmentation_v2_masked_images/ISIC_0000042_masked.png', as_gray=True)

    if img_gray.max() > 1:
        img_gray = img_gray / 255.0

    structureless = detect_structureless_area(img_gray)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[0].axis('off')

    axes[1].imshow(structureless)
    axes[1].set_title('Structureless Areas')
    axes[1].axis('off')


    plt.tight_layout()
    plt.show()