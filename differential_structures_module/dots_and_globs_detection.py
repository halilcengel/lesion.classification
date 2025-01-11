import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology
from skimage.morphology import disk, reconstruction

def detect_dots_globules(
    img,
    threshold_sensitivity=0.02,
    rmin=2,
    rmax=8,
    radius_step=3
):
    """
    Optimized version of dots and globules detection.
    Parameters:
        img_gray (ndarray)       : Grayscale input image.
        img_segmented (ndarray)  : Binary segmentation mask for region of interest.
        threshold_sensitivity    : Sensitivity factor for threshold calculation.
        rmin, rmax               : Min/Max radius for circular structuring element.
        radius_step              : Increment step for increasing radius.
    Returns:
        ndarray: Binary mask containing detected dots/globules.
    """
    height, width = img.shape
    final_detection = np.zeros((height, width), dtype=bool)
    
    # Pre-compute all structuring elements
    structuring_elements = [disk(r) for r in range(rmin, rmax + 1, radius_step)]
    
    # Process image with each structuring element
    for se in structuring_elements:
        # Apply morphological closing
        img_closed = morphology.closing(img, se)
        
        # Compute difference directly instead of using reconstruction
        img_diff = img_closed - img
        
        # Threshold the difference
        if img_diff.any():
            thresh = np.percentile(img_diff[img_diff > 0], 75) * threshold_sensitivity
            detection = img_diff > thresh
            final_detection |= detection
    
    # Clean up the result with a small opening to remove noise
    final_detection = morphology.opening(final_detection, disk(1))
    
    return final_detection

# Example usage:
if __name__ == "__main__":
    # Load or create your grayscale and segmentation images:
    # (Below is just a placeholder; replace with actual images)
    img = io.imread('../segmentation_v2_masked_images/ISIC_0000042_masked.png', as_gray=True)

    # Detect dots and globules
    detected_mask = detect_dots_globules(
        img,
        threshold_sensitivity=0.02,
        rmin=2,
        rmax=8,
        radius_step=3
    )

    # Visualize the result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Grayscale Image')
    axes[0].axis('off')

    axes[1].imshow(detected_mask, cmap='gray')
    axes[1].set_title('Detected Dots & Globules')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()