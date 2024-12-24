import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology
from skimage.morphology import disk, reconstruction
#use that
def detect_dots_globules(
    img,
    threshold_sensitivity=0.02,
    rmin=2,
    rmax=8,
    radius_step=2
):
    """
    Detect dots and globules in a grayscale image using a bottom-hat filter approach.
    Parameters:
        img_gray (ndarray)       : Grayscale input image.
        img_segmented (ndarray)  : Binary segmentation mask for region of interest.
        threshold_sensitivity    : Sensitivity factor for threshold calculation.
        rmin, rmax               : Min/Max radius for circular structuring element.
        radius_step              : Increment step for increasing radius.
    Returns:
        ndarray: Binary mask containing detected dots/globules.
    """

    # Step-1: Mask the grayscale image so we only process relevant areas
    img_masked = img

    # Prepare an accumulator for the final detections
    height, width = img.shape
    final_detection = np.zeros((height, width), dtype=bool)

    # Step-3: Vary the circular structuring element from rmin to rmax
    for r in range(rmin, rmax + 1, radius_step):
        # 3.2: Create a circular structuring element (disk used here as approximation)
        se = disk(r)

        # 3.3: Apply morphological closing
        img_closed = morphology.closing(img_masked, se)

        # 3.4: Morphological reconstruction
        # Note: in skimage, reconstruction is done with "seed" under "mask".
        # Here we reconstruct the closed image `img_closed` under the original masked image.
        seed = img_closed
        mask = img_masked
        img_reconstructed = reconstruction(seed, mask, method='erosion')

        # 3.5: Bottom-hat filter = reconstructed image (larger structures) - original
        img_bottom_hat = img_reconstructed - img

        # 3.6 and 3.7: Find min/max intensities in the extracted bottom-hat regions
        extracted_pixels = img_bottom_hat[img_bottom_hat > 0]
        if extracted_pixels.size == 0:
            # If no pixels, continue to next radius
            continue
        Id_min, Id_max = extracted_pixels.min(), extracted_pixels.max()

        # 3.8: Threshold calculation
        Th = threshold_sensitivity * (Id_max - Id_min) + Id_min

        # 3.9: Segment objects above threshold
        detection_mask = img_bottom_hat > Th

        # Combine current detection with final detection
        final_detection |= detection_mask

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
        radius_step=2
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