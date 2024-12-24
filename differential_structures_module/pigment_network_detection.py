import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, filters, util
from skimage.transform import rotate


def detect_pigment_network(img_gray, num_directions=12, length=15):
    """
    Detect pigment network in a dermoscopic image using:
    1) Morphological bottom-hat filtering with a linear structuring element (SE).
    2) Directional filtering at multiple orientations.
    3) Combining results to emphasize the reticular pattern.

    Parameters:
    -----------
    img_gray      : np.ndarray
        Grayscale dermoscopic image.
    img_mask      : np.ndarray
        Binary mask to focus on the lesion area (1 = lesion, 0 = outside).
    num_directions: int
        Number of directional filters to apply (covering [0, pi] in increments).
    length        : int
        Length of the linear structuring element (SE).

    Returns:
    --------
    final_detection: np.ndarray (same shape as img_gray)
        A float or integer array representing the combined response emphasizing pigment network.
    """

    # Step 1: Mask the input to limit computations to lesion area
    img_lesion = img_gray

    # Step 2: Apply morphological bottom-hat filter in various orientations
    # The black_tophat function from scikit-image performs bottom-hat filtering.
    # We'll rotate a linear SE and keep track of the maximum response across orientations.
    angles = np.linspace(0, 180, num_directions, endpoint=False)
    responses = []

    for angle in angles:
        # Create a linear SE (e.g., a 'rectangle' of size (1, length)),
        # then rotate it to the desired angle via rotate(...).
        # Because morphology.selem doesn't provide direct rotation, we'll rotate
        # a small binary image representing our SE, then convert it to a structuring element.

        # Create a base "line" structuring element of shape (length x 1)
        base_se = np.zeros((length, 1), dtype=bool)
        base_se[:, 0] = True  # vertical line

        # Convert to a 2D image large enough to rotate
        # (make it a square for proper rotation)
        max_dim = length
        se_image = np.zeros((max_dim, max_dim), dtype=bool)
        # Place the base SE in the center
        center = max_dim // 2
        start = center - (length // 2)
        se_image[start:start + length, center] = True

        # Rotate the SE image by 'angle'
        se_rotated_img = rotate(se_image.astype(float), angle, resize=False, order=0, preserve_range=True) > 0.5

        # Convert rotated image back to a structuring element
        se_rotated = morphology.binary_dilation(se_rotated_img)

        # Perform bottom-hat filtering using the rotated structuring element
        # black_tophat returns bright output for darker structures on a lighter background
        response = morphology.black_tophat(img_lesion, footprint=se_rotated)
        responses.append(response)

    # Step 3: Combine the responses by taking the maximum across all directions
    # (This helps capture reticular patterns that appear in multiple orientations.)
    combined_response = np.max(np.stack(responses, axis=-1), axis=-1)

    # Step 4: (Optional) Post-process to emphasize thin grid lines
    # For example, we can apply a threshold or further enhancing filters
    # Here, we apply a simple Otsu threshold to highlight network lines:
    thresh_val = filters.threshold_otsu(combined_response[combined_response > 0]) \
        if np.any(combined_response > 0) else 0
    final_detection = combined_response >= thresh_val

    return final_detection


if __name__ == "__main__":
    # Example usage (files need to be replaced with actual paths):
    img_gray = io.imread('../segmentation_v2_masked_images/ISIC_0000042_masked.png', as_gray=True)

    pigment_network = detect_pigment_network(img_gray, num_directions=12, length=15)

    # Visualize results:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Grayscale Dermoscopic Image')
    axes[0].axis('off')

    axes[1].imshow(pigment_network)
    axes[1].set_title('Detected Pigment Network')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()