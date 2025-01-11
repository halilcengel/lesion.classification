import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, filters, util
from skimage.transform import rotate


def detect_pigment_network(img_gray, num_directions=6, length=15):
    """
    Optimized version of pigment network detection with fewer directions and pre-computed SEs.
    """
    # Pre-compute structuring elements for all angles
    angles = np.linspace(0, 180, num_directions, endpoint=False)
    structuring_elements = []
    
    # Create base structuring element once
    base_se = np.zeros((length, 1), dtype=bool)
    base_se[:, 0] = True

    # Pre-compute all rotated structuring elements
    for angle in angles:
        rotated = rotate(base_se, angle, resize=True, preserve_range=True)
        structuring_elements.append(rotated > 0.5)

    # Initialize response accumulator
    max_response = np.zeros_like(img_gray, dtype=float)

    # Apply bottom-hat filter with each pre-computed SE
    for se in structuring_elements:
        response = morphology.black_tophat(img_gray, se)
        max_response = np.maximum(max_response, response)

    # Normalize and threshold the result
    max_response = (max_response - max_response.min()) / (max_response.max() - max_response.min() + 1e-8)
    threshold = filters.threshold_otsu(max_response)
    final_detection = max_response > threshold

    return final_detection


def detect_pigment_network_v2(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply directional filters (Gabor filter bank)
    theta = np.pi / 4  # 45 degrees
    kernel = cv2.getGaborKernel((21, 21), sigma=3, theta=theta, lambd=10, gamma=0.5)
    filtered = cv2.filter2D(enhanced, -1, kernel)

    # Threshold to get binary mask
    _, mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask


if __name__ == "__main__":
    # Example usage (files need to be replaced with actual paths):
    img_gray = io.imread('../images/rapor/segmentation/segmented_images/ISIC_0000148_masked.png', as_gray=True)

    pigment_network = detect_pigment_network(img_gray)

    pigment_network_uint8 = (pigment_network * 255).astype(np.uint8)
    cv2.imwrite('../images/rapor/pigment_network/ISIC_0000148_pigment_network.png', pigment_network_uint8)

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