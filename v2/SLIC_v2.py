import numpy as np
from sklearn.cluster import KMeans
import cv2


def create_superpixels(image, num_segments=200, compactness=10.0):
    """
    Generate superpixels using SLIC-like algorithm

    Parameters:
    image: Input RGB image (numpy array)
    num_segments: Number of desired superpixels
    compactness: Trade-off between color similarity and spatial proximity

    Returns:
    labels: Label map where each pixel is assigned to a superpixel
    """

    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    height, width = image.shape[:2]

    # Initialize grid
    grid_step = int((height * width / num_segments) ** 0.5)

    # Create feature matrix
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Stack spatial and color features
    features = np.stack([
        X.ravel(),
        Y.ravel(),
        lab_image[:, :, 0].ravel() * (compactness / grid_step),
        lab_image[:, :, 1].ravel() * (compactness / grid_step),
        lab_image[:, :, 2].ravel() * (compactness / grid_step)
    ], axis=1)

    # Perform clustering
    kmeans = KMeans(n_clusters=num_segments, n_init=1)
    labels = kmeans.fit_predict(features)

    # Reshape labels back to image dimensions
    labels = labels.reshape(height, width)

    return labels


def draw_superpixel_boundaries(image, labels):
    """
    Draw boundaries of superpixels on the image

    Parameters:
    image: Original RGB image
    labels: Superpixel label map

    Returns:
    result: Image with superpixel boundaries drawn
    """
    result = image.copy()

    # Find boundaries
    dx = np.diff(labels, axis=1)
    dy = np.diff(labels, axis=0)

    # Create boundary mask
    boundary_mask = np.zeros_like(labels, dtype=bool)
    boundary_mask[:, :-1] |= dx != 0
    boundary_mask[:, 1:] |= dx != 0
    boundary_mask[:-1, :] |= dy != 0
    boundary_mask[1:, :] |= dy != 0

    # Draw boundaries in black
    result[boundary_mask] = [0, 0, 0]

    return result


def color_superpixels(image, labels):
    """
    Color each superpixel with its mean color

    Parameters:
    image: Original RGB image
    labels: Superpixel label map

    Returns:
    result: Image with colored superpixels
    """
    result = np.zeros_like(image, dtype=np.float32)

    for label in range(labels.max() + 1):
        mask = labels == label
        if mask.any():
            # Calculate mean color for each channel
            for channel in range(3):
                result[:, :, channel][mask] = image[:, :, channel][mask].mean()

    return result.astype(np.uint8)


# Example usage
def process_image(image_path, num_segments=32, compactness=10.0):
    """
    Process an image to create and visualize superpixels

    Parameters:
    image_path: Path to input image
    num_segments: Number of desired superpixels
    compactness: Trade-off between color similarity and spatial proximity

    Returns:
    tuple: (original image, superpixel boundaries, colored superpixels)
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate superpixels
    labels = create_superpixels(image, num_segments, compactness)

    # Draw boundaries
    boundaries = draw_superpixel_boundaries(image, labels)

    # Color superpixels
    colored = color_superpixels(image, labels)

    return image, boundaries, colored


if __name__ == "__main__":
    # Example usage
    image_path = 'segmentation_v2_masked_images/ISIC_0000042_masked.png'
    original, boundaries, colored = process_image(image_path)

    # Display results
    cv2.imshow('Original', original)
    cv2.imshow('Superpixel Boundaries', boundaries)
    cv2.imshow('Colored Superpixels', cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
