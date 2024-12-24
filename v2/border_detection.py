import numpy as np
import cv2
from scipy import ndimage


def detect_border(segmented_image):
    """
    Detect single pixel border of a segmented skin lesion image.

    Args:
        segmented_image: Binary segmented image where lesion is white (255)
                        and background is black (0)

    Returns:
        border_image: Binary image with only the border pixels
    """
    # Ensure binary image
    binary_image = np.uint8(segmented_image > 0) * 255

    # Create structuring element for morphological operations
    # Paper specifies 2-pixel diameter circular SE
    diameter = 2
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

    # Perform morphological erosion
    eroded = cv2.erode(binary_image, se, iterations=1)

    # Get border through morphological gradient
    # (Subtraction of eroded image from original)
    border = binary_image - eroded

    # Clean up border
    # Remove isolated pixels and ensure single pixel width
    border = ndimage.binary_fill_holes(border)
    border = np.uint8(border) * 255

    return border


def compute_border_irregularity(border_image, num_segments=8):
    """
    Compute border irregularity by splitting border into segments
    and calculating Katz fractal dimension.

    Args:
        border_image: Binary image containing only border pixels
        num_segments: Number of segments to divide border into (default 8)

    Returns:
        irregularity_score: Score from 0-8 based on irregular segments
    """
    # Get border coordinates
    border_coords = np.column_stack(np.where(border_image > 0))

    # Get centroid
    centroid = np.mean(border_coords, axis=0)

    # Convert to polar coordinates
    r = np.sqrt(np.sum((border_coords - centroid) ** 2, axis=1))
    theta = np.arctan2(border_coords[:, 0] - centroid[0],
                       border_coords[:, 1] - centroid[1])

    # Sort by angle
    sorted_idx = np.argsort(theta)
    r = r[sorted_idx]
    theta = theta[sorted_idx]

    # Split into segments
    segment_size = len(r) // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size if i < num_segments - 1 else len(r)
        segments.append((r[start:end], theta[start:end]))

    # Compute Katz fractal dimension for each segment
    irregularity_score = 0
    for r_seg, theta_seg in segments:
        # Convert back to cartesian coordinates
        x = r_seg * np.cos(theta_seg)
        y = r_seg * np.sin(theta_seg)
        points = np.column_stack((x, y))

        # Compute fractal dimension
        d = compute_katz_fd(points)

        # Add 1 to score if segment is irregular
        # Paper specifies >10% deviation from circle's FD indicates irregularity
        if abs(d - np.pi / 2) > 0.1 * np.pi / 2:
            irregularity_score += 1

    return irregularity_score


def compute_katz_fd(points):
    """
    Compute Katz fractal dimension of a curve defined by points.

    Args:
        points: Nx2 array of point coordinates

    Returns:
        fd: Fractal dimension
    """
    # Compute distances between successive points
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))

    # Total length of curve
    L = np.sum(dists)

    # Maximum distance between first point and any other point
    d = np.max(np.sqrt(np.sum((points - points[0]) ** 2, axis=1)))

    # Number of steps
    n = len(points) - 1

    # Compute Katz FD
    # FD = log(n) / (log(d/L) + log(n))
    fd = np.log(n) / (np.log(d / L) + np.log(n))

    return fd

if __name__ == "__main__":
    # Read the mask image
    mask = cv2.imread('segmentation_v2_masked_images/ISIC_0000042_masked.png', cv2.IMREAD_GRAYSCALE)

    # Detect border
    border = detect_border(mask)

    # Compute irregularity score
    score = compute_border_irregularity(border)

    print("Irregularity Score:", score)

    # Save or display results
    cv2.imwrite('border.png', border)