import math

import math

import cv2

import math
import numpy as np


def compute_katz_fractal_dimension(points):
    """
    Computes the Katz Fractal Dimension (KD) for a set of 2D points.
    points: a NumPy array or list of (x, y) coordinates.
    Returns a float representing the fractal dimension.
    """
    # If using NumPy, checking points.size == 0 handles the 'truth value' issue
    if isinstance(points, np.ndarray):
        if points.size == 0:
            return 1.0
    else:
        if len(points) == 0:
            return 1.0

    # Calculate P: total path length
    P = 0.0
    for i in range(len(points) - 1):
        P += math.dist(points[i], points[i + 1])

    # Compute diameter (d): max distance from the first point
    d = 0.0
    for i in range(1, len(points)):
        d = max(d, math.dist(points[0], points[i]))

    if P <= 0 or d <= 0:
        return 1.0

    # Average step a: total length / number of segments
    a = P / (len(points) - 1)
    if a == 0:
        return 1.0

    # Number of steps n
    n = P / a
    if n <= 0:
        return 1.0

    # Handle potential division by zero in the denominator
    denominator = math.log(d / P) + math.log(n)
    if math.isclose(denominator, 0.0, abs_tol=1e-9):
        return 1.0

    return math.log(n) / denominator

def border_irregularity_score(border_points, segment_count=8, deviation_threshold=0.1):
    """
    Subdivides the border into 'segment_count' segments, computes Katz
    Fractal Dimension for each, and assigns 1 point for each segment
    whose dimension deviates more than 'deviation_threshold' from 1.0.
    Returns the total irregularity score.
    """
    # Basic check
    if len(border_points) < segment_count:
        # For fewer points than segments, return 0 or handle differently
        return 0

    # Split the border points into segments (approx. equal-length subdivisions)
    segment_size = len(border_points) // segment_count
    segments = []

    start_idx = 0
    for _ in range(segment_count - 1):
        end_idx = start_idx + segment_size
        segments.append(border_points[start_idx:end_idx])
        start_idx = end_idx
    # Add the remaining points to the last segment
    segments.append(border_points[start_idx:])

    # Calculate irregularity score
    score = 0
    for seg in segments:
        if len(seg) < 2:
            continue
        kd = compute_katz_fractal_dimension(seg)

        # Check deviation from 1.0 by more than deviation_threshold
        if (kd < (1.0 - deviation_threshold)) or (kd > (1.0 + deviation_threshold)):
            score += 1

    return score

def detect_border(image):
    """
    Detect single pixel border of a skin lesion image using the paper's specifications.
    Handles both grayscale and RGB input images.
    Paper specifies a 2-pixel diameter circular SE and morphological operations.

    Args:
        image: Input image (grayscale or RGB)

    Returns:
        border_image: Binary image containing only the border pixels
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to get binary image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create circular structuring element with 2-pixel diameter as specified in paper
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Apply morphological erosion (specified in paper)
    eroded = cv2.erode(binary, se, iterations=1)

    # Morphological gradient: subtract eroded from original
    border = cv2.subtract(binary, eroded)

    return border


# Example usage:
if __name__ == "__main__":
    image = cv2.imread('segmentation_v2_masked_images/ISIC_0024436_masked.png', cv2.IMREAD_GRAYSCALE)
    border = detect_border(image)

    irregularity = border_irregularity_score(border, segment_count=8, deviation_threshold=0.1)
    print(f"Border Irregularity Score: {irregularity}")