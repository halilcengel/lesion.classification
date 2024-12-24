import cv2
import numpy as np


def detect_border(image, border_thickness=1):
    """
    Detect borders in a binary segmentation mask.

    Parameters:
        mask: numpy.ndarray
            Binary segmentation mask (2D array with 0s and 1s or 0s and 255s)
        border_thickness: int
            Thickness of the border to detect (default: 1)

    Returns:
        numpy.ndarray: Binary image with only the borders
    """
    # Ensure mask is binary
    if image.max() > 1:
        image = image / 255.0

    # Convert to uint8
    image = (image * 255).astype(np.uint8)

    # Method 1: Using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=border_thickness)
    border = cv2.subtract(image, erosion)

    # Alternative Method 2: Using contours
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # border = np.zeros_like(mask)
    # cv2.drawContours(border, contours, -1, (255,255,255), border_thickness)

    return border



def split_border_into_segments(border_image, num_segments=8):
    """
    Split border into equal segments for irregularity analysis.
    Paper specifies 8 equal segments.
    """
    # Get border points
    y_coords, x_coords = np.nonzero(border_image)
    points = np.column_stack((x_coords, y_coords))

    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Convert to polar coordinates
    r = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
    theta = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort by angle
    idx = np.argsort(theta)
    r = r[idx]
    theta = theta[idx]
    points = points[idx]

    # Split into equal segments
    segments = []
    points_per_segment = len(points) // num_segments

    for i in range(num_segments):
        start_idx = i * points_per_segment
        end_idx = start_idx + points_per_segment if i < num_segments - 1 else len(points)
        segment_points = points[start_idx:end_idx]
        segments.append(segment_points)

    return segments


def compute_katz_fd(points):
        """
        Compute Katz fractal dimension of a curve defined by points.

        Args:
            points: Nx2 array of point coordinates

        Returns:
            fd: Fractal dimension
        """
        dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))

        L = np.sum(dists)

        d = np.max(np.sqrt(np.sum((points - points[0]) ** 2, axis=1)))

        n = len(points) - 1

        # FD = log(n) / (log(d/L) + log(n))
        fd = np.log(n) / (np.log(d / L) + np.log(n))

        return fd


def compute_border_irregularity(segments):
    """
    Compute border irregularity score (0-8) as specified in paper.
    Score of 1 is added for each segment deviating >10% from circle's FD.
    """

    # Reference FD for a circle is π/2
    circle_fd = np.pi / 2
    irregularity_threshold = 0.1 * circle_fd

    # Count irregular segments
    irregularity_score = 0
    for segment in segments:
        if len(segment) > 2:  # Need at least 3 points for meaningful FD
            fd = compute_katz_fd(segment)
            if abs(fd - circle_fd) > irregularity_threshold:
                irregularity_score += 1

    return irregularity_score