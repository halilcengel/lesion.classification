from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import math
from typing import Dict, Tuple


def calculate_diameters(contour: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculate greatest and smallest diameters passing through centroid

    Args:
        contour: Contour points of the lesion

    Returns:
        Tuple containing greatest diameter, smallest diameter, and their endpoint points
    """
    # Find centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        raise ValueError("Invalid contour - zero area")

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])

    # Convert contour to points array
    points = contour.reshape(-1, 2)

    # Calculate distances from centroid to all boundary points
    distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))

    # Find the farthest point from centroid
    max_idx = np.argmax(distances)
    farthest_point = points[max_idx]

    # Get the angle of the farthest point
    angle = np.arctan2(farthest_point[1] - cy, farthest_point[0] - cx)

    # Find the opposite point (180 degrees)
    opposite_angle = angle + np.pi
    distances_to_opposite = np.abs(np.arctan2(points[:, 1] - cy, points[:, 0] - cx) - opposite_angle)
    distances_to_opposite = np.minimum(distances_to_opposite, 2 * np.pi - distances_to_opposite)
    opposite_idx = np.argmin(distances_to_opposite)
    opposite_point = points[opposite_idx]

    # Calculate greatest diameter
    greatest_diameter = np.linalg.norm(farthest_point - opposite_point)
    gd_points = np.array([farthest_point, opposite_point])

    # Find smallest diameter (perpendicular to greatest diameter)
    perpendicular_angle = angle + np.pi / 2
    distances_to_perp = np.abs(np.arctan2(points[:, 1] - cy, points[:, 0] - cx) - perpendicular_angle)
    distances_to_perp = np.minimum(distances_to_perp, 2 * np.pi - distances_to_perp)
    perp_idx = np.argmin(distances_to_perp)
    perp_point = points[perp_idx]

    # Find opposite point for smallest diameter
    opposite_perp_angle = perpendicular_angle + np.pi
    distances_to_opposite_perp = np.abs(np.arctan2(points[:, 1] - cy, points[:, 0] - cx) - opposite_perp_angle)
    distances_to_opposite_perp = np.minimum(distances_to_opposite_perp, 2 * np.pi - distances_to_opposite_perp)
    opposite_perp_idx = np.argmin(distances_to_opposite_perp)
    opposite_perp_point = points[opposite_perp_idx]

    # Calculate smallest diameter
    smallest_diameter = np.linalg.norm(perp_point - opposite_perp_point)
    sd_points = np.array([perp_point, opposite_perp_point])

    return greatest_diameter, smallest_diameter, gd_points, sd_points


def calculate_border_irregularity(binary_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate border irregularity indices from binary mask
    """
    try:
        # Find contours
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the image")

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Calculate perimeter and area
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # Calculate diameters
        greatest_diameter, smallest_diameter, gd_points, sd_points = calculate_diameters(contour)

        # Calculate irregularity indices
        irregularity_A = area / perimeter
        irregularity_B = perimeter / greatest_diameter
        irregularity_C = perimeter * (1 / smallest_diameter - 1 / greatest_diameter)
        irregularity_D = greatest_diameter - smallest_diameter

        # Create visualization
        vis_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        # Draw contour
        cv2.drawContours(vis_mask, [contour], -1, (0, 255, 0), 2)

        # Draw centroid
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(vis_mask, (cx, cy), 5, (0, 0, 255), -1)

        # Draw diameters
        cv2.line(vis_mask,
                 tuple(map(int, gd_points[0])),
                 tuple(map(int, gd_points[1])),
                 (255, 0, 0), 2)  # Greatest diameter in blue

        cv2.line(vis_mask,
                 tuple(map(int, sd_points[0])),
                 tuple(map(int, sd_points[1])),
                 (0, 255, 255), 2)  # Smallest diameter in yellow

        # Save visualization
        temp_path = "images/border_irregularity.png"
        cv2.imwrite(temp_path, vis_mask)

        return {
            "irregularity_A": float(irregularity_A),
            "irregularity_B": float(irregularity_B),
            "irregularity_C": float(irregularity_C),
            "irregularity_D": float(irregularity_D),
            "perimeter": float(perimeter),
            "area": float(area),
            "greatest_diameter": float(greatest_diameter),
            "smallest_diameter": float(smallest_diameter),
            "visualization_path": temp_path
        }

    except Exception as e:
        raise ValueError(f"Error calculating border irregularity: {str(e)}")