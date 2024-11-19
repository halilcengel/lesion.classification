import numpy as np
import cv2


def calculate_asymmetry(binary_mask: np.ndarray) -> tuple:
    """
    Calculate asymmetry indices of a lesion from its binary mask
    """
    # Find the centroid
    moments = cv2.moments(binary_mask.astype(np.uint8))
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    # Total area
    total_area = np.sum(binary_mask)

    # Split into halves
    left_half = binary_mask[:, :centroid_x]
    right_half = binary_mask[:, centroid_x:]
    right_half_flipped = np.fliplr(right_half)

    top_half = binary_mask[:centroid_y, :]
    bottom_half = binary_mask[centroid_y:, :]
    bottom_half_flipped = np.flipud(bottom_half)

    # Pad if necessary
    if left_half.shape[1] != right_half_flipped.shape[1]:
        diff = abs(left_half.shape[1] - right_half_flipped.shape[1])
        if left_half.shape[1] < right_half_flipped.shape[1]:
            left_half = np.pad(left_half, ((0, 0), (0, diff)))
        else:
            right_half_flipped = np.pad(right_half_flipped, ((0, 0), (0, diff)))

    if top_half.shape[0] != bottom_half_flipped.shape[0]:
        diff = abs(top_half.shape[0] - bottom_half_flipped.shape[0])
        if top_half.shape[0] < bottom_half_flipped.shape[0]:
            top_half = np.pad(top_half, ((0, diff), (0, 0)))
        else:
            bottom_half_flipped = np.pad(bottom_half_flipped, ((0, diff), (0, 0)))

    # Calculate asymmetry
    x_asymmetry = np.sum(np.logical_xor(left_half, right_half_flipped))
    y_asymmetry = np.sum(np.logical_xor(top_half, bottom_half_flipped))

    # Calculate indices
    A1 = (min(x_asymmetry, y_asymmetry) / total_area) * 100
    A2 = ((x_asymmetry + y_asymmetry) / total_area) * 100

    return A1, A2