import cv2
import numpy as np


def calculate_asymmetry(image):
    """
    Calculate shape asymmetry by comparing two halves of a lesion image.

    Args:
        image: Binary image where the lesion is white (255) and background is black (0)

    Returns:
        asymmetry_score: True if asymmetric, False if symmetric
        diff_percentage: Percentage difference between two halves
    """

    # Step 1: Find the orientation of the lesion
    # Calculate moments of the binary image
    moments = cv2.moments(image)

    # Calculate the orientation angle
    if moments['mu20'] != moments['mu02']:
        theta = 0.5 * np.arctan2(2 * moments['mu11'],
                                 moments['mu20'] - moments['mu02'])
    else:
        theta = 0

    # Step 2: Rotate the image to align with main axis
    height, width = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(theta), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Step 3: Split the image into two halves
    left_half = rotated_image[:, :width // 2]
    right_half = rotated_image[:, width // 2:]

    # Step 4: Calculate areas
    left_area = cv2.countNonZero(left_half)
    right_area = cv2.countNonZero(right_half)
    total_area = left_area + right_area

    # Step 5: Calculate difference percentage
    area_diff = abs(left_area - right_area)
    diff_percentage = (area_diff / total_area) * 100

    # Step 6: Determine if asymmetric (threshold of 2%)
    asymmetry_score = diff_percentage > 2.0

    return asymmetry_score, diff_percentage


def get_vertical_asymmetry(image):
    """
    Calculate vertical asymmetry by comparing top and bottom halves
    """
    height = image.shape[0]
    top_half = image[:height // 2, :]
    bottom_half = image[height // 2:, :]

    top_area = cv2.countNonZero(top_half)
    bottom_area = cv2.countNonZero(bottom_half)
    total_area = top_area + bottom_area

    area_diff = abs(top_area - bottom_area)
    diff_percentage = (area_diff / total_area) * 100

    return diff_percentage > 2.0, diff_percentage


img = cv2.imread('segmentation_v2_masks/ISIC_0000042_segmented.png', cv2.IMREAD_GRAYSCALE)
# Threshold the image to get binary image if needed
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

horizontal_asymmetry, h_diff = calculate_asymmetry(binary_img)
vertical_asymmetry, v_diff = get_vertical_asymmetry(binary_img)

print(f"Horizontal asymmetry: {horizontal_asymmetry} (diff: {h_diff:.2f}%)")
print(f"Vertical asymmetry: {vertical_asymmetry} (diff: {v_diff:.2f}%)")