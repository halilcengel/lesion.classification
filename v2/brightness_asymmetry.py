import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import moments_central, moments
from scipy.ndimage import rotate
import cv2


def rotate_image(segmented_image):
    """
    Rotate an image by a given angle in degrees.
    """
    M = moments_central(segmented_image)
    theta = 0.5 * np.arctan2(2 * M[1, 1], (M[2, 0] - M[0, 2]))
    angle_degrees = np.degrees(theta)

    rotated_mask = rotate(segmented_image, angle_degrees, reshape=False)
    return rotated_mask


def split_horizontal(rotated_mask):
    """
    Calculate shape asymmetry and return detailed intermediate results
    """
    height, width = rotated_mask.shape
    center_y = height // 2

    top_half = rotated_mask[:center_y, :].copy()
    bottom_half = rotated_mask[center_y:, :].copy()

    return top_half, bottom_half


def split_vertical(rotated_mask):
    """
    Calculate shape asymmetry and return detailed intermediate results
    """
    height, width = rotated_mask.shape
    center_x = width // 2

    left_half = rotated_mask[:, :center_x].copy()
    right_half = rotated_mask[:, center_x:].copy()

    return left_half, right_half


def calculate_pixel_intensity(image_array):
    """
    Calculate pixel intensity of an image array.
    """
    if len(image_array.shape) == 2:
        return image_array

    return np.mean(image_array, axis=2).astype(np.uint8)


def calculate_asymmetry_metrics(image_array):
    """
    Calculate normalized asymmetry metrics
    """
    mean_intensity = image_array.mean()
    if mean_intensity == 0:
        return 0
    return mean_intensity


def main():
    image = cv2.imread('segmentation_v2_masked_images/ISIC_0000042_masked.png', cv2.IMREAD_GRAYSCALE)
    rotated_image = rotate_image(image)

    top_half, bottom_half = split_horizontal(rotated_image)
    left_half, right_half = split_vertical(rotated_image)

    top_intensity = calculate_asymmetry_metrics(top_half)
    bottom_intensity = calculate_asymmetry_metrics(bottom_half)
    left_intensity = calculate_asymmetry_metrics(left_half)
    right_intensity = calculate_asymmetry_metrics(right_half)

    total_intensity = rotated_image.mean()

    vertical_asymmetry_percent = (abs(top_intensity - bottom_intensity) / total_intensity) * 100
    horizontal_asymmetry_percent = (abs(left_intensity - right_intensity) / total_intensity) * 100

    print("\nAsymmetry Analysis:")
    print(f"Vertical Asymmetry: {vertical_asymmetry_percent:.2f}%")
    print(f"Horizontal Asymmetry: {horizontal_asymmetry_percent:.2f}%")

    if (vertical_asymmetry_percent > 3):
        print("\nResult: Vertical Asymmetric")
    else:
        print("\nResult: Vertical Symmetric")

    if (horizontal_asymmetry_percent > 3):
        print("Result: Horizontal Asymmetric")
    else:
        print("Result: Horizontal Symmetric")

if __name__ == "__main__":
    main()
