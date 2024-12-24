from asymmetry_module.utils import rotate_image, split_vertically, split_horizontally
import numpy as np
import cv2


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


def calculate_brightness_asymmetry(image):
    rotated_image = rotate_image(image)

    top_half, bottom_half = split_horizontally(rotated_image)
    left_half, right_half = split_vertically(rotated_image)

    top_intensity = calculate_asymmetry_metrics(top_half)
    bottom_intensity = calculate_asymmetry_metrics(bottom_half)
    left_intensity = calculate_asymmetry_metrics(left_half)
    right_intensity = calculate_asymmetry_metrics(right_half)
    total_intensity = rotated_image.mean()

    vertical_asymmetry_percentage = (abs(top_intensity - bottom_intensity) / total_intensity) * 100
    horizontal_asymmetry_percentage = (abs(left_intensity - right_intensity) / total_intensity) * 100

    print(f"Vertical Asymmetry: {vertical_asymmetry_percentage}")
    print(f"Horizontal Asymmetry: {horizontal_asymmetry_percentage}")

    if vertical_asymmetry_percentage > 3:
        print("Result: Vertical Asymmetric")
    else:
        print("Result: Vertical Symmetric")

    if horizontal_asymmetry_percentage > 3:
        print("Result: Horizontal Asymmetric")
    else:
        print("Result: Horizontal Symmetric")

    return vertical_asymmetry_percentage, horizontal_asymmetry_percentage


if __name__ == '__main__':
    image = cv2.imread('../segmentation_v2_masked_images/ISIC_0000042_masked.png', cv2.IMREAD_GRAYSCALE)
    calculate_brightness_asymmetry(image)
