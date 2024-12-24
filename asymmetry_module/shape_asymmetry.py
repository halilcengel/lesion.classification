import cv2
import matplotlib.pyplot as plt
from asymmetry_module.utils import rotate_image, split_vertically, split_horizontally


def calculate_asymmetry(image, split_by='vertical'):
    diff_percentage = 0  # Initialize with a default value

    if split_by == 'vertical':
        left_half, right_half = split_vertically(image)
        left_area = cv2.countNonZero(left_half)
        right_area = cv2.countNonZero(right_half)
        total_area = left_area + right_area
        area_diff = abs(left_area - right_area)
        diff_percentage = (area_diff / total_area) * 100

    elif split_by == 'horizontal':
        top_half, bottom_half = split_horizontally(image)
        top_area = cv2.countNonZero(top_half)
        bottom_area = cv2.countNonZero(bottom_half)
        total_area = top_area + bottom_area
        area_diff = abs(top_area - bottom_area)
        diff_percentage = (area_diff / total_area) * 100

    return diff_percentage


def visualize_asymmetry_steps(image):
    """Ï
    Visualize each step of asymmetry calculation with matplotlib

    Args:
        image: Input binary image (lesion should be white, background black)
    """
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))

    # 1. Original Image
    plt.subplot(231)
    plt.imshow(image, cmap='gray')
    plt.title('1. Original Image')
    plt.axis('off')

    height, width = image.shape
    rotated_image = rotate_image(image)

    plt.subplot(232)
    plt.imshow(rotated_image, cmap='gray')
    plt.title('2. Rotated Along Main Axis')
    plt.axis('off')

    # 3. Split Vertically
    left_half, right_half = split_vertically(image)

    # 4. Split Horizontally
    # top_half, bottom_half = split_horizontally(image)

    split_viz = rotated_image.copy()
    cv2.line(split_viz, (width // 2, 0), (width // 2, height), (127, 127, 127), 2)
    cv2.line(split_viz, (0, height // 2), (width, height // 2), (127, 127, 127), 2)

    plt.subplot(233)
    plt.imshow(split_viz, cmap='gray')
    plt.title('3. Vertical Split')
    plt.axis('off')

    # 4. Show Left Half
    plt.subplot(234)
    plt.imshow(left_half, cmap='gray')
    plt.title('4. Left Half')
    plt.axis('off')

    # 5. Show Right Half
    plt.subplot(235)
    plt.imshow(right_half, cmap='gray')
    plt.title('5. Right Half')
    plt.axis('off')

    # 6. Calculate and Display Results
    vertical_diff_percentage = calculate_asymmetry(image, split_by='vertical')
    horizontal_diff_percentage = calculate_asymmetry(image, split_by='horizontal')

    plt.subplot(236)
    plt.bar(['Vertical', 'Horizontal'], [vertical_diff_percentage, horizontal_diff_percentage])
    plt.title('6. Asymmetry Percentage')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)

    print(f'Vertical Asymmetry: {vertical_diff_percentage:.2f}%')
    print(f'Horizontal Asymmetry: {horizontal_diff_percentage:.2f}%')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    your_image = cv2.imread('../segmentation_v2_masked_images/ISIC_0000042_masked.png', cv2.IMREAD_GRAYSCALE)
    visualize_asymmetry_steps(your_image)
