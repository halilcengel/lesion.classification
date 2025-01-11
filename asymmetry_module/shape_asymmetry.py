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
    """
    Visualize each step of asymmetry calculation and return individual plots

    Args:
        image: Input binary image (lesion should be white, background black)

    Returns:
        dict: Dictionary containing individual figures for each step and calculated asymmetry values
    """
    results = {}
    height, width = image.shape

    # Store the original image array
    results['original_array'] = image.copy()

    # 1. Original Image
    fig1 = plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title('1. Original Image')
    plt.axis('off')
    results['original'] = fig1

    # 2. Rotated Image
    rotated_image = rotate_image(image)
    results['rotated_array'] = rotated_image.copy()
    fig2 = plt.figure(figsize=(6, 6))
    plt.imshow(rotated_image, cmap='gray')
    plt.title('2. Rotated Along Main Axis')
    plt.axis('off')
    results['rotated'] = fig2

    # 3. Split Visualization
    split_viz = rotated_image.copy()
    cv2.line(split_viz, (width // 2, 0), (width // 2, height), (127, 127, 127), 2)
    cv2.line(split_viz, (0, height // 2), (width, height // 2), (127, 127, 127), 2)
    results['split_array'] = split_viz.copy()

    fig3 = plt.figure(figsize=(6, 6))
    plt.imshow(split_viz, cmap='gray')
    plt.title('3. Split Visualization')
    plt.axis('off')
    results['split'] = fig3

    # 4. Left Half
    left_half, right_half = split_vertically(image)
    results['left_half_array'] = left_half.copy()
    fig4 = plt.figure(figsize=(6, 6))
    plt.imshow(left_half, cmap='gray')
    plt.title('4. Left Half')
    plt.axis('off')
    results['left_half'] = fig4

    # 5. Right Half
    results['right_half_array'] = right_half.copy()
    fig5 = plt.figure(figsize=(6, 6))
    plt.imshow(right_half, cmap='gray')
    plt.title('5. Right Half')
    plt.axis('off')
    results['right_half'] = fig5

    # 6. Asymmetry Bar Plot
    vertical_diff_percentage = calculate_asymmetry(image, split_by='vertical')
    horizontal_diff_percentage = calculate_asymmetry(image, split_by='horizontal')

    fig6 = plt.figure(figsize=(6, 6))
    plt.bar(['Dikey', 'Yatay'],
            [vertical_diff_percentage, horizontal_diff_percentage])
    plt.title('Asimetri Yüzdesi')
    plt.ylabel('Yüzde')
    plt.ylim(0, 50)
    results['asymmetry_plot'] = fig6

    # Store calculated values
    results['vertical_asymmetry'] = vertical_diff_percentage
    results['horizontal_asymmetry'] = horizontal_diff_percentage

    # Close all figures to free memory
    plt.close('all')

    return results

if __name__ == '__main__':
    your_image = cv2.imread('../images/segmented_images/ISIC_0000042_masked.png', cv2.IMREAD_GRAYSCALE)
    results = visualize_asymmetry_steps(your_image)

    import os

    save_dir = "../images/rapor/sekil_asimetri"
    os.makedirs(save_dir, exist_ok=True)

    # Save arrays using cv2.imwrite
    cv2.imwrite(os.path.join(save_dir, "original.png"), results['original_array'])
    cv2.imwrite(os.path.join(save_dir, "rotated.png"), results['rotated_array'])
    cv2.imwrite(os.path.join(save_dir, "split.png"), results['split_array'])
    cv2.imwrite(os.path.join(save_dir, "left_half.png"), results['left_half_array'])
    cv2.imwrite(os.path.join(save_dir, "right_half.png"), results['right_half_array'])

    # Save the asymmetry plot using matplotlib
    results['asymmetry_plot'].savefig(os.path.join(save_dir, "asymmetry_plot.png"))

    # Display any specific plot
    plt.figure(results['original'].number)
    plt.show()

    print(f"Vertical Asymmetry: {results['vertical_asymmetry']:.2f}%")
    print(f"Horizontal Asymmetry: {results['horizontal_asymmetry']:.2f}%")
