import numpy as np
import cv2
import matplotlib.pyplot as plt


def analyze_asymmetry(image_path):
    """
    Analyze image asymmetry by comparing upper and lower halves

    Parameters:
    image_path: str, path to the input image

    Returns:
    dict containing processed images for visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    height, width = gray.shape
    mid_height = height // 2

    upper_half = gray[:mid_height, :]
    lower_half = gray[mid_height:, :]

    if lower_half.shape[0] != upper_half.shape[0]:
        lower_half = cv2.resize(lower_half, (width, mid_height))

    rotated_upper = cv2.rotate(upper_half, cv2.ROTATE_180)

    rotated_lower = cv2.rotate(lower_half, cv2.ROTATE_180)

    folded_lower = cv2.flip(lower_half, 0)

    diff_image = cv2.absdiff(upper_half, folded_lower)

    diff_normalized = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)

    asymmetry_score = np.mean(diff_image)

    return {
        'original': gray,
        'upper_half': upper_half,
        'lower_half': lower_half,
        'rotated_upper': rotated_upper,
        'rotated_lower': rotated_lower,
        'folded_lower': folded_lower,
        'difference': diff_normalized,
        'asymmetry_score': asymmetry_score
    }


def visualize_results(results):
    """
    Visualize the asymmetry analysis results
    """
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(results['original'], cmap='gray')
    plt.title('Original Image')

    plt.subplot(232)
    plt.imshow(results['rotated_upper'], cmap='gray')
    plt.title('Rotated Upper Half')

    plt.subplot(233)
    plt.imshow(results['rotated_lower'], cmap='gray')
    plt.title('Rotated Lower Half')

    plt.subplot(234)
    plt.imshow(results['folded_lower'], cmap='gray')
    plt.title('Folded Lower Half')

    plt.subplot(235)
    plt.imshow(results['difference'], cmap='hot')
    plt.title(f'Difference Image\nAsymmetry Score: {results["asymmetry_score"]:.2f}')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "segmentation_v2_masks/ISIC_0000042_segmented.png"
    results = analyze_asymmetry(image_path)
    visualize_results(results)