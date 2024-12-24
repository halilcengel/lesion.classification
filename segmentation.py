import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from pathlib import Path


def ShowImage(title, img, ctype, save_path=None):
    """Display and optionally save an image"""
    plt.figure(figsize=(10, 10))
    if ctype == 'bgr':
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        plt.imshow(rgb_img)
    elif ctype == 'hsv':
        rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        plt.imshow(rgb)
    elif ctype == 'gray':
        plt.imshow(img, cmap='gray')
    elif ctype == 'rgb':
        plt.imshow(img)
    else:
        raise Exception("Unknown colour type")

    plt.axis('off')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def process_image(image_path, output_dir):
    """Process a single image with watershed segmentation"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get filename without extension for saving results
    filename = Path(image_path).stem

    # Read and convert image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initial thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Connected components analysis
    ret, markers = cv2.connectedComponents(thresh)

    # Get areas and largest component
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
    largest_component = np.argmax(marker_area) + 1
    brain_mask = markers == largest_component

    # Watershed segmentation
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # Morphological operations
    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    # Final masking
    result = img.copy()
    result[closing == False] = (0, 0, 0)

    # Save results
    cv2.imwrite(os.path.join(output_dir, f"{filename}_segmented.jpg"), result)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_mask.jpg"), closing * 255)

    return result, closing


def batch_process_images():
    """Process all images in the input directory"""
    input_dir = Path("clean_images")
    output_dir = Path("watershed_segments")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in input_dir.glob("*") if f.suffix.lower() in image_extensions]

    if not image_files:
        print("No image files found in input directory!")
        return

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"Processing image {i}/{len(image_files)}: {image_path.name}")
            result, mask = process_image(image_path, output_dir)
            print(f"Successfully processed {image_path.name}")
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            continue


if __name__ == "__main__":
    batch_process_images()