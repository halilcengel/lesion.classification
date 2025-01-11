import cv2
import numpy as np
from pathlib import Path


def remove_hair(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to open")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use morphological operations to highlight hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Create a binary mask where hair is highlighted
    _, binary_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Remove small noise
    # Define a smaller kernel for morphological operations
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Perform morphological opening to remove small noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, small_kernel)

    # Perform connected component analysis to remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    sizes = stats[1:, -1]  # Get the sizes of each component (excluding the background)
    min_size = 800  # Minimum size of components to keep

    # Create a new binary mask with only the large components
    filtered_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if sizes[i - 1] >= min_size:
            filtered_mask[labels == i] = 255

    # Inpaint the original image using the filtered mask
    inpainted_image = cv2.inpaint(image, filtered_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return inpainted_image


if __name__ == "__main__":
    # Setup paths
    input_dir = Path("images/ISIC_2017/images")
    output_dir = Path("images/ISIC_2017/clean")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in input_dir.glob("*")
        if f.suffix.lower() in image_extensions
    ]

    total = len(image_files)
    successful = 0
    failed = 0

    for input_path in image_files:
        try:
            if "superpixels" in input_path.name:
                continue

            output_path = output_dir / input_path.name
            clean_image = remove_hair(str(input_path))
            cv2.imwrite(str(output_path), clean_image)
            successful += 1
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    print("\nProcessing Complete!")
    print(f"Total images: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
