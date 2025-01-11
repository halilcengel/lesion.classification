import os
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import Union

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class ImageProcessing:
    def __init__(self, image_source: Union[str, np.ndarray]) -> None:
        """
        Initialize with either a path to the image or an image array.

        Parameters:
        image_source (Union[str, np.ndarray]): Path to input image or numpy array of the image.
        """
        if isinstance(image_source, str):
            # Load from file path
            self.image = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
            self.rgb_image = cv2.cvtColor(cv2.imread(image_source), cv2.COLOR_BGR2RGB)

            if self.image is None or self.rgb_image is None:
                raise ValueError(f"Could not read image from path: {image_source}")

        elif isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 2:
                # Input is grayscale
                self.image = image_source
                self.rgb_image = cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
            elif len(image_source.shape) == 3 and image_source.shape[2] == 3:
                # Input is BGR (OpenCV default) or RGB
                # Assume BGR input for consistency with OpenCV
                self.rgb_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
                self.image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError("Input array must be 2D grayscale or 3D BGR/RGB image")
        else:
            raise ValueError("Input must be either a file path (str) or numpy array")

    # All other methods remain exactly the same since they work with self.image and self.rgb_image
    def sobel_edge_detection(self, ksize: int = 3) -> np.ndarray:
        """
        Apply Sobel edge detection on the grayscale image.

        Parameters:
        ksize (int): Aperture size for the Sobel operator.

        Returns:
        np.ndarray: Image containing the magnitude of gradients.
        """
        sobel_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        return np.uint8(sobel_edges)

    def log_edge_detection(self, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Laplacian of Gaussian (LoG) edge detection.

        Parameters:
        sigma (float): Standard deviation for Gaussian blur.

        Returns:
        np.ndarray: Image with LoG-detected edges.
        """
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred_image = cv2.GaussianBlur(self.image, (ksize, ksize), sigma)
        log_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
        return np.uint8(log_edges)

    def global_thresholding(self) -> np.ndarray:
        """
        Apply global thresholding using Otsu's method on the grayscale image.

        Returns:
        np.ndarray: Thresholded (binary) image.
        """
        _, thresholded_image = cv2.threshold(
            self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresholded_image

    def apply_mask(self, mask: np.ndarray, reverse: bool = False) -> np.ndarray:
        """
        Apply a binary mask to the stored RGB image.

        Parameters:
        mask (np.ndarray): Binary or grayscale mask to apply.
        reverse (bool): If True, apply an inverted mask.

        Returns:
        np.ndarray: Masked RGB image.
        """
        if mask.shape != self.image.shape:
            mask = cv2.resize(mask, (self.rgb_image.shape[1], self.rgb_image.shape[0]))

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if reverse:
            binary_mask = cv2.bitwise_not(binary_mask)

        # Convert single-channel mask to 3-channel
        mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

        # Apply mask to RGB image
        masked_image = cv2.bitwise_and(self.rgb_image, mask_3channel)
        return masked_image

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to the specified path. Converts RGB to BGR if necessary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # For RGB images, convert to BGR before saving
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, image)


def plot_images(images: list[np.ndarray], titles: list[str], output_dir: str, cols: int = 3) -> None:
    """
    Plot a list of images with corresponding titles and save them.

    Parameters:
    images (list[np.ndarray]): List of images (grayscale or RGB).
    titles (list[str]): Corresponding titles for the images.
    output_dir (str): Directory to save the images.
    cols (int): Number of columns for the subplot layout.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten axes for easy iteration

    for i, (img, title) in enumerate(zip(images, titles)):
        if img is None:
            logging.warning(f"Skipping empty image for '{title}'")
            continue
        if len(img.shape) == 3:  # RGB
            axes[i].imshow(img)
            plt.imsave(os.path.join(output_dir, f"{title}.png"), img)
        else:  # Grayscale
            axes[i].imshow(img, cmap='gray')
            plt.imsave(os.path.join(output_dir, f"{title}.png"), img, cmap='gray')

        axes[i].set_title(title)
        axes[i].axis('off')

    # Turn off remaining axes if any
    for j in range(len(images), rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function that processes images:
    1. Reads images from a specified folder.
    2. Performs thresholding to create a mask.
    3. Applies the mask to the original image (optionally reversed).
    4. Saves and optionally plots the results.
    """
    # Input and output directories
    input_dir = "images/rapor/hair_removed"
    segment_dir = 'images/rapor/segmentation/mask'
    masked_output_dir = 'images/rapor/segmentation/segmented_images'

    # Create necessary directories
    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(masked_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        title = os.path.splitext(image_file)[0]
        mask_path = os.path.join(segment_dir, f"{title}_mask.png")

        try:
            processor = ImageProcessing(image_path)

            # Global thresholding to create mask
            thresholded = processor.global_thresholding()
            save_image(thresholded, mask_path)

            # Load just-created mask and apply it
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask not found at: {mask_path}")

            masked_image = processor.apply_mask(mask, reverse=True)
            masked_path = os.path.join(masked_output_dir, f"{title}_masked.png")
            save_image(masked_image, masked_path)

            logging.info(f"Successfully processed {title}")

        except Exception as e:
            logging.error(f"Error processing {title}: {e}")


if __name__ == "__main__":
    main()