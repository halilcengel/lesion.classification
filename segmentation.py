import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


class ImageProcessing:
    def __init__(self, image_path):
        """
        Initialize with the path to the image.

        Parameters:
        image_path (str): Path to the input image.
        """
        self.image = cv2.imread(image_path, 0)  # Grayscale for processing
        self.rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # RGB for display

        if self.image is None:
            raise ValueError("Image not found or path is incorrect")

    def sobel_edge_detection(self, ksize=3):
        """
        Apply Sobel edge detection.
        """
        sobel_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        return np.uint8(sobel_edges)

    def log_edge_detection(self, sigma=1.0):
        """
        Apply Laplacian of Gaussian (LoG) edge detection.
        """
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred_image = cv2.GaussianBlur(self.image, (ksize, ksize), sigma)
        log_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
        return np.uint8(log_edges)

    def roberts_edge_detection(self):
        """
        Apply Roberts edge detection.
        """
        kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)
        edges_x = cv2.filter2D(self.image, cv2.CV_16S, kernel_x)
        edges_y = cv2.filter2D(self.image, cv2.CV_16S, kernel_y)
        roberts_edges = cv2.addWeighted(cv2.convertScaleAbs(edges_x), 0.5, cv2.convertScaleAbs(edges_y), 0.5, 0)
        return roberts_edges

    def global_thresholding(self):
        """
        Apply global thresholding using Otsu's method.
        """
        _, thresholded_image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded_image

    def apply_mask(self, mask, reverse=False):
        """
        Apply a binary mask to the RGB image.

        Parameters:
        mask (numpy.ndarray): Binary mask to apply
        reverse (bool): If True, applies inverted mask

        Returns:
        numpy.ndarray: Masked RGB image
        """
        # Ensure mask has same dimensions as image
        if mask.shape != self.image.shape:
            mask = cv2.resize(mask, (self.rgb_image.shape[1], self.rgb_image.shape[0]))

        # Normalize mask to binary values
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Invert mask if reverse is True
        if reverse:
            binary_mask = cv2.bitwise_not(binary_mask)

        # Create 3-channel mask for RGB image
        binary_mask = binary_mask.astype(np.uint8)
        mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

        # Apply mask to RGB image
        masked_image = cv2.bitwise_and(self.rgb_image, mask_3channel)
        return masked_image


def save_image(image, output_path):
    """
    Save an image to the specified path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert RGB to BGR for cv2.imwrite
    if len(image.shape) == 3:  # If RGB image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, image)


def plot_images(images, titles, output_dir, cols=3):
    """
    Plot a list of images with corresponding titles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes = [ax for sublist in axes for ax in sublist]  # Flatten the list of axes

    for i, (img, title) in enumerate(zip(images, titles)):
        # Handle both RGB and grayscale images
        if len(img.shape) == 3:  # RGB image
            axes[i].imshow(img)
            plt.imsave(os.path.join(output_dir, f"{title}.png"), img)
        else:  # Grayscale image
            axes[i].imshow(img, cmap='gray')
            plt.imsave(os.path.join(output_dir, f"{title}.png"), img, cmap='gray')

        axes[i].set_title(title)
        axes[i].axis('off')

    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Input and output directories
    input_dir = "images"
    # use this in report
    # output_dir = 'images/segmentation'
    segment_dir = 'images/segmentation_masks'
    masked_output_dir = 'images/segmented_images'

    # Create output directories if they don't exist
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(masked_output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    for image_file in image_files:
        try:
            # Setup paths
            image_path = os.path.join(input_dir, image_file)
            title = os.path.splitext(image_file)[0]
            mask_path = os.path.join(segment_dir, f"{title}_mask.png")

            # Process original image
            processor = ImageProcessing(image_path)

            # Get thresholded image and save it as mask
            thresholded = processor.global_thresholding()
            save_image(thresholded, mask_path)

            # Load mask and apply it
            mask = cv2.imread(mask_path, 0)
            masked_image = processor.apply_mask(mask, reverse=True)

            # Save masked image
            masked_path = os.path.join(masked_output_dir, f"{title}_masked.png")
            save_image(masked_image, masked_path)

            # Plot and save all variations (use this in report)
            # plot_images(
            #     [processor.rgb_image,  # Show RGB original
            #      thresholded,
            #      masked_image,  # This will be RGB
            #      processor.sobel_edge_detection(ksize=3),
            #      processor.log_edge_detection(sigma=1.0),
            #      processor.roberts_edge_detection()],
            #     [f'{title} - Original',
            #      f'{title} - Mask',
            #      f'{title} - Masked',
            #      f'{title} - Sobel',
            #      f'{title} - LoG',
            #      f'{title} - Roberts'],
            #     output_dir
            # )

            print(f"Successfully processed {title}")

        except Exception as e:
            print(f"Error processing {title}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
