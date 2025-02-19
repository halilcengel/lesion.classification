import numpy as np
from PIL.ImageChops import overlay
from skimage.segmentation import slic
from skimage.color import rgb2lab
import cv2


class ColorInformationExtractor:
    def __init__(self):
        # Reference colors in RGB format
        self.reference_colors = {
            'white': np.array([255, 255, 255]),
            'red': np.array([255, 0, 0]),
            'light_brown': np.array([181, 134, 84]),
            'dark_brown': np.array([91, 60, 17]),
            'blue_gray': np.array([104, 137, 148]),
            'black': np.array([0, 0, 0])
        }

    def generate_superpixels(self, image):
        """Generate superpixels using SLIC algorithm with 32 pixels per superpixel"""
        # Calculate number of segments based on image size and 32 pixels per superpixel
        height, width = image.shape[:2]
        total_pixels = height * width
        n_segments = total_pixels // 32  # Each superpixel should have 32 pixels

        segments = slic(image, n_segments=32, compactness=10, sigma=1)
        return segments

    def get_dominant_color(self, region):
        """Find dominant color in a region by comparing to reference colors"""
        min_dist = float('inf')
        dominant_color = None

        # Calculate mean color of region
        mean_color = np.mean(region, axis=0)

        # Compare with reference colors
        for color_name, ref_color in self.reference_colors.items():
            dist = np.linalg.norm(mean_color - ref_color)
            if dist < min_dist:
                min_dist = dist
                dominant_color = color_name

        return dominant_color

    def extract_colors(self, image):
        """Main function to extract colors from dermoscopic image"""

        # Generate superpixels
        segments = self.generate_superpixels(image)

        # Initialize color counts
        color_counts = {color: 0 for color in self.reference_colors.keys()}

        # Analyze each superpixel
        for segment_id in np.unique(segments):
            # Get region mask
            region_mask = segments == segment_id
            region = image[region_mask]

            if len(region) > 0:
                # Get dominant color
                dominant_color = self.get_dominant_color(region)
                color_counts[dominant_color] += 1

        return color_counts

    def calculate_color_score(self, color_counts):
        """Calculate C score based on number of colors present"""
        # Count colors with significant presence (more than threshold)
        threshold = 5  # Minimum number of superpixels for color to be considered present
        colors_present = sum(1 for count in color_counts.values() if count > threshold)

        return colors_present

    def create_red_segmentation(path, mask_path):
        # Read images
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create red overlay mask
        red_mask = np.zeros_like(image)
        red_mask[mask == 0] = [0, 0, 255]  # Set red color where mask is 0 (lesion)

        # Blend original image with red mask
        alpha = 0.5  # Transparency factor
        overlay = cv2.addWeighted(image, 1, red_mask, alpha, 0)

        # Save result
        cv2.imwrite('segmentation_visualization.png', overlay)

        return overlay






if __name__ == "__main__":
    # Load image and mask
    image = cv2.imread('segmentation_v2_masked_images/ISIC_0000042_masked.png')
    mask = cv2.imread('segmentation_v2_masks/ISIC_0000042_segmented.png', cv2.IMREAD_GRAYSCALE)


    # Initialize color information extractor
    color_extractor = ColorInformationExtractor()


    overlay = color_extractor.create_red_segmentation('segmentation_v2_masked_images/ISIC_0000042_masked.png', 'segmentation_v2_masks/ISIC_0000042_segmented.png')


    # Extract color information
    color_counts = color_extractor.extract_colors(image)
    color_score = color_extractor.calculate_color_score(color_counts)

    # Calculate color asymmetry
    print("Color counts:", color_counts)
    print("Color score:", color_score)
