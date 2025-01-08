import numpy as np
from skimage.segmentation import slic
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

        self.color_score_thresholds = {
            'white': {1: 3, 2: 8, 3: 15},
            'red': {1: 3, 2: 8, 3: 15},
            'light_brown': {1: 3, 2: 8, 3: 15},
            'dark_brown': {1: 3, 2: 8, 3: 15},
            'blue_gray': {1: 3, 2: 8, 3: 15},
            'black': {1: 3, 2: 8, 3: 15}
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

    def calculate_individual_color_scores(self, color_counts):
        """
        Calculate score for each individual color based on its presence
        Returns both color scores and detailed counts
        """
        color_scores = {}
        for color, count in color_counts.items():
            # Default score is 0
            score = 0
            # Check thresholds in descending order
            for potential_score in [3, 2, 1]:
                if count >= self.color_score_thresholds[color][potential_score]:
                    score = potential_score
                    break
            color_scores[color] = score

        return color_scores


if __name__ == "__main__":
    image = cv2.imread('../segmentation_v2_masked_images/ISIC_0000042_masked.png')
    mask = cv2.imread('../segmentation_v2_masks/ISIC_0000042_segmented.png', cv2.IMREAD_GRAYSCALE)

    color_extractor = ColorInformationExtractor()

    color_counts = color_extractor.extract_colors(image)
    color_score = color_extractor.calculate_color_score(color_counts)

    print("Color counts:", color_counts)
    print("Color score:", color_score)
