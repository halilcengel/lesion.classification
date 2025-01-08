import asyncio
import os

import numpy as np
import cv2
from typing import Dict, Tuple

from typing import List

from hair_removal import demo_hair_removal

from asymmetry_module.color_asymmetry import ColorAsymmetryAnalyzer
from asymmetry_module.utils import rotate_image, split_horizontally, split_vertically
from asymmetry_module.shape_asymmetry import calculate_asymmetry
from asymmetry_module.brightness_asymmetry import calculate_asymmetry_metrics

from differential_structures_module.pigment_network_detection import detect_pigment_network
from differential_structures_module.blue_white_veil_detection import detect_blue_white_veil
from differential_structures_module.dots_and_globs_detection import detect_dots_globules
from differential_structures_module.structless_area_detection import detect_structureless_area
from differential_structures_module.utils import calculate_area_percentage


from border_irregularity_module import utils

from color_module.color import ColorInformationExtractor


class LesionFeatureExtractor:
    def __init__(self, n_segments: int = 200, compactness: int = 10, color_threshold: float = 1e-3, border_segments: int = 8):
        """
        Initialize combined asymmetry analyzer that incorporates color, brightness, and shape features.

        Args:
            n_segments: Number of segments for color-based SLIC analysis
            compactness: Compactness parameter for SLIC
            color_threshold: Threshold for color similarity comparison
        """
        self.color_analyzer = ColorAsymmetryAnalyzer(
            n_segments=n_segments,
            compactness=compactness,
            color_threshold=color_threshold
        )
        self.border_segments = border_segments
        self.color_extractor = ColorInformationExtractor()


    def calculate_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features from color-based asymmetry analysis."""
        h_asym, v_asym, details = self.color_analyzer.analyze_image(image)

        features = {
            'color_horizontal_symmetric_ratio': details['horizontal']['symmetric_count'] /
                                                (details['horizontal']['symmetric_count'] + details['horizontal'][
                                                    'asymmetric_count']),
            'color_vertical_symmetric_ratio': details['vertical']['symmetric_count'] /
                                              (details['vertical']['symmetric_count'] + details['vertical'][
                                                  'asymmetric_count']),
            'color_horizontal_is_asymmetric': float(h_asym),
            'color_vertical_is_asymmetric': float(v_asym)
        }
        return features

    def calculate_brightness_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features from brightness-based asymmetry analysis."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        rotated = rotate_image(gray_image)

        top_half, bottom_half = split_horizontally(rotated)
        left_half, right_half = split_vertically(rotated)

        top_intensity = calculate_asymmetry_metrics(top_half)
        bottom_intensity = calculate_asymmetry_metrics(bottom_half)
        left_intensity = calculate_asymmetry_metrics(left_half)
        right_intensity = calculate_asymmetry_metrics(right_half)
        total_intensity = rotated.mean()

        features = {
            'brightness_vertical_asymmetry': (abs(top_intensity - bottom_intensity) / total_intensity) * 100,
            'brightness_horizontal_asymmetry': (abs(left_intensity - right_intensity) / total_intensity) * 100,
            'brightness_top_bottom_ratio': top_intensity / (bottom_intensity + 1e-6),
            'brightness_left_right_ratio': left_intensity / (right_intensity + 1e-6)
        }
        return features

    def calculate_differential_structure_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features from differential structures analysis."""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        pigment_network = detect_pigment_network(img_gray)
        blue_white_veil = detect_blue_white_veil(image)
        dots_and_globs = detect_dots_globules(img_gray)
        structless_area = detect_structureless_area(img_gray)

        pn_percentage = calculate_area_percentage(pigment_network)
        bw_percentage = calculate_area_percentage(blue_white_veil)
        dg_percentage = calculate_area_percentage(dots_and_globs)
        sa_percentage = calculate_area_percentage(structless_area)

        features = {
            'differential_pigment_network_percentage': pn_percentage,
            'differential_blue_white_veil_percentage': bw_percentage,
            'differential_dots_globules_percentage': dg_percentage,
            'differential_structureless_percentage': sa_percentage,
        }

        return features

    def calculate_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features from shape-based asymmetry analysis."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        vertical_diff = calculate_asymmetry(gray, split_by='vertical')
        horizontal_diff = calculate_asymmetry(gray, split_by='horizontal')

        features = {
            'shape_vertical_asymmetry': vertical_diff,
            'shape_horizontal_asymmetry': horizontal_diff,
            'shape_total_asymmetry': (vertical_diff + horizontal_diff) / 2
        }
        return features

    def calculate_border_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features from border irregularity analysis."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Detect border
        image_border = utils.detect_border(image_gray)

        # Split border into segments
        segments = utils.split_border_into_segments(image_border, num_segments=self.border_segments)

        # Calculate irregularity score
        total_score = utils.compute_border_irregularity(segments)

        features = {
            'border_total_irregularity': total_score,
            'border_irregularity_normalized': total_score / self.border_segments
        }
        return features

    def calculate_detailed_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract detailed color distribution features including individual color scores."""
        # Get color counts and scores
        color_counts = self.color_extractor.extract_colors(image)
        total_color_score = self.color_extractor.calculate_color_score(color_counts)
        individual_color_scores = self.color_extractor.calculate_individual_color_scores(color_counts)

        # Calculate total superpixels
        total_superpixels = sum(color_counts.values())

        features = {}

        for color in self.color_extractor.reference_colors.keys():
            features[f'color_score_{color}'] = individual_color_scores[color]

        nonzero_colors = [c for c in color_counts.values() if c > 0]
        features.update({
            'color_diversity': len(nonzero_colors),
            'color_entropy': -sum((c / total_superpixels) * np.log2(c / total_superpixels + 1e-10)
                                  for c in color_counts.values()),
            'dominant_color_ratio': max(color_counts.values()) / total_superpixels
        })

        return features

    def extract_feature_vector(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract a combined feature vector incorporating all asymmetry measures.

        Args:
            image: Input image array (BGR or grayscale)

        Returns:
            Tuple of (feature_vector, feature_dict) where:
            - feature_vector is a numpy array of all features in a fixed order
            - feature_dict is a dictionary with named features and their values
        """
        # Get all features
        color_features = self.calculate_color_features(image)
        detailed_color_features = self.calculate_detailed_color_features(image)
        brightness_features = self.calculate_brightness_features(image)
        shape_features = self.calculate_shape_features(image)
        border_features = self.calculate_border_features(image)
        differential_features = self.calculate_differential_structure_features(image)

        all_features = {
            **color_features,
            **detailed_color_features,
            **brightness_features,
            **shape_features,
            **border_features,
            **differential_features
        }

        # Create fixed-order feature vector
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])

        return feature_vector, all_features

    def get_feature_names(self) -> List[str]:
        """Get the names of all features in the order they appear in the feature vector."""
        # Create a dummy image to get feature names
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, feature_dict = self.extract_feature_vector(dummy_image)
        return sorted(feature_dict.keys())


async def main():
    image_path = 'images/ISIC_2018/Test/actinic keratosis/ISIC_0010889.jpg'

    try:
        if not os.path.exists(image_path):
            print(f"Image not found at path: {image_path}")
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")

            result = await demo_hair_removal(image)
            analyzer = LesionFeatureExtractor()
            feature_vector, feature_dict = analyzer.extract_feature_vector(result)

            print("\nFeature Vector Shape:", feature_vector.shape)
            print("\nFeature Details:")
            for name, value in feature_dict.items():
                print(f"{name}: {value:.4f}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())