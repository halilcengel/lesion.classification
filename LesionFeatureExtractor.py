import asyncio
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from typing import Dict, Tuple

from typing import List

from hair_removal import demo_hair_removal, remove_hair_with_visualization

from asymmetry_module.color_asymmetry import ColorAsymmetryAnalyzer
from asymmetry_module.utils import rotate_image, split_horizontally, split_vertically
from asymmetry_module.shape_asymmetry import calculate_asymmetry
from asymmetry_module.brightness_asymmetry import calculate_asymmetry_metrics

from differential_structures_module.pigment_network_detection import detect_pigment_network
from differential_structures_module.blue_white_veil_detection import detect_blue_white_veil
from differential_structures_module.dots_and_globs_detection import detect_dots_globules
from differential_structures_module.structless_area_detection import detect_structureless_area
from differential_structures_module.utils import calculate_area_percentage

from border_irregularity_module import main

from color_module.color import ColorInformationExtractor

from segmentation_v2 import ImageProcessing


class LesionFeatureExtractor:
    def __init__(self, n_segments: int = 50, compactness: int = 10, color_threshold: float = 1e-3,
                 border_segments: int = 8):
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
        print("Calculating color features")
        features = {
            'color_horizontal_symmetric_ratio': details['horizontal']['symmetric_count'] / \
                                                (details['horizontal']['symmetric_count'] + details['horizontal'][
                                                    'asymmetric_count']),
            'color_vertical_symmetric_ratio': details['vertical']['symmetric_count'] / \
                                              (details['vertical']['symmetric_count'] + details['vertical'][
                                                  'asymmetric_count']),
            'color_horizontal_is_asymmetric': float(h_asym),
            'color_vertical_is_asymmetric': float(v_asym)
        }
        return features

    def calculate_brightness_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features from brightness-based asymmetry analysis."""
        print("Calculating brightness features")
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
        print("Calculating differential structure features")
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
        print("calculating shape features")
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

        total_score = main.calculate_total_b_score(image)

        features = {
            'border_total_irregularity': total_score,
            'border_irregularity_normalized': total_score / 8
        }
        return features

    def calculate_detailed_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract detailed color distribution features including individual color scores."""
        # Get color counts and scores
        color_counts = self.color_extractor.extract_colors(image)
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
    # Initialize feature extractor
    extractor = LesionFeatureExtractor()

    # Base directory containing all lesion type folders
    base_dir = os.path.join(os.path.dirname(__file__), "images", "ISIC_2018", "Train_masked_lesions")

    # Lists to store features and labels
    all_features = []
    labels = []
    image_paths = []

    # Process each lesion type folder
    for lesion_type in os.listdir(base_dir):
        lesion_dir = os.path.join(base_dir, lesion_type)
        if not os.path.isdir(lesion_dir):
            continue

        print(f"Processing {lesion_type} images...")

        # Process each image in the lesion folder
        for img_name in tqdm(os.listdir(lesion_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(lesion_dir, img_name)

            try:
                # Read and preprocess image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                # Remove hair from the image using remove_hair_with_visualization function
                #cleaned_image = await remove_hair_with_visualization(image)
                #image = cleaned_image['result']  # Get the cleaned image from the result key

                #processor = ImageProcessing(image)

                #thresholded = processor.global_thresholding()

                #result = processor.apply_mask(thresholded, reverse=True)

                # Extract all features
                feature_vector, feature_dict = extractor.extract_feature_vector(image)

                # Store features and label
                all_features.append(feature_dict)  # Store the dictionary with named features
                labels.append(lesion_type)
                image_paths.append(img_path)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    # Create DataFrame with features
    df = pd.DataFrame(all_features)

    # Add labels and image paths
    df['label'] = labels
    df['image_path'] = image_paths

    # Save features to CSV
    output_file = os.path.join(os.path.dirname(__file__), 'lesion_features.csv')
    df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to {output_file}")
    print(f"Total samples processed: {len(df)}")
    print("\nFeature statistics:")
    print(df.describe())


import cv2
import os
import pandas as pd
import numpy as np


async def not_main():
    # Initialize feature extractor
    extractor = LesionFeatureExtractor()

    # Lists to store features and labels
    all_features = []
    labels = []
    image_paths = []

    img_path = "./images/ISIC_2018/Train/melanoma/ISIC_0000139.jpg"

    try:
        # Read and preprocess image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")

        # Remove hair from the image using remove_hair_with_visualization function
        cleaned_image = await remove_hair_with_visualization(image)
        image = cleaned_image['result']  # Get the cleaned image from the result key

        processor = ImageProcessing(image)
        thresholded = processor.global_thresholding()
        result = processor.apply_mask(thresholded, reverse=True)

        # Extract all features
        _, feature_dict = extractor.extract_feature_vector(result)

        # Store features and label
        all_features.append(feature_dict)  # Store the dictionary with named features
        labels.append("melanoma")  # Fixed typo in "melanoma"
        image_paths.append(img_path)

        # Create DataFrame from collected features
        df = pd.DataFrame(all_features)

        # Add labels and image paths as columns
        df['label'] = labels
        df['image_path'] = image_paths

        # Save features to CSV
        output_file = os.path.join(os.path.dirname(__file__), 'lesion_features.csv')
        df.to_csv(output_file, index=False)

        print(f"\nFeatures saved to {output_file}")
        print(f"Total samples processed: {len(df)}")
        print("\nFeature statistics:")
        print(df.describe())

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == '__main__':
    asyncio.run(main())
