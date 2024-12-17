import numpy as np
import cv2
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from fastapi import HTTPException, UploadFile
from hair_removal_v10 import demo_hair_removal
import io


@dataclass
class LesionFeatures:
    """Data class to store all features in a structured way"""
    # [Previous dataclass definition remains the same]
    asymmetry_index_1: float
    asymmetry_index_2: float
    irregularity_A: float
    irregularity_B: float
    irregularity_C: float
    irregularity_D: float
    perimeter: float
    area: float
    l_mean: float
    a_mean: float
    b_mean: float
    l_std: float
    a_std: float
    b_std: float
    mean_color_difference: float
    max_color_difference: float
    contrast: float
    dissimilarity: float
    homogeneity: float
    energy: float
    correlation: float
    ASM: float
    sum_variance: float
    diff_variance: float
    sum_entropy: float
    diff_entropy: float
    mean_intensity: float
    std_intensity: float
    mean_gradient: float
    std_gradient: float

    def to_vector(self) -> np.ndarray:
        """[Previous method remains the same]"""
        return np.array([
            self.asymmetry_index_1,
            self.asymmetry_index_2,
            self.irregularity_A,
            self.irregularity_B,
            self.irregularity_C,
            self.irregularity_D,
            self.perimeter,
            self.area,
            self.l_mean,
            self.a_mean,
            self.b_mean,
            self.l_std,
            self.a_std,
            self.b_std,
            self.mean_color_difference,
            self.max_color_difference,
            self.contrast,
            self.dissimilarity,
            self.homogeneity,
            self.energy,
            self.correlation,
            self.ASM,
            self.sum_variance,
            self.diff_variance,
            self.sum_entropy,
            self.diff_entropy,
            self.mean_intensity,
            self.std_intensity,
            self.mean_gradient,
            self.std_gradient
        ])


class MelanomaFeatureCollector:
    """Coordinates all processing modules and collects features"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    async def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess image by removing hair and segmenting lesion"""
        try:
            # Now you can directly pass the numpy array
            results = await demo_hair_removal(image)

            # Get the clean image from results
            clean_image_bytes = results['clean']
            clean_image = cv2.imdecode(
                np.frombuffer(clean_image_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)

            # Create binary mask
            gray = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return {
                'clean_image': clean_image,
                'mask': mask > 0
            }
        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")


    async def collect_features(self, image: np.ndarray) -> LesionFeatures:
        """[Previous method implementation remains the same]"""
        try:
            # Import required modules
            from calculate_asymmetry import calculate_asymmetry
            from border_irregularity import calculate_border_irregularity
            from calculate_color_variation import calculate_color_variation
            from calculate_glcm_features import calculate_glcm_features
            from pigment_transition import calculate_pigment_transition

            # Preprocess image
            preprocessed = await self._preprocess_image(image)
            clean_image = preprocessed['clean_image']
            mask = preprocessed['mask']

            # Calculate asymmetry
            A1, A2 = calculate_asymmetry(mask)

            # Calculate border irregularity
            border_features = calculate_border_irregularity(mask.astype(np.uint8))

            # Calculate color variation
            color_features = calculate_color_variation(clean_image, mask)

            # Calculate GLCM features
            gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
            texture_features = calculate_glcm_features(gray, mask)

            # Calculate pigment transition
            pigment_features = calculate_pigment_transition(clean_image, mask)

            # Combine all features
            features = LesionFeatures(
                asymmetry_index_1=float(A1),
                asymmetry_index_2=float(A2),
                irregularity_A=float(border_features['irregularity_A']),
                irregularity_B=float(border_features['irregularity_B']),
                irregularity_C=float(border_features['irregularity_C']),
                irregularity_D=float(border_features['irregularity_D']),
                perimeter=float(border_features['perimeter']),
                area=float(border_features['area']),
                l_mean=float(color_features['l_mean']),
                a_mean=float(color_features['a_mean']),
                b_mean=float(color_features['b_mean']),
                l_std=float(color_features['l_std']),
                a_std=float(color_features['a_std']),
                b_std=float(color_features['b_std']),
                mean_color_difference=float(color_features['mean_color_difference']),
                max_color_difference=float(color_features['max_color_difference']),
                contrast=float(texture_features['contrast']),
                dissimilarity=float(texture_features['dissimilarity']),
                homogeneity=float(texture_features['homogeneity']),
                energy=float(texture_features['energy']),
                correlation=float(texture_features['correlation']),
                ASM=float(texture_features['ASM']),
                sum_variance=float(texture_features['sum_variance']),
                diff_variance=float(texture_features['diff_variance']),
                sum_entropy=float(texture_features['sum_entropy']),
                diff_entropy=float(texture_features['diff_entropy']),
                mean_intensity=float(pigment_features['mean_intensity']),
                std_intensity=float(pigment_features['std_intensity']),
                mean_gradient=float(pigment_features['mean_gradient']),
                std_gradient=float(pigment_features['std_gradient'])
            )

            return features

        except Exception as e:
            self.logger.error(f"Error collecting features: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error collecting features: {str(e)}")

    def get_feature_names(self) -> List[str]:
        """[Previous method remains the same]"""
        return [
            'asymmetry_index_1', 'asymmetry_index_2',
            'irregularity_A', 'irregularity_B', 'irregularity_C', 'irregularity_D',
            'perimeter', 'area',
            'l_mean', 'a_mean', 'b_mean', 'l_std', 'a_std', 'b_std',
            'mean_color_difference', 'max_color_difference',
            'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation',
            'ASM', 'sum_variance', 'diff_variance', 'sum_entropy', 'diff_entropy',
            'mean_intensity', 'std_intensity', 'mean_gradient', 'std_gradient'
        ]


async def main():
    collector = MelanomaFeatureCollector()
    img = cv2.imread("images/input/ISIC_0000264.jpg")
    if img is None:
        raise ValueError("Image not found or unable to read!")
    features = await collector.collect_features(img)
    print(features)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
