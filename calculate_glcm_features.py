from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2
from typing import Dict
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt


def calculate_glcm_features(gray_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate GLCM texture features from grayscale image

    Args:
        gray_image: Grayscale image
        mask: Binary mask of the lesion

    Returns:
        Dictionary containing GLCM texture features
    """
    try:
        # Apply mask to get lesion area only
        masked_image = gray_image.copy()
        masked_image[mask == 0] = 0

        # Normalize to fewer gray levels to make GLCM computation more efficient
        bins = 16  # Number of gray levels
        normalized_image = ((masked_image / masked_image.max()) * (bins - 1)).astype(np.uint8)

        # Calculate GLCM for multiple directions
        distances = [1]  # Distance between pixel pairs
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
        glcm = graycomatrix(normalized_image, distances, angles, bins, symmetric=True, normed=True)

        # Calculate GLCM properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        ASM = graycoprops(glcm, 'ASM').mean()

        # Calculate additional statistics from mean GLCM
        mean_glcm = glcm.mean(axis=3)[:, :, 0]  # Average over all directions

        # Calculate sum and difference variance
        sum_variance = 0
        diff_variance = 0
        sum_entropy = 0
        diff_entropy = 0
        sum_avg = 0

        rows, cols = mean_glcm.shape
        for i in range(rows):
            for j in range(cols):
                # Sum variance and average
                sum_ij = i + j
                sum_variance += (sum_ij - sum_avg) ** 2 * mean_glcm[i, j]
                sum_avg += sum_ij * mean_glcm[i, j]

                # Difference variance
                diff_ij = abs(i - j)
                diff_variance += diff_ij ** 2 * mean_glcm[i, j]

                # Entropies (avoiding log(0))
                if mean_glcm[i, j] > 0:
                    sum_entropy -= mean_glcm[i, j] * np.log2(mean_glcm[i, j])
                    diff_entropy -= mean_glcm[i, j] * np.log2(mean_glcm[i, j])

        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

        # Original masked image
        ax1.imshow(masked_image, cmap='gray')
        ax1.set_title('Masked Image')
        ax1.axis('off')

        # GLCM visualization for 0° direction
        ax2.imshow(glcm[:, :, 0, 0], cmap='hot')
        ax2.set_title('GLCM (0°)')
        ax2.axis('off')

        # Feature distribution
        feature_names = ['Contrast', 'Homogeneity', 'Energy', 'Correlation']
        feature_values = [contrast, homogeneity, energy, correlation]
        ax3.bar(feature_names, feature_values)
        ax3.set_title('GLCM Features')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # Surface plot of GLCM
        X, Y = np.meshgrid(range(bins), range(bins))
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.plot_surface(X, Y, glcm[:, :, 0, 0], cmap='viridis')
        ax4.set_title('GLCM 3D Surface')

        # Save visualization
        temp_path = "images/texture_features.png"
        plt.tight_layout()
        plt.savefig(temp_path)
        plt.close()

        return {
            "contrast": float(contrast),
            "dissimilarity": float(dissimilarity),
            "homogeneity": float(homogeneity),
            "energy": float(energy),
            "correlation": float(correlation),
            "ASM": float(ASM),
            "sum_variance": float(sum_variance),
            "diff_variance": float(diff_variance),
            "sum_entropy": float(sum_entropy),
            "diff_entropy": float(diff_entropy),
            "visualization_path": temp_path
        }

    except Exception as e:
        raise ValueError(f"Error calculating texture features: {str(e)}")