import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
import warnings


class ColorAsymmetryAnalyzer:
    def __init__(self, n_segments: int = 50, compactness: int = 10, color_threshold: float = 1e-3):
        """
        Initialize the asymmetry analyzer.

        Args:
            n_segments: Number of segments for SLIC algorithm
            compactness: Compactness parameter for SLIC
            color_threshold: Threshold for color similarity comparison
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.color_threshold = color_threshold

    def calculate_median_color(self, image: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Calculate median color for each superpixel region.

        Args:
            image: Input image array
            labels: Superpixel labels from SLIC

        Returns:
            List of median colors for each region
        """
        num_labels = labels.max() + 1
        median_colors = []

        for region_id in range(num_labels):
            region_mask = labels == region_id
            region_pixels = image[region_mask]
            if len(region_pixels) > 0:
                median_val = np.median(region_pixels, axis=0)
                median_colors.append(median_val)
            else:
                median_colors.append(np.zeros(image.shape[-1]))

        return median_colors

    def compare_medians(self, medians_left: List[np.ndarray],
                        medians_right: List[np.ndarray]) -> Tuple[int, int]:
        """
        Compare median colors between two halves of the image.

        Args:
            medians_left: List of median colors from left/top half
            medians_right: List of median colors from right/bottom half

        Returns:
            Tuple of (symmetric_count, asymmetric_count)
        """
        symmetric_count = 0
        asymmetric_count = 0

        min_length = min(len(medians_left), len(medians_right))
        for i in range(min_length):
            if np.allclose(medians_left[i], medians_right[i], atol=self.color_threshold):
                symmetric_count += 1
            else:
                asymmetric_count += 1

        extra_regions = abs(len(medians_left) - len(medians_right))
        asymmetric_count += extra_regions

        return symmetric_count, asymmetric_count

    def analyze_image(self, image) -> Tuple[bool, bool, dict]:
        """
        Analyze both horizontal and vertical asymmetry in the image.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (horizontal_asymmetric, vertical_asymmetric, analysis_details)
        """
        height, width, _ = image.shape
        mid_h, mid_w = height // 2, width // 2

        # Split image
        top_half = image[:mid_h, :]
        bottom_half = image[mid_h:, :]
        left_half = image[:, :mid_w]
        right_half = image[:, mid_w:]

        # Generate superpixels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels_pairs = {
                'horizontal': (
                    slic(top_half, n_segments=self.n_segments, compactness=self.compactness),
                    slic(bottom_half, n_segments=self.n_segments, compactness=self.compactness)
                ),
                'vertical': (
                    slic(left_half, n_segments=self.n_segments, compactness=self.compactness),
                    slic(right_half, n_segments=self.n_segments, compactness=self.compactness)
                )
            }

        # Calculate medians and compare
        results = {}
        for direction, (labels1, labels2) in labels_pairs.items():
            medians1 = self.calculate_median_color(
                image[:mid_h, :] if direction == 'horizontal' else image[:, :mid_w],
                labels1
            )
            medians2 = self.calculate_median_color(
                image[mid_h:, :] if direction == 'horizontal' else image[:, mid_w:],
                labels2
            )
            sym_count, asym_count = self.compare_medians(medians1, medians2)
            results[direction] = {
                'symmetric_count': sym_count,
                'asymmetric_count': asym_count,
                'is_asymmetric': asym_count > sym_count
            }

        return (
            results['horizontal']['is_asymmetric'],
            results['vertical']['is_asymmetric'],
            results
        )

    def draw_boundaries(self, image: np.ndarray, labels: np.ndarray,
                        color: tuple = (1, 1, 0), thickness: float = 2.0) -> np.ndarray:
        """
        Draw superpixel boundaries on the image.

        Args:
            image: Input image
            labels: Superpixel labels
            color: RGB color tuple for boundaries (default: yellow)
            thickness: Line thickness

        Returns:
            Image with superpixel boundaries
        """
        # Normalize image if needed
        if image.dtype != np.float64:
            image = image.astype(np.float64) / 255.0

        # Draw boundaries
        marked_img = mark_boundaries(
            image,
            labels,
            color=color,
            mode='thick',
            background_label=0
        )

        # Enhance line visibility
        marked_img = np.clip(marked_img * thickness, 0, 1)
        return marked_img

    def create_asymmetry_map(self, image: np.ndarray, labels1: np.ndarray,
                             labels2: np.ndarray, direction: str) -> np.ndarray:
        """
        Create a color-coded asymmetry map showing matching and non-matching regions.

        Args:
            image: Input image
            labels1: Superpixel labels for first half
            labels2: Superpixel labels for second half
            direction: 'horizontal' or 'vertical'

        Returns:
            Color-coded asymmetry map
        """
        height, width = image.shape[:2]
        asymmetry_map = np.zeros((height, width, 3), dtype=np.float32)

        # Calculate medians for both halves
        if direction == 'horizontal':
            mid = height // 2
            medians1 = self.calculate_median_color(image[:mid], labels1)
            medians2 = self.calculate_median_color(image[mid:], labels2)
        else:
            mid = width // 2
            medians1 = self.calculate_median_color(image[:, :mid], labels1)
            medians2 = self.calculate_median_color(image[:, mid:], labels2)

        # Color code the regions
        for i in range(max(len(medians1), len(medians2))):
            if i < len(medians1) and i < len(medians2):
                # Green for matching regions, red for non-matching
                is_match = np.allclose(medians1[i], medians2[i], atol=self.color_threshold)
                color = np.array([0, 1, 0]) if is_match else np.array([1, 0, 0])

                if direction == 'horizontal':
                    asymmetry_map[:mid][labels1 == i] = color
                    asymmetry_map[mid:][labels2 == i] = color
                else:
                    asymmetry_map[:, :mid][labels1 == i] = color
                    asymmetry_map[:, mid:][labels2 == i] = color
            else:
                # Yellow for unmatched regions (different number of superpixels)
                color = np.array([1, 1, 0])
                if i < len(medians1):
                    if direction == 'horizontal':
                        asymmetry_map[:mid][labels1 == i] = color
                    else:
                        asymmetry_map[:, :mid][labels1 == i] = color
                if i < len(medians2):
                    if direction == 'horizontal':
                        asymmetry_map[mid:][labels2 == i] = color
                    else:
                        asymmetry_map[:, mid:][labels2 == i] = color

        return asymmetry_map

    def visualize_results(self, image: np.ndarray, save_path: str = None) -> None:
        """
        Asimetri analizinin kapsamlı görselleştirmesini oluşturur.

        Args:
            image_path: Girdi görüntüsünün yolu
            save_path: Görselleştirmeyi kaydetmek için isteğe bağlı yol
        """
        height, width = image.shape[:2]
        mid_h, mid_w = height // 2, width // 2

        # SLIC segmentasyonlarını oluştur
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = {
                'top': slic(image[:mid_h, :], n_segments=self.n_segments, compactness=self.compactness),
                'bottom': slic(image[mid_h:, :], n_segments=self.n_segments, compactness=self.compactness),
                'left': slic(image[:, :mid_w], n_segments=self.n_segments, compactness=self.compactness),
                'right': slic(image[:, mid_w:], n_segments=self.n_segments, compactness=self.compactness)
            }

        # Asimetri haritalarını oluştur
        h_asymmetry_map = self.create_asymmetry_map(
            image, labels['top'], labels['bottom'], 'horizontal'
        )
        v_asymmetry_map = self.create_asymmetry_map(
            image, labels['left'], labels['right'], 'vertical'
        )

        # Görselleştirmeyi oluştur
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3)

        # Süperpiksel sınırları ile orijinal görüntü
        ax_orig = fig.add_subplot(gs[0, 0])

        # Birleştirilmiş sınırlar görselleştirmesini oluştur
        image_float = image.astype(np.float64) / 255.0

        # Her yarım için farklı renklerle sınırları çiz
        if direction == 'horizontal':
            top_overlay = self.draw_boundaries(
                image_float[:mid_h, :],
                labels['top'],
                color=(1, 1, 0),  # Sarı
                thickness=1.5
            )
            bottom_overlay = self.draw_boundaries(
                image_float[mid_h:, :],
                labels['bottom'],
                color=(0, 1, 1),  # Camgöbeği
                thickness=1.5
            )

            # Yarımları birleştir
            combined_img = np.vstack([top_overlay, bottom_overlay])
        else:
            left_overlay = self.draw_boundaries(
                image_float[:, :mid_w],
                labels['left'],
                color=(1, 0.5, 0),  # Turuncu
                thickness=1.5
            )
            right_overlay = self.draw_boundaries(
                image_float[:, mid_w:],
                labels['right'],
                color=(0.5, 0, 1),  # Mor
                thickness=1.5
            )

            # Yarımları birleştir
            combined_img = np.hstack([left_overlay, right_overlay])

        ax_orig.imshow(combined_img)
        ax_orig.set_title("Süperpiksel Sınırları ile Orijinal Görüntü")
        ax_orig.axis("off")

        # Yatay analiz
        ax_h = fig.add_subplot(gs[0, 1])
        ax_h.imshow(h_asymmetry_map)
        ax_h.set_title("Yatay Asimetri Analizi")
        ax_h.axis("off")

        # Dikey analiz
        ax_v = fig.add_subplot(gs[0, 2])
        ax_v.imshow(v_asymmetry_map)
        ax_v.set_title("Dikey Asimetri Analizi")
        ax_v.axis("off")

        # SLIC segmentasyonları
        ax_top = fig.add_subplot(gs[1, 0])
        ax_top.imshow(labels['top'], cmap='nipy_spectral')
        ax_top.set_title("Üst Yarı Segmentasyonu")
        ax_top.axis("off")

        ax_bottom = fig.add_subplot(gs[1, 1])
        ax_bottom.imshow(labels['bottom'], cmap='nipy_spectral')
        ax_bottom.set_title("Alt Yarı Segmentasyonu")
        ax_bottom.axis("off")

        # Açıklama
        ax_legend = fig.add_subplot(gs[1, 2])
        ax_legend.axis("off")
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='green', label='Eşleşen Bölgeler'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Eşleşmeyen Bölgeler'),
            plt.Rectangle((0, 0), 1, 1, facecolor='yellow', label='Eşleştirilmemiş Bölgeler'),
            plt.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.5, label='Üst/Sol Sınırlar'),
            plt.Rectangle((0, 0), 1, 1, facecolor='cyan', alpha=0.5, label='Alt/Sağ Sınırlar')
        ]
        ax_legend.legend(handles=legend_elements, loc='center')
        ax_legend.set_title("Renk Açıklamaları")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Usage example:
if __name__ == "__main__":
    analyzer = ColorAsymmetryAnalyzer(n_segments=200, compactness=10)
    img = cv2.imread('../images/segmented_images/ISIC_0000042_masked.png', cv2.IMREAD_COLOR)

    try:
        h_asym, v_asym, details = analyzer.analyze_image(img)
        print(f"Horizontal Asymmetry: {h_asym}")
        print(f"Vertical Asymmetry: {v_asym}")
        print("\nDetailed Results:")
        for direction, results in details.items():
            print(f"\n{direction.title()} Analysis:")
            print(f"  Symmetric regions: {results['symmetric_count']}")
            print(f"  Asymmetric regions: {results['asymmetric_count']}")

        analyzer.visualize_results(img)

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
