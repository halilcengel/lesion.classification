import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from scipy import stats


def analyze_color_asymmetry(image, roi_mask):
    """
    Analyze color asymmetry in lesion image using SLIC superpixels.

    Parameters:
    image: np.array
        RGB image of the lesion
    roi_mask: np.array
        Binary mask defining the lesion region

    Returns:
    dict:
        Results containing asymmetry analysis for both axes
    """
    # Get ROI dimensions
    height, width = roi_mask.shape
    mid_h = width // 2
    mid_v = height // 2

    # Split ROI into halves
    left_mask = roi_mask[:, :mid_h]
    right_mask = roi_mask[:, mid_h:]
    top_mask = roi_mask[:mid_v, :]
    bottom_mask = roi_mask[mid_v:, :]

    def get_superpixels(img, mask):
        """Generate superpixels for image region"""
        # Calculate number of superpixels based on ROI size
        n_pixels = np.sum(mask)
        n_superpixels = max(1, n_pixels // 32)  # minimum 32 pixels per superpixel

        # Generate SLIC superpixels
        segments = slic(img, n_segments=n_superpixels, mask=mask,
                        start_label=0, compactness=10)
        return segments

    def get_region_colors(img, segments):
        """Get median RGB values for each superpixel region"""
        props = regionprops(segments + 1)  # Add 1 to avoid 0 label
        colors = []

        for prop in props:
            region_mask = segments == (prop.label - 1)
            r = np.median(img[:, :, 0][region_mask])
            g = np.median(img[:, :, 1][region_mask])
            b = np.median(img[:, :, 2][region_mask])
            colors.append((r, g, b))

        return colors

    def compare_color_regions(colors1, colors2):
        """Compare color regions between two halves"""
        symmetric_count = 0
        color_threshold = 10  # RGB difference threshold

        # For each color in first half
        for c1 in colors1:
            # Look for matching color in second half
            for c2 in colors2:
                if np.all(np.abs(np.array(c1) - np.array(c2)) <= color_threshold):
                    symmetric_count += 1
                    break

        return symmetric_count

    # Analyze horizontal asymmetry
    left_segments = get_superpixels(image[:, :mid_h], left_mask)
    right_segments = get_superpixels(image[:, mid_h:], right_mask)

    left_colors = get_region_colors(image[:, :mid_h], left_segments)
    right_colors = get_region_colors(image[:, mid_h:], right_segments)

    h_symmetric_regions = compare_color_regions(left_colors, right_colors)
    h_total_regions = len(left_colors) + len(right_colors)
    h_asymmetric_regions = h_total_regions - (2 * h_symmetric_regions)

    # Analyze vertical asymmetry
    top_segments = get_superpixels(image[:mid_v, :], top_mask)
    bottom_segments = get_superpixels(image[mid_v:, :], bottom_mask)

    top_colors = get_region_colors(image[:mid_v, :], top_segments)
    bottom_colors = get_region_colors(image[mid_v:, :], bottom_segments)

    v_symmetric_regions = compare_color_regions(top_colors, bottom_colors)
    v_total_regions = len(top_colors) + len(bottom_colors)
    v_asymmetric_regions = v_total_regions - (2 * v_symmetric_regions)

    # Determine overall asymmetry
    h_is_asymmetric = h_asymmetric_regions > (2 * h_symmetric_regions)
    v_is_asymmetric = v_asymmetric_regions > (2 * v_symmetric_regions)

    return {
        'horizontal_asymmetry': {
            'symmetric_regions': h_symmetric_regions,
            'asymmetric_regions': h_asymmetric_regions,
            'is_asymmetric': h_is_asymmetric
        },
        'vertical_asymmetry': {
            'symmetric_regions': v_symmetric_regions,
            'asymmetric_regions': v_asymmetric_regions,
            'is_asymmetric': v_is_asymmetric
        }
    }


if __name__ == "__main__":
    import cv2

    image = cv2.imread('clean_images/ISIC_0024320.jpg')
    mask = cv2.imread('segmentation_v2_masks/ISIC_0024320_segmented.png', 0)

    results = analyze_color_asymmetry(image, mask)

    print(f"Horizontal asymmetry: {results['horizontal_asymmetry']['is_asymmetric']}")
    print(f"Vertical asymmetry: {results['vertical_asymmetry']['is_asymmetric']}")
