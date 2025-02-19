import cv2

from asymmetry_module.main import calculate_total_a_score
from border_irregularity_module.main import calculate_total_b_score
from color import ColorInformationExtractor
from differential_structures_module.main import calculate_total_d_score


def calculate_tds(image):
    """
    Calculate total dermatoscopic score (TDS) for a given image

    TDS = A × 1.3 + B × 0.1 + C × 0.5 + D × 0.5

    Returns:
    - tds: float - Total dermatoscopic score
    - category: str - Classification based on TDS value
    """

    # Calculate individual scores
    a_score = calculate_total_a_score(image)  # 0-2
    b_score = calculate_total_b_score(image)  # 0-8

    color_extractor = ColorInformationExtractor()
    color_counts = color_extractor.extract_colors(image)

    c_score = color_extractor.calculate_color_score(color_counts)

    d_score, pn_percentage, dg_percentage, sa_percentage, bw_percentage = calculate_total_d_score(image)  # 0-5

    # Calculate TDS using weights from ABCD rule
    tds = (a_score * 1.3) + (b_score * 0.1) + (c_score * 0.5) + (d_score * 0.5)

    # Classify based on TDS value
    if tds < 4.75:
        category = "Benign"
    elif 4.75 <= tds <= 5.45:
        category = "Suspicious"
    else:
        category = "Malignant"

    return {
        'tds': round(tds, 2),
        'category': category,
        'scores': {
            'asymmetry': a_score,
            'border': b_score,
            'color': c_score,
            'differential_structures': d_score,
            'differential_structures_details': {
                'pigment_network_percentage': pn_percentage,
                'dots_globules_percentage': dg_percentage,
                'structureless_percentage': sa_percentage,
                'blue_white_veil_percentage': bw_percentage
            }
        }
    }


if __name__ == '__main__':
    # Usage
    img = cv2.imread('images/ISIC_2018/Train_masked_lesions/melanoma/ISIC_0000139.jpg')

    # Calculate TDS
    result = calculate_tds(img)

    print(f"TDS Score: {result['tds']}")
    print(f"Category: {result['category']}")
    print("\nIndividual Scores:")
    for key, value in result['scores'].items():
       if isinstance(value, dict):
         print(f"{key}:")
         for sub_key, sub_value in value.items():
             print(f"  {sub_key}: {sub_value}")
       else:
             print(f"{key}: {value}")
