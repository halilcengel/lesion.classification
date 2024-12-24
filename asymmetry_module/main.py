import cv2

from asymmetry_module.brightness_asymmetry import calculate_brightness_asymmetry
from asymmetry_module.color_asymmetry import ColorAsymmetryAnalyzer
from asymmetry_module.shape_asymmetry import calculate_asymmetry


def calculate_total_a_score(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parlaklık asimetrisi hesaplama
    v_bright, h_bright = calculate_brightness_asymmetry(gray_image)
    brightness_asymmetry = {
        'vertical': v_bright > 0.03,  # %3'ten fazla fark varsa asimetrik
        'horizontal': h_bright > 0.03
    }

    # Renk asimetrisi hesaplama
    color_analyzer = ColorAsymmetryAnalyzer(n_segments=200, compactness=10)
    h_color_asym, v_color_asym, _ = color_analyzer.analyze_image(image)
    color_asymmetry = {
        'vertical': v_color_asym,
        'horizontal': h_color_asym
    }

    # Şekil asimetrisi hesaplama
    v_shape = calculate_asymmetry(gray_image, split_by='vertical')
    h_shape = calculate_asymmetry(gray_image, split_by='horizontal')
    shape_asymmetry = {
        'vertical': v_shape > 0.02,  # %2'den fazla fark varsa asimetrik
        'horizontal': h_shape > 0.02
    }

    # Her eksen için asimetri sayısını hesaplama
    v_asymmetries = sum([shape_asymmetry['vertical'],
                         brightness_asymmetry['vertical'],
                         color_asymmetry['vertical']])

    h_asymmetries = sum([shape_asymmetry['horizontal'],
                         brightness_asymmetry['horizontal'],
                         color_asymmetry['horizontal']])

    # A skorunu hesaplama
    a_score = 0

    # Tek eksen asimetrisi (dikey VEYA yatay)
    if (v_asymmetries > 0) != (h_asymmetries > 0):  # XOR
        if v_asymmetries == 1 or h_asymmetries == 1:
            a_score = 0.25  # Tek parametre
        elif v_asymmetries == 2 or h_asymmetries == 2:
            a_score = 0.5  # İki parametre
        elif v_asymmetries == 3 or h_asymmetries == 3:
            a_score = 1.0  # Üç parametre

    # İki eksen asimetrisi (hem dikey HEM yatay)
    elif v_asymmetries > 0 and h_asymmetries > 0:
        max_asymmetries = max(v_asymmetries, h_asymmetries)
        if max_asymmetries == 1:
            a_score = 1.25  # Tek parametre
        elif max_asymmetries == 2:
            a_score = 1.5  # İki parametre
        elif max_asymmetries == 3:
            a_score = 2.0  # Üç parametre

    return a_score


if __name__ == '__main__':
    # Test image
    img = cv2.imread('../images/segmented_images/ISIC_0024306_masked.png')

    # Calculate total A score
    a_score = calculate_total_a_score(img)
    print('Total A score:', a_score)
