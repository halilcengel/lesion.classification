import cv2
from matplotlib import pyplot as plt

from differential_structures_module.pigment_network_detection import detect_pigment_network
from differential_structures_module.blue_white_veil_detection import detect_blue_white_veil
from differential_structures_module.dots_and_globs_detection import detect_dots_globules
from differential_structures_module.structless_area_detection import detect_structureless_area
from differential_structures_module.utils import calculate_area_percentage


def calculate_total_d_score(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d_score = 0
    scores = {
        'pigment_network': 0,
        'dots_globules': 0,
        'structureless': 0,
        'blue_white_veil': 0
    }

    pigment_network = detect_pigment_network(img_gray)
    blue_white_veil = detect_blue_white_veil(img)
    dots_and_globs = detect_dots_globules(img_gray)
    structless_area = detect_structureless_area(img_gray)

    # Pigment ağı değerlendirmesi
    pn_percentage = calculate_area_percentage(pigment_network)
    if pn_percentage > 10:
        scores['pigment_network'] = 1
        if 10 < pn_percentage < 50:  # Lokalize dağılım
            scores['pigment_network'] += 0.5

    # Noktalar ve globüller değerlendirmesi
    dg_percentage = calculate_area_percentage(dots_and_globs)
    if dg_percentage > 10:
        scores['dots_globules'] = 1
        if 10 < dg_percentage < 50:  # Lokalize dağılım
            scores['dots_globules'] += 0.5

    # Yapısız alan değerlendirmesi
    sa_percentage = calculate_area_percentage(structless_area)
    if sa_percentage > 10:
        scores['structureless'] = 1

    # Mavi-beyaz peçe değerlendirmesi
    bw_percentage = calculate_area_percentage(blue_white_veil)
    if bw_percentage > 10:
        scores['blue_white_veil'] = 1
        if bw_percentage > 30:  # İyi huylu gösterge
            scores['blue_white_veil'] -= 0.5

    # Toplam D skorunu hesapla
    d_score = sum(scores.values())

    print("Pigment Network Area Percentage:", pn_percentage)
    print("Dots & Globules Area Percentage:", dg_percentage)
    print("Structureless Area Percentage:", sa_percentage)
    print("Blue-White Veil Area Percentage:", bw_percentage)
    print("D Score:", d_score)

    return d_score,pn_percentage,dg_percentage,sa_percentage,bw_percentageÏ


def calculate_and_visualize(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d_score = 0
    scores = {
        'pigment_network': 0,
        'dots_globules': 0,
        'structureless': 0,
        'blue_white_veil': 0
    }

    pigment_network = detect_pigment_network(img_gray)
    blue_white_veil = detect_blue_white_veil(img)
    dots_and_globs = detect_dots_globules(img_gray)
    structless_area = detect_structureless_area(img_gray)

    # Pigment ağı değerlendirmesi
    pn_percentage = calculate_area_percentage(pigment_network)
    if pn_percentage > 10:
        scores['pigment_network'] = 1
        if 10 < pn_percentage < 50:  # Lokalize dağılım
            scores['pigment_network'] += 0.5

    # Noktalar ve globüller değerlendirmesi
    dg_percentage = calculate_area_percentage(dots_and_globs)
    if dg_percentage > 10:
        scores['dots_globules'] = 1
        if 10 < dg_percentage < 50:  # Lokalize dağılım
            scores['dots_globules'] += 0.5

    # Yapısız alan değerlendirmesi
    sa_percentage = calculate_area_percentage(structless_area)
    if sa_percentage > 10:
        scores['structureless'] = 1

    # Mavi-beyaz peçe değerlendirmesi
    bw_percentage = calculate_area_percentage(blue_white_veil)
    if bw_percentage > 10:
        scores['blue_white_veil'] = 1
        if bw_percentage > 30:  # İyi huylu gösterge
            scores['blue_white_veil'] -= 0.5

    # Toplam D skorunu hesapla
    d_score = sum(scores.values())

    print("Pigment Network Area Percentage:", pn_percentage)
    print("Dots & Globules Area Percentage:", dg_percentage)
    print("Structureless Area Percentage:", sa_percentage)
    print("Blue-White Veil Area Percentage:", bw_percentage)
    print("D Score:", d_score)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title(f'Original Image\nTotal D Score: {d_score:.1f}')
    axes[0].axis('off')

    axes[1].imshow(pigment_network)
    axes[1].set_title(f'Pigment Network\nScore: {scores["pigment_network"]:.1f}')
    axes[1].axis('off')

    axes[2].imshow(blue_white_veil)
    axes[2].set_title(f'Blue White Veil\nScore: {scores["blue_white_veil"]:.1f}')
    axes[2].axis('off')

    axes[3].imshow(dots_and_globs)
    axes[3].set_title(f'Dots & Globules\nScore: {scores["dots_globules"]:.1f}')
    axes[3].axis('off')

    axes[4].imshow(structless_area)
    axes[4].set_title(f'Structureless Area\nScore: {scores["structureless"]:.1f}')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()
