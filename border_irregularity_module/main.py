import cv2
import numpy as np
from skimage import measure


def analyze_border(image):
    """
    Cilt lezyonunun sınır özelliklerini analiz eden fonksiyon.

    Parametreler:
    image_path (str): Analiz edilecek görüntünün dosya yolu

    Dönüş:
    int: 8 üzerinden sınır skoru
    """
    # Görüntüyü oku
    if image is None:
        raise ValueError("Görüntü okunamadı")

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian bulanıklaştırma uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu eşikleme ile segmentasyon
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Lezyon konturları bulunamadı")

    # En büyük konturu al (ana lezyon)
    main_contour = max(contours, key=cv2.contourArea)

    # Sınır düzensizliği analizi
    perimeter = cv2.arcLength(main_contour, True)
    area = cv2.contourArea(main_contour)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    irregularity_score = 1 - circularity

    # Sınır asimetrisi analizi
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.int_(box)
    width = rect[1][0]
    height = rect[1][1]
    asymmetry_ratio = abs(1 - (width / height))

    # Kenarlık keskinliği analizi
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, (255), 1)
    edge_pixels = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
    edge_sharpness = np.mean(np.abs(edge_pixels[mask == 255]))

    # Risk skorunu hesapla (0-1 arası)
    risk_score = (irregularity_score * 0.4 + asymmetry_ratio * 0.4 + min(edge_sharpness / 100, 1) * 0.2)

    # 8'lik skalaya dönüştür ve döndür
    return round(risk_score * 8)


def calculate_total_b_score(image):
    irregularity_score = analyze_border(image)

    print(f"Border irregularity score: {irregularity_score}/8")
    return irregularity_score


# Kullanım örneği
if __name__ == "__main__":
    try:
        img = cv2.imread("../images/segmented_images/ISIC_0000171_masked.png")
        score = calculate_total_b_score(img)
        print(score)
    except Exception as e:
        print(f"Hata: {str(e)}")
