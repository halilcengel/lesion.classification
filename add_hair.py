import cv2
import numpy as np

def add_hair_noise(image, num_hairs=100, hair_thickness_range=(1, 3), hair_length_range=(20, 100)):
    """
    Görüntüye yapay kıl efekti ekler

    Parametreler:
    image: numpy array - Giriş görüntüsü (BGR formatında)
    num_hairs: int - Eklenecek kıl sayısı
    hair_thickness_range: tuple - Kıl kalınlığı aralığı (min, max)
    hair_length_range: tuple - Kıl uzunluğu aralığı (min, max)

    Dönüş:
    numpy array - Kıl efekti eklenmiş görüntü
    """
    # Görüntünün kopyasını al
    result = image.copy()
    h, w = image.shape[:2]

    # Her bir kıl için
    for _ in range(num_hairs):
        # Rastgele başlangıç noktası
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)

        # Rastgele açı (0-360 derece)
        angle = np.random.randint(0, 360)

        # Rastgele uzunluk
        length = np.random.randint(hair_length_range[0], hair_length_range[1])

        # Bitiş noktasını hesapla
        rad = np.deg2rad(angle)
        x2 = int(x1 + length * np.cos(rad))
        y2 = int(y1 + length * np.sin(rad))

        # Rastgele kalınlık
        thickness = np.random.randint(hair_thickness_range[0], hair_thickness_range[1])

        # Kılın rengini belirle (koyu kahverengi-siyah arası)
        color = (np.random.randint(0, 30), np.random.randint(0, 30), np.random.randint(0, 30))

        # Kılı çiz
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        # Kıla hafif bulanıklık ekle
        result = cv2.GaussianBlur(result, (3, 3), 0)

    return result


def apply_to_image(input_path, output_path, num_hairs=100):
    """
    Görüntü dosyasına kıl efekti uygular ve kaydeder

    Parametreler:
    input_path: str - Giriş görüntüsünün dosya yolu
    output_path: str - Çıkış görüntüsünün kaydedileceği dosya yolu
    num_hairs: int - Eklenecek kıl sayısı
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Görüntü okunamadı!")

    result = add_hair_noise(image, num_hairs)

    cv2.imwrite(output_path, result)


