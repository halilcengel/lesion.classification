import cv2
from matplotlib import pyplot as plt

from v2.differential_structures_module.pigment_network_detection import detect_pigment_network
from v2.differential_structures_module.blue_white_veil_detection import detect_blue_white_veil
from v2.differential_structures_module.dots_and_globs_detection import detect_dots_globules
from v2.differential_structures_module.structless_area_detection import detect_structureless_area

def main():
    image_path = "../segmentation_v2_masked_images/ISIC_0024306_masked.png"
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    pigment_network = detect_pigment_network(img_gray)
    blue_white_veil = detect_blue_white_veil(image_path)
    dots_and_globs = detect_dots_globules(img_gray)
    structless_area = detect_structureless_area(img_gray)

    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Grayscale Dermoscopic Image')
    axes[0].axis('off')

    axes[1].imshow(pigment_network)
    axes[1].set_title('Detected Pigment Network')
    axes[1].axis('off')

    axes[2].imshow(blue_white_veil)
    axes[2].set_title('Blue White Veil Detection')
    axes[2].axis('off')

    axes[3].imshow(dots_and_globs)
    axes[3].set_title('Detected Dots & Globules')
    axes[3].axis('off')

    axes[4].imshow(structless_area)
    axes[4].set_title('Detected Structureless Area')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()