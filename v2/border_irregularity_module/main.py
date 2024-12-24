import cv2

from v2.border_irregularity_module import utils


def main():
    image = cv2.imread('../segmentation_v2_masks/ISIC_0000042_segmented.png', cv2.IMREAD_GRAYSCALE)

    image_border = utils.detect_border(image)

    cv2.imwrite('border.png', image_border)

    segments = utils.split_border_into_segments(image_border, num_segments=8)

    irregularity_score = utils.compute_border_irregularity(segments)

    print(f"Border irregularity score: {irregularity_score}/8")


if __name__ == "__main__":
    main()