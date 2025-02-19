import cv2

from border_irregularity_module import utils


def calculate_total_b_score(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_border = utils.detect_border(image_gray)

    segments = utils.split_border_into_segments(image_border, num_segments=8)

    irregularity_score = utils.compute_border_irregularity(segments)

    print(f"Border irregularity score: {irregularity_score}/8")
    return irregularity_score


if __name__ == "__main__":
    image = cv2.imread('../images/segmented_images/ISIC_0000171_masked.png', cv2.IMREAD_COLOR)
    image_mask = cv2.imread('../images/segmentation_masks/ISIC_0000171_mask.png', cv2.IMREAD_COLOR)

    calculate_total_b_score(image_mask)
    calculate_total_b_score(image)
