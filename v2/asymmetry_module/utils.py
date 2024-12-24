import cv2
import numpy as np

def rotate_image(image):
    moments = cv2.moments(image)
    if moments['mu20'] != moments['mu02']:
        theta = 0.5 * np.arctan2(2 * moments['mu11'],
                                 moments['mu20'] - moments['mu02'])
    else:
        theta = 0

    height, width = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(theta), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def split_vertically(image):
    height, width = image.shape
    rotated_image = rotate_image(image)
    left_half = rotated_image[:, :width // 2]
    right_half = rotated_image[:, width // 2:]

    return left_half, right_half

def split_horizontally(image):
    height, width = image.shape
    rotated_image = rotate_image(image)
    top_half = rotated_image[:height // 2, :]
    bottom_half = rotated_image[height // 2:, :]

    return top_half, bottom_half
