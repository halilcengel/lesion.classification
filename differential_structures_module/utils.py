import numpy as np


def calculate_area_percentage(binary_mask):
    total_pixels = binary_mask.size
    structure_pixels = np.sum(binary_mask > 0)
    return (structure_pixels / total_pixels) * 100
