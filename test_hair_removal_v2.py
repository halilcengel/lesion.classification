import cv2
import numpy as np
from hair_removal_v2 import removeHair


def bytes_to_numpy(image_bytes):
    """Convert bytes to numpy array"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the array into an image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error converting bytes to numpy array: {str(e)}")
        return None


def test_hair_removal():
        # Read input image
        input_path = 'test/input_images/ISIC_0000042.jpg'

        # Process image
        result = removeHair(input_path)

        # Save result
        output_path = 'output_image.jpg'    
        success = cv2.imwrite(output_path, result)

        if not success:
            raise RuntimeError("Failed to save output image")

        print(f"Successfully processed and saved image to {output_path}")

if __name__ == "__main__":
    test_hair_removal()