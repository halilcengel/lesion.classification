import asyncio
import os
import io
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from fastapi import UploadFile
from typing import Dict, Union


async def remove_hair_with_visualization(image: np.ndarray, kernel_size=20, threshold=25) -> Dict[str, np.ndarray]:
    """
    Remove hair-like structures from images and return intermediate steps
    for visualization.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale or color)
    kernel_size : int
        Size of the morphological kernel (should be odd)
    threshold : int
        Threshold value for hair detection

    Returns:
    --------
    dict
        Dictionary containing all intermediate steps and final result
    """
    steps = {'original': image.copy()}

    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        steps['grayscale'] = gray
    else:
        gray = image.copy()
        steps['grayscale'] = gray

    # Create structural element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply blackhat operation
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    steps['blackhat'] = blackhat

    # Threshold to create binary mask
    _, threshold_image = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    steps['threshold'] = threshold_image

    # Dilate the hair mask
    kernel_dilate = np.ones((2, 2), np.uint8)
    hair_mask = cv2.dilate(threshold_image, kernel_dilate, iterations=1)
    steps['mask'] = hair_mask

    # Apply inpainting
    if len(image.shape) == 3:
        result = cv2.inpaint(image, hair_mask, 5, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(gray, hair_mask, 5, cv2.INPAINT_TELEA)
    steps['result'] = result

    return steps


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to improve hair detection.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image

    Returns:
    --------
    numpy.ndarray
        Preprocessed image
    """
    # Convert to float and normalize
    img_float = image.astype(float) / 255.0

    # Apply contrast enhancement
    enhanced = np.clip((img_float - img_float.mean()) * 1.5 + img_float.mean(), 0, 1)

    # Convert back to uint8
    return (enhanced * 255).astype(np.uint8)


async def demo_hair_removal(file: Union[UploadFile, np.ndarray]) -> Dict[str, Union[bytes, np.ndarray]]:
    """
    Demonstrate hair removal process with visualization.
    Accepts either UploadFile or numpy array.

    Parameters:
    -----------
    file : Union[UploadFile, np.ndarray]
        Input image as either UploadFile or numpy array

    Returns:
    --------
    Dict[str, Union[bytes, np.ndarray]]
        Dictionary containing processed image and results
    """
    try:
        # Handle input type
        if isinstance(file, UploadFile):
            # Read image from UploadFile
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Input is already a numpy array
            image = file

        if image is None:
            raise ValueError("Could not read image")

        # Preprocess image
        preprocessed = preprocess_image(image)

        # Get all steps
        steps = await remove_hair_with_visualization(preprocessed)

        # Convert result to bytes
        _, img_encoded = cv2.imencode('.png', steps['result'])

        return {
            'clean': img_encoded.tobytes(),
            'steps': steps,
            'original_shape': image.shape
        }

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

async def process_single_image(input_path, output_path):
    """Process a single image and save the cleaned version."""
    try:
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Could not read image: {input_path}")
            return False

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = await demo_hair_removal(image)

        # Convert result back to displayable image
        clean_image_bytes = results['clean']
        clean_image = cv2.imdecode(
            np.frombuffer(clean_image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the cleaned image
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR)
        )
        print(f"Successfully processed: {input_path.name}")
        return True

    except Exception as e:
        print(f"Error processing {input_path.name}: {str(e)}")
        return False


async def batch_process_images():
    """Process all images in the input folder and save to output folder."""
    # Setup paths
    input_dir = Path("images")
    output_dir = Path("images/clean_images")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in input_dir.glob("*")
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print("No image files found in input directory!")
        return

    # Process statistics
    total = len(image_files)
    successful = 0
    failed = 0

    # Process each image
    for input_path in image_files:
        output_path = output_dir / input_path.name
        success = await process_single_image(input_path, output_path)

        if success:
            successful += 1
        else:
            failed += 1

    # Print summary
    print("\nProcessing Complete!")
    print(f"Total images: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    asyncio.run(batch_process_images())
