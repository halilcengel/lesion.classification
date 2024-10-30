from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os


class HairRemovalProcessor:
    """Class for handling hair removal from dermoscopic images."""

    @staticmethod
    async def process_image(file: UploadFile) -> dict:
        """
        Process uploaded image file and remove hair artifacts.

        Parameters:
        file (UploadFile): FastAPI UploadFile object containing the image

        Returns:
        dict: Dictionary containing processed image data encoded as base64 strings
        """
        # Create temporary file to store upload
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        try:
            # Process image using temporary file
            results = HairRemovalProcessor._remove_hair(temp_file_path)

            # Convert images to base64 for API response
            encoded_results = {}
            for key, img in results.items():
                if img is not None:
                    _, buffer = cv2.imencode('.png', img)
                    encoded_results[key] = buffer.tobytes()

            return encoded_results

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    @staticmethod
    def _remove_hair(image_path: str) -> dict:
        """
        Remove hair artifacts from dermoscopic images using morphological operations.

        Parameters:
        image_path (str): Path to the input image

        Returns:
        dict: Dictionary containing various stages of processing

        Raises:
        HTTPException: If image cannot be read or processed
        """
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read uploaded image")

        # Image cropping
        img = image[30:410, 30:560]

        # Convert to grayscale
        grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply blackhat filter to detect dark hair
        kernel = cv2.getStructuringElement(1, (9, 9))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        # Apply Gaussian blur to reduce noise
        bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)

        # Create binary mask of hair locations
        ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)

        # Inpaint to remove hair
        clean = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)

        return {
            'original': image,
            'cropped': img,
            'grayscale': grayScale,
            'blackhat': blackhat,
            'mask': mask,
            'clean': clean
        }