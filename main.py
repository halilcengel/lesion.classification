from fastapi import FastAPI, File, UploadFile, HTTPException
import logging
import tempfile
import os

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from segmentation_v2 import ImageProcessing
from hair_removal import demo_hair_removal
from asymmetry_module.shape_asymmetry import rotate_image, calculate_asymmetry
from border_irregularity_module.old_main import calculate_total_b_score
from border_irregularity_module.utils import detect_border

from calculate_tds import calculate_tds

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# USE REMOVE_HAIR PLZ :D

@app.post("/remove-hair")
async def remove_hair(file: UploadFile = File(...), include_steps: bool = False):
    """
    Endpoint to remove hair from an image.

    Parameters:
    - file: Image file to process
    - include_steps: If True, returns visualization steps (default: False)

    Returns:
    - Processed image and optionally visualization steps
    """
    try:
        # First read and convert the uploaded file to numpy array
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process the image with numpy array instead of UploadFile
        result = await demo_hair_removal(img)

        if include_steps:
            # Convert steps to base64 for JSON response
            steps_encoded = {}
            for step_name, step_image in result['steps'].items():
                _, buffer = cv2.imencode('.png', step_image)
                steps_encoded[step_name] = buffer.tobytes()

            return JSONResponse({
                'steps': steps_encoded,
                'original_shape': result['original_shape']
            })
        else:
            # Return just the cleaned image
            return StreamingResponse(
                io.BytesIO(result['clean']),
                media_type="image/png"
            )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-lesion")
async def remove_hair(file: UploadFile = File(...)):
    """
    Endpoint to remove hair from an image.

    Parameters:
    - file: Image file to process

    Returns:
    - Processed image with hair removed
    """
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Use original ImageProcessing class with file path
        processor = ImageProcessing(temp_file_path)

        # Create mask using thresholding
        thresholded = processor.global_thresholding()

        # Apply mask to remove hair
        result = processor.apply_mask(thresholded, reverse=True)

        # Convert result to bytes for response
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode resulting image")

        # Clean up temporary file
        os.unlink(temp_file_path)

        # Create response
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        # Clean up temp file in case of error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import logging
from starlette.responses import Response


@app.post("/shape-asymmetry")
async def shape_asymmetry(file: UploadFile = File(...)):
    """
    Endpoint to calculate shape asymmetry of a lesion.

    Parameters:
    - file: Image file to process

    Returns:
    - StreamingResponse with visualization and asymmetry percentages in headers
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_color is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale for processing
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape

        # Process grayscale image for asymmetry calculations
        rotated_image_gray = rotate_image(img_gray)
        vertical_diff_percentage = calculate_asymmetry(rotated_image_gray, split_by='vertical')
        horizontal_diff_percentage = calculate_asymmetry(rotated_image_gray, split_by='horizontal')

        # Create visualization on color image
        split_viz = img_color.copy()
        cv2.line(split_viz, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
        cv2.line(split_viz, (0, height // 2), (width, height // 2), (0, 255, 0), 2)

        # Encode the visualization image
        _, buffer = cv2.imencode('.png', split_viz)

        # Create response with headers
        headers = {
            "X-Vertical-Asymmetry": f"{float(vertical_diff_percentage):.2f}",
            "X-Horizontal-Asymmetry": f"{float(horizontal_diff_percentage):.2f}",
            "Access-Control-Expose-Headers": "X-Vertical-Asymmetry, X-Horizontal-Asymmetry"
        }

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png",
            headers=headers
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/border-irregularity")
async def border_irregularity(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED instead of IMREAD_COLOR

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        irregularity_score = calculate_total_b_score(img)

        # Create visualization
        border_image = detect_border(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        border_viz = np.stack((border_image,) * 3, axis=-1)

        # Encode the visualization image
        _, buffer = cv2.imencode('.png', border_viz)

        headers = {
            "X-Border-Irregularity": f"{float(irregularity_score):.2f}",
            "Access-Control-Expose-Headers": "X-Border-Irregularity"
        }

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png",
            headers=headers
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-lesion")
async def process_lesion(file: UploadFile = File(...)):
    """
    Endpoint to process an image of a lesion and calculate TDS.

    Parameters:
    - file: Image file to process

    Returns:
    - JSON response with TDS and classification
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Calculate TDS
        tds_result = calculate_tds(img)

        return JSONResponse(tds_result)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
