from fastapi import FastAPI, UploadFile, File, HTTPException
from hair_removal import HairRemovalProcessor
from lesion_segmentation import segment_lesion
from fastapi.responses import StreamingResponse
from combined_processor import DermoscopyImageProcessor, process_dermoscopy_image
from image_quality_assessor import process_and_assess_image

import io

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/remove-hair/")
async def remove_hair(file: UploadFile = File(...)):
    """
    Endpoint to remove hair from uploaded dermoscopic images.

    Parameters:
    file (UploadFile): The image file to process

    Returns:
    StreamingResponse: Processed images as binary data<
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    results = await HairRemovalProcessor.process_image(file)

    # Return the cleaned image as a streaming response
    return StreamingResponse(
        io.BytesIO(results['clean']),
        media_type="image/png"
    )


@app.post("/segment-lesion/")
async def segment_lesion_endpoint(file: UploadFile = File(...)):
    return await segment_lesion(file)


@app.post("/process-dermoscopy/")
async def process_dermoscopy_endpoint(
    input_file: UploadFile = File(...),
    reference_file: UploadFile = None
):
    return await process_and_assess_image(input_file, reference_file)
