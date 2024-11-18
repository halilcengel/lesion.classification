from fastapi import FastAPI, UploadFile, File, HTTPException
from add_hair import apply_to_image
from lesion_segmentation import segment_lesion
from image_quality_assessor import process_and_assess_image
from hair_removal import remove_hair

import io

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/remove-hair")
async def remove_hair_endpoint(
    file: UploadFile = File(...),
    include_metrics: bool = True
):
 return await remove_hair(file, include_metrics)


@app.post("/segment-lesion")
async def segment_lesion_endpoint(file: UploadFile = File(...)):
    return await segment_lesion(file)


@app.post("/process-dermoscopy")
async def process_dermoscopy_endpoint(
    input_file: UploadFile = File(...),
    reference_file: UploadFile = None
):
    return await process_and_assess_image(input_file, reference_file)

@app.post("/generate")
async def process_image_endpoint():
    input_image_path = "images/clean_skin2.png"
    output_image_path = "images/noisy_skin2.jpg"

    apply_to_image(input_image_path, output_image_path, num_hairs=100)
