from fastapi import FastAPI, UploadFile, File
from add_hair import apply_to_image
from lesion_segmentation import segment_lesion
from image_quality_assessor import process_and_assess_image
from hair_removal import remove_hair
from calculate_asymmetry import calculate_asymmetry
from border_irregularity import calculate_border_irregularity
from pigment_transition import calculate_pigment_transition
from calculate_color_variation import calculate_color_variation
from calculate_glcm_features import calculate_glcm_features
from lession_classifier import LesionClassifier
from fastapi import UploadFile, File, HTTPException
from typing import Dict
import numpy as np
import cv2

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


@app.post("/calculate-asymmetry")
async def calculate_asymmetry_endpoint(
        file: UploadFile = File(...)
) -> dict:
    """
    Calculate asymmetry indices from an uploaded image file

    Args:
        file: Uploaded image file

    Returns:
        Dictionary containing asymmetry indices and image paths
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding to get binary mask
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = binary_mask / 255  # Normalize to 0-1

        # Calculate asymmetry indices
        A1, A2 = calculate_asymmetry(binary_mask)

        # Create visualization
        vis_mask = binary_mask.copy()
        moments = cv2.moments(binary_mask.astype(np.uint8))

        if moments["m00"] == 0:
            raise HTTPException(status_code=400, detail="No lesion detected in image")

        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        # Draw axes through centroid
        vis_mask = cv2.cvtColor((vis_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.line(vis_mask, (centroid_x, 0), (centroid_x, vis_mask.shape[0]), (0, 255, 0), 2)
        cv2.line(vis_mask, (0, centroid_y), (vis_mask.shape[1], centroid_y), (0, 255, 0), 2)

        # Save visualization image temporarily
        temp_path = "images/temp_visualization.png"
        cv2.imwrite(temp_path, vis_mask)

        # Return results with image path
        return {
            "asymmetry_index_1": float(A1),
            "asymmetry_index_2": float(A2),
            "visualization_path": temp_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating asymmetry: {str(e)}")


@app.post("/border-irregularity")
async def calculate_border_irregularity_endpoint(
        file: UploadFile = File(...)
) -> Dict:
    """
    Calculate border irregularity from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        Dictionary containing border irregularity measurements and visualization path
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate border irregularity
        results = calculate_border_irregularity(binary_mask)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/calculate-pigment-transition")
async def calculate_pigment_transition_endpoint(
        file: UploadFile = File(...)
) -> Dict:
    """
    Calculate pigment transition from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        Dictionary containing pigment transition measurements and visualization path
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding for segmentation
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate pigment transition
        results = calculate_pigment_transition(img, mask > 0)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/calculate-color-variation")
async def calculate_color_variation_endpoint(
        file: UploadFile = File(...)
) -> Dict:
    """
    Calculate color variation from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        Dictionary containing color variation measurements and visualization paths
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get binary mask using Otsu's thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate color variation
        results = calculate_color_variation(img, mask > 0)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/calculate-texture-features")
async def calculate_texture_features_endpoint(
        file: UploadFile = File(...)
) -> Dict:
    """
    Calculate texture features from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        Dictionary containing texture features and visualization path
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get binary mask using Otsu's thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate texture features
        results = calculate_glcm_features(gray, mask > 0)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/classify-lesion")
async def classify_lesion(results: Dict):
    """Lezyon sınıflandırma endpoint'i"""
    try:
        classifier = LesionClassifier()

        # Eğitim verisini yükle (gerçek uygulamada bu önceden eğitilmiş olmalı)
        # Bu örnek için dummy veri kullanıyoruz
        X_train = np.random.rand(100, 17)  # 17 özellik
        y_train = np.random.randint(0, 2, 100)  # 0: benign, 1: malignant

        # Modeli eğit
        classifier.train(X_train, y_train)

        # Özellikleri çıkar
        features = classifier.extract_features(results)

        # Sınıflandırma yap
        classification_result = classifier.predict(features)

        return classification_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/generate")
async def process_image_endpoint():
    input_image_path = "images/clean_skin2.png"
    output_image_path = "images/noisy_skin2.jpg"

    apply_to_image(input_image_path, output_image_path, num_hairs=100)
