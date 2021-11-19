from typing import Any

import pickle
import numpy as np
from PIL import Image
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from models.prediction import HealthResponse, MachineLearningResponse
from services.predict import MachineLearningModelHandlerScore as model


router = APIRouter()

get_prediction = lambda data_input: model.predict(data_input, load_wrapper=pickle.load)


@router.get("/predict", name="predict:get-data")
async def predict(file_path: str):
    if not file_path:
        raise HTTPException(status_code=404, detail=f"'data_input' argument invalid!")
    try:
        image_binary = Image.open(file_path)
        image_resized = image_binary.resize((28, 28)).convert('L')
        prediction = get_prediction(np.array(image_resized))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return {"prediction": prediction}


@router.get(
    "/health", response_model=HealthResponse, name="health:get-data",
)
async def health():
    is_health = False
    try:
        get_prediction("lorem ipsum")
        is_health = True
        return HealthResponse(status=is_health)
    except Exception:
        raise HTTPException(status_code=404, detail="Unhealthy")
