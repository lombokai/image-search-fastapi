from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Uniform(Enum):
    U0 = "T-shirt/top"
    U1 = "Trouser"
    U2 = "Pullover"
    U3 = "Dress"
    U4 = "Coat"
    U5 = "Sandal"
    U6 = "Shirt"
    U7 = "Sneaker"
    U8 = "Bag"
    U9 = "Ankle boot"


class MachineLearningResponse(BaseModel):
    prediction: List[int]


class HealthResponse(BaseModel):
    status: bool
