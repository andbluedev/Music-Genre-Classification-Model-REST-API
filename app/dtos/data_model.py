from typing import List
from pydantic import BaseModel


class Prediction(BaseModel):
    filename: str
    predicted: str
    extracted_features: List
