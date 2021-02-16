from fastapi import FastAPI

import joblib
from tensorflow.keras.models import load_model

from app.routes import predict


app = FastAPI(
    title="Model API for Music Genre Classification",
    version="1.0"
)

@app.get('/live')
async def api_health_check():

    return {"status": "OK"}

app.include_router(
    predict.router,
    prefix="/predict",
    tags=["predict"],
)


