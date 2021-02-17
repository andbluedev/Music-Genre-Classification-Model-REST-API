
from fastapi import FastAPI, File, UploadFile, APIRouter, HTTPException
import logging

import joblib 
from tensorflow.keras.models import load_model
import librosa

from app.dtos.data_model import Prediction
from app.services.feature_extraction import extract_audio_file_features

import os
import tempfile

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


model_path = "./models/musicgenre_nn_classifier_CSV-V2.h5"
scaler_path = "./models/musicgenre_standard_scaler_CSV-V2.bin"
encoder_path = "./models/musicgenre_encoder_CSV-V2.bin"

model = load_model(model_path)
std_scaler = joblib.load(scaler_path)
classes = joblib.load(encoder_path).classes_


@router.post("", response_model=Prediction)
async def upload_sound_and_predict(audio_file: UploadFile = File(...)):
    """The uploaded file should be a valid mp3 file of a song."""
    extension = os.path.splitext(audio_file.filename)[1]
    logger.info(f"Processing: {audio_file.filename}")
    _, path = tempfile.mkstemp(prefix='parser_', suffix=extension)
    
    logger.info(f"Generated '{path}' temporary file ")

    if extension != ".mp3":
        raise HTTPException(status_code=400, detail="Uplaoded file should have an mp3 extension")
    
    try:
        with open(path, 'ab') as f:
            for chunk in iter(lambda: audio_file.file.read(10000), b''):
                f.write(chunk)

        std_audio_data = extract_audio_file_features(path, std_scaler)
        
        y_prob = model.predict(std_audio_data)
        y_pred = classes[y_prob.argmax(axis=-1)[0]]

        return {"filename": audio_file.filename, "predicted": y_pred, "extracted_features": list(std_audio_data[0])}

    except Exception as e:
        logger.error(f"Error while processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error while processing audio file.")
    finally: 
        # remove temp file
        os.close(_)
        os.remove(path)




