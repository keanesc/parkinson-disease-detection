# Create api.py
import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile

from inference import ParkinsonsPredictor

app = FastAPI(title="Parkinson's Disease Detection API")
predictor = ParkinsonsPredictor(model_path="./data/models/ensemble_model_full.pt", feature_extractor_path="./data/models/vit_feature_extractor")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_filename = temp_file.name

    try:
        contents = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(contents)

        # Make prediction
        result = predictor.predict(temp_filename)
        return result
    finally:
        os.remove(temp_filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
