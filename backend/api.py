# Create api.py
import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import ParkinsonsPredictor

app = FastAPI(title="Parkinson's Disease Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ensemble_model.pth")
EXTRACTOR_PATH = os.path.join(BASE_DIR, "models", "vit_feature_extractor")

predictor = ParkinsonsPredictor(
    model_path=MODEL_PATH, feature_extractor_path=EXTRACTOR_PATH
)


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
