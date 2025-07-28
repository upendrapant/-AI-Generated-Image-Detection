from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import os
import torch
from io import BytesIO

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import load_model, analyze_image_bytes
from config import Config

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth')

app = FastAPI()

@app.on_event("startup")
def load_vit_model():
    global model, config, device
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, config)
    model.to(device)

@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    print(f"Received {len(files)} image(s) for analysis.")
    results = []
    for file in files:
        contents = await file.read()
        print(f"Processing image: {file.filename}, size: {len(contents)} bytes")
        result = analyze_image_bytes(model, BytesIO(contents), config, device)
        result["filename"] = file.filename
        print(f"Result for {file.filename}: {result}")
        results.append(result)
    print(f"All results: {results}")
    return JSONResponse(content=results) 