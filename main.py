from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List

app = FastAPI()

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        print(f"Image received: {file.filename}, size: {len(contents)} bytes")
        # Manually set test values for each image
        result = {
            "filename": file.filename,
            "is_real": True,  # or False for testing
            "score": 0.95,    # set your test score here
            "accuracy": 98.7, # set your test accuracy here
            "description": "This is a test description you can change."
        }
        results.append(result)
    return JSONResponse(content=results)