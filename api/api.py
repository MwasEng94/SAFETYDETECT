from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from typing import List, Generator, Optional
from pydantic import BaseModel
import os
import io
from fastapi.responses import StreamingResponse
import time


app = FastAPI(title="Hazard Detection API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("last.pt")  # Or your custom trained model
class_names = model.names  # Get class names


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = model(img)
        
        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
        
        return {"detections": detections}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def safe_file_cleanup(filepath: str, max_retries: int = 5, delay: float = 0.5) -> bool:
    """Robust file deletion with retries and resource releasing"""
    for attempt in range(max_retries):
        try:
            # Try to delete the file
            os.remove(filepath)
            return True
        except PermissionError as e:
            if attempt == max_retries - 1:
                print(f"Failed to delete {filepath} after {max_retries} attempts")
                return False
            time.sleep(delay * (attempt + 1))  # Exponential backoff
    return False

