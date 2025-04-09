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

def process_video_with_tracking(
    video_path: str,
    confidence_thresh: float = 0.5,
    frame_skip: int = 3
) -> Generator[bytes, None, None]:
    """Process video with proper resource handling"""
    cap: Optional[cv2.VideoCapture] = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Process frame with YOLOv8 tracking
            results = model.track(frame, conf=confidence_thresh, persist=True)
            annotated_frame = results[0].plot()

            # Encode and yield frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + 
                buffer.tobytes() + 
                b'\r\n'
            )

    finally:
        # Properly release resources
        if cap is not None:
            cap.release()
            # Important: Add delay to ensure resources are freed
            time.sleep(0.5)  

@app.post("/track/video")
async def track_hazards_in_video(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    frame_skip: int = 3
):
    """Endpoint with complete Windows file handling"""
    temp_path = None
    try:
        # Create temp file with unique name
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"track_{os.getpid()}_{time.time()}.mp4")

        # Write in chunks with explicit file closing
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)

        # Process video
        return StreamingResponse(
            process_video_with_tracking(temp_path, confidence, frame_skip),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

    finally:
        # Robust cleanup
        if temp_path:
            if not safe_file_cleanup(temp_path):
                print(f"Warning: Could not delete {temp_path}")

        # Clean up directory
        if 'temp_dir' in locals():
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass