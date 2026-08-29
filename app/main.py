from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'http://localhost:5174',
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load YOLO model
model = YOLO("model/best.pt")

@app.get("/")
def home():
    return {"message": "Fish Freshness Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Save uploaded file
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run YOLO prediction
    results = model.predict(source=file_path, conf=0.80)
    
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            detections.append({
                "class": label,
                "confidence": round(conf, 3)
            })

    # Delete temp file
    os.remove(file_path)

    return {
        "total_detections": len(detections),
        "results": detections
    }