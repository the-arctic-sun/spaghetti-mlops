from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from ultralytics import YOLO

app = FastAPI(title="Spaghetti Segmentation API")

# Load model once
model = YOLO("models/best.pt")


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)

    detections = []
    segments = [] 

    if results[0].boxes:
        for cls, conf, box in zip(
            results[0].boxes.cls,
            results[0].boxes.conf,
            results[0].boxes.xyxy
        ):
            detections.append({
                "class": int(cls),
                "confidence": float(conf),
                "bbox": box.tolist()
            })

    # Segmentation (VERY IMPORTANT)
    if results[0].masks:
        for seg in results[0].masks.xy:
            segments.append(seg.tolist())

    return {
        "detections": detections,
        "segments": segments
    }
