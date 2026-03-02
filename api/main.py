import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="Drone Detection API")

# Allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally on startup (lazy loading fallback)
try:
    model = YOLO("best.pt")
except Exception as e:
    model = None
    print(f"Warning: Model 'best.pt' not found or failed to load. {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.10):
    if model is None:
        raise HTTPException(status_code=503, detail="Model 'best.pt' not loaded on server.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")

    try:
        # Read image to memory
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference specifying the confidence threshold
        results = model(image, conf=conf)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                confidence = float(box.conf[0].item())
                
                # xyxy coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        return JSONResponse(status_code=200, content={"detections": detections, "image_size": image.size})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
