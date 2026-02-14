from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hybrid_ai_detector import HybridAIDetector
import os

app = FastAPI()

class CodeRequest(BaseModel):
    code: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    try:
        detector = HybridAIDetector()
        # Try multiple paths to find the saved_models folder
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models"),
            os.path.join(os.getcwd(), "saved_models")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                detector.load_pretrained(path=path)
                break
        print("✅ Hybrid AI Detector loaded successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        # We don't raise here so the API can still start and report its status

@app.get("/")
def root():
    status = "running" if detector and detector.feature_detector.is_trained else "loading/error"
    return {"status": f"AI Code Detector API {status}"}

@app.post("/detect")
def detect(payload: CodeRequest):
    if detector is None or not getattr(detector.feature_detector, 'is_trained', False):
        raise HTTPException(status_code=503, detail="Detector not initialized. Check logs for Git LFS or memory issues.")
    
    try:
        # Use the code from the validated Pydantic model
        result = detector.detect(payload.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
