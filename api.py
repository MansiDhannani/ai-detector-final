from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hybrid_ai_detector import HybridAIDetector
import logging

# ---------------------------
# Request Model
# ---------------------------

class CodeRequest(BaseModel):
    code: str


# ---------------------------
# Logging Setup
# ---------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Global Variables
# ---------------------------

detector = None
initialization_error = None

# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI()

@app.on_event("startup")
def load_detector():
    global detector, initialization_error
    try:
        logger.info("🚀 Initializing Hybrid AI Detector (startup load)...")
        detector = HybridAIDetector()
        detector.load_pretrained()  # Load model ONCE
        logger.info("✅ Hybrid AI Detector loaded successfully")
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"❌ Detector initialization failed: {initialization_error}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def root():
    if detector:
        return {"status": "AI Code Detector API running"}
    return {"status": "AI Code Detector API error", "detail": initialization_error}


@app.post("/detect")
def detect(payload: CodeRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized. Check logs.")
    try:
        result = detector.detect(payload.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
