from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hybrid_ai_detector import HybridAIDetector
from pathlib import Path
from contextlib import asynccontextmanager
import logging

class CodeRequest(BaseModel):
    code: str

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize detector
detector = None
initialization_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, initialization_error
    try:
        detector = HybridAIDetector()
        base_dir = Path(__file__).resolve().parent
        saved_models_path = base_dir / "saved_models"
        
        success = False
        if saved_models_path.exists():
            try:
                logger.info(f"📂 [Step 1/2] Attempting to load from folder: {saved_models_path}")
                detector.load_pretrained(path=str(saved_models_path))
                success = True
            except Exception as e:
                logger.warning(f"⚠️ Folder load skipped/failed: {e}. Moving to Step 2...")

        if not success:
            logger.info(f"🏠 [Step 2/2] Attempting to load from root: {base_dir}")
            detector.load_pretrained(path=str(base_dir))

        if detector and getattr(detector.feature_detector, 'is_trained', False):
            logger.info("✅ Hybrid AI Detector loaded successfully")
        else:
            raise RuntimeError("Detector initialized but 'is_trained' flag is False.")
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"❌ Initialization failed: {initialization_error}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    if detector and getattr(detector.feature_detector, 'is_trained', False):
        return {"status": "AI Code Detector API running"}
    
    return {
        "status": "AI Code Detector API loading/error",
        "detail": initialization_error or "Detector is still initializing or model files were not found."
    }

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
