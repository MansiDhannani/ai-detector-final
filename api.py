from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hybrid_ai_detector import HybridAIDetector
from pathlib import Path
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
# Lazy Model Loader
# ---------------------------

def get_detector():
    global detector, initialization_error

    if detector is None:
        try:
            logger.info("🚀 Initializing Hybrid AI Detector (lazy load)...")

            detector = HybridAIDetector()
            base_dir = Path(__file__).resolve().parent
            saved_models_path = base_dir / "saved_models"

            # Try loading from saved_models folder
            if saved_models_path.exists():
                logger.info(f"📂 Loading model from folder: {saved_models_path}")
                detector.load_pretrained(path=str(saved_models_path))
            else:
                logger.info(f"🏠 Loading model from root: {base_dir}")
                detector.load_pretrained(path=str(base_dir))

            if detector and getattr(detector.feature_detector, 'is_trained', False):
                logger.info("✅ Hybrid AI Detector loaded successfully")
            else:
                raise RuntimeError("Detector initialized but 'is_trained' flag is False.")

        except Exception as e:
            initialization_error = str(e)
            logger.error(f"❌ Initialization failed: {initialization_error}")
            raise

    return detector


# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI()

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
    return {"status": "AI Code Detector API running"}


@app.post("/detect")
def detect(payload: CodeRequest):
    try:
        model = get_detector()  # Lazy load happens here
        result = model.detect(payload.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
