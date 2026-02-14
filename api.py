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
initialization_error = None

@app.on_event("startup")
async def startup_event():
    global detector, initialization_error
    try:
        detector = HybridAIDetector()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            base_dir,
            os.path.join(base_dir, "saved_models"),
            os.getcwd(),
            os.path.join(os.getcwd(), "saved_models")
        ]
        
        success = False
        attempts = []
        for path in possible_paths:
            if not os.path.exists(path):
                continue
            
            files_in_path = os.listdir(path)
            has_ensemble = "hybrid_ai_detector_ensemble.pkl" in files_in_path
            has_standard = "hybrid_ai_detector.pkl" in files_in_path
            
            try:
                if has_ensemble or has_standard:
                    detector.load_pretrained(path=path)
                    success = True
                    attempts.append(f"✅ Success at {path}")
                    break
                else:
                    attempts.append(f"❌ No model files in {path} (Found: {files_in_path[:5]}...)")
            except RuntimeError as e:
                if "Git LFS pointer" in str(e):
                    attempts.append(f"⚠️ LFS Pointer at {path}")
                    continue
                raise e
        
        if success and detector and getattr(detector.feature_detector, 'is_trained', False):
            print("✅ Hybrid AI Detector loaded successfully")
        else:
            raise FileNotFoundError(f"Initialization failed. Search history: {'; '.join(attempts)}")
    except Exception as e:
        initialization_error = str(e)
        print(f"❌ Initialization failed: {initialization_error}")
        # We don't raise here so the API can still start and report its status

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
