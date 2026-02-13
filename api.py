from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hybrid_ai_detector import HybridAIDetector
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = HybridAIDetector()
# Use absolute path resolution for the Railway environment
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
detector.load_pretrained(path=base_path)

@app.get("/")
def root():
    return {"status": "AI Code Detector API running"}

@app.post("/detect")
def detect(payload: dict):
    code = payload["code"]
    result = detector.detect(code)
    return result
