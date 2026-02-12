from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hybrid_ai_detector import HybridAIDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = HybridAIDetector()
detector.load_pretrained()

@app.get("/")
def root():
    return {"status": "AI Code Detector API running"}

@app.post("/detect")
def detect(payload: dict):
    code = payload["code"]
    result = detector.detect(code)
    return result
