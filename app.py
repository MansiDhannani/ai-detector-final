from hybrid_ai_detector import detect
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "AI Detector running"}

@app.post("/predict")
def predict(data: dict):
    return detect(data["text"])
