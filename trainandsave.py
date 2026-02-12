from hybrid_ai_detector import HybridAIDetector
from pathlib import Path
import os

def load_codes(folder):
    return [
        Path(folder, f).read_text(encoding="utf-8", errors="ignore")
        for f in os.listdir(folder)
        if f.endswith(".py")
    ]

if __name__ == "__main__":
    human_codes = load_codes("training_data/human")
    ai_codes = load_codes("training_data/ai")

    detector = HybridAIDetector(use_gpu=False)
    detector.train(human_codes, ai_codes, epochs=1)
    detector.save_model()
