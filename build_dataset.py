import joblib
import random
from pathlib import Path

def load_code(files, label):
    data = []
    for file in files:
        code = file.read_text(encoding="utf-8", errors="ignore")
        if len(code.strip()) > 30:
            data.append({"code": code, "label": label})
    return data

human_files = list(Path("training_data/human").glob("*.py"))
ai_files = list(Path("training_data/ai").glob("*.py"))

human_data = load_code(human_files, label=0)
ai_data = load_code(ai_files, label=1)

print(f"Filtered Human samples: {len(human_data)}")
print(f"Filtered AI samples: {len(ai_data)}")

# Balance AFTER filtering
min_size = min(len(human_data), len(ai_data))

random.shuffle(human_data)
random.shuffle(ai_data)

human_data = human_data[:min_size]
ai_data = ai_data[:min_size]

data = human_data + ai_data
random.shuffle(data)

print(f"Balanced samples per class: {min_size}")
print(f"Total samples: {len(data)}")

joblib.dump(data, "training_data.pkl")
print("✅ training_data.pkl created (properly balanced)")
