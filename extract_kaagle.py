# extract_kaggle.py
from pathlib import Path
import random
import shutil

# Find all Python files
py_files = list(Path('raw_kaggle').rglob('*.py'))
print(f"Found {len(py_files)} Python files")

# Randomly select 100
selected = random.sample(py_files, min(100, len(py_files)))

# Copy to training folder
for i, file in enumerate(selected):
    shutil.copy(file, f'training_data/human/human_{i:03d}.py')

print(f"Copied {len(selected)} files to training_data/human/")
