#!/usr/bin/env python3
"""
Quick Kaggle Dataset Downloader for AI Code Detector
Downloads and prepares datasets for training
"""

import os
import json
from pathlib import Path

def setup_kaggle():
    """Setup Kaggle API"""
    print("🔧 KAGGLE SETUP")
    print("=" * 60)
    print()
    print("Step 1: Get Kaggle API credentials")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Scroll to 'API' section")
    print("  3. Click 'Create New Token'")
    print("  4. Download kaggle.json")
    print()
    print("Step 2: Place credentials")
    print("  Linux/Mac: ~/.kaggle/kaggle.json")
    print("  Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
    print()
    print("Step 3: Install Kaggle CLI")
    print("  pip install kaggle")
    print()
    
def download_recommended_datasets():
    """Download pre-selected datasets"""
    
    datasets = {
        "github_python": {
            "name": "GitHub Python Code Dataset",
            "command": "kaggle datasets download -d CooperUnion/github-python-code",
            "size": "~500MB",
            "samples": "~10000 files",
            "type": "human"
        },
        "python_functions": {
            "name": "Python Functions Dataset", 
            "command": "kaggle datasets download -d aravindkolla/python-functions",
            "size": "~50MB",
            "samples": "~5000 files",
            "type": "human"
        }
    }
    
    print("\n📊 RECOMMENDED DATASETS")
    print("=" * 60)
    for key, info in datasets.items():
        print(f"\n{info['name']}")
        print(f"  Type: {info['type']}")
        print(f"  Size: {info['size']}")
        print(f"  Samples: {info['samples']}")
        print(f"  Command: {info['command']}")
    
    print("\n" + "=" * 60)
    print("\n💡 QUICK START:")
    print("  1. Pick a dataset above")
    print("  2. Copy the command")
    print("  3. Run in terminal")
    print("  4. Unzip the file")
    print()

def extract_and_organize():
    """Guide for organizing downloaded data"""
    
    print("\n📁 ORGANIZING DATA")
    print("=" * 60)
    print("""
After downloading:

1. Unzip the dataset:
   unzip github-python-code.zip -d raw_data/

2. Create folder structure:
   mkdir -p training_data/human
   mkdir -p training_data/ai

3. Copy Python files:
   find raw_data/ -name "*.py" -exec cp {} training_data/human/ \;

4. Randomly select 200 files:
   python -c "
   import os, random, shutil
   from pathlib import Path
   
   src = Path('training_data/human')
   files = list(src.glob('*.py'))
   selected = random.sample(files, min(200, len(files)))
   
   for i, f in enumerate(selected):
       shutil.copy(f, src / f'human_{i:03d}.py')
   "

Done! You now have human code samples.
    """)

def search_kaggle_datasets():
    """Search Kaggle for code datasets"""
    
    print("\n🔍 MANUAL SEARCH ON KAGGLE")
    print("=" * 60)
    print("""
Search terms that work well:

1. "python code"
   - Lots of Python files
   - Usually from GitHub

2. "code snippets"
   - Smaller code samples
   - Good for training

3. "programming dataset"
   - Various languages
   - Can filter for Python

4. "github dataset python"
   - Actual GitHub repos
   - High quality

WHAT TO LOOK FOR:
✅ At least 1000+ code files
✅ Python (.py files)
✅ From GitHub (pre-2022 = pre-ChatGPT)
✅ Good documentation
✅ CC0 or MIT license

AVOID:
❌ Datasets after 2023 (might include AI code)
❌ Very small datasets (< 100 files)
❌ Non-Python code
❌ Datasets without clear source
    """)

if __name__ == "__main__":
    print("=" * 60)
    print("KAGGLE DATA COLLECTION FOR AI DETECTOR")
    print("=" * 60)
    
    setup_kaggle()
    download_recommended_datasets()
    extract_and_organize()
    search_kaggle_datasets()
    
    print("\n" + "=" * 60)
    print("✅ NEXT STEPS:")
    print("  1. Setup Kaggle API credentials")
    print("  2. Download a dataset using commands above")
    print("  3. Extract and organize into training_data/human/")
    print("  4. Generate AI code (see ai_code_generator.py)")
    print("=" * 60)
