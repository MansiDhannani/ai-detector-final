from pathlib import Path
import re

INPUT_DIR = Path(".")
OUTPUT_DIR = Path(".")
counter = 1

# Match ONLY numbers 1–100 at start of line
# No dot, no hashtag
SPLIT_PATTERN = re.compile(
    r'(?=^(?:[1-9]|[1-9][0-9]|100)\b)',
    re.MULTILINE
)

files = sorted(INPUT_DIR.glob("sample_ai_*.py"))

print(f"Found {len(files)} combined files")

for file in files:
    print(f"\nProcessing {file.name}")
    text = file.read_text(encoding="utf-8", errors="ignore")

    chunks = [c.strip() for c in SPLIT_PATTERN.split(text) if c.strip()]
    print(f"  Found {len(chunks)} samples")

    for chunk in chunks:
        out_file = OUTPUT_DIR / f"ai_sample_{counter:03d}.py"
        out_file.write_text(chunk, encoding="utf-8")
        counter += 1

print(f"\n✅ DONE: Created {counter - 1} files")

