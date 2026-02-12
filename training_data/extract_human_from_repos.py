from pathlib import Path
import shutil

SOURCE_DIRS = [
    Path(r"C:\Users\mansimaheshwari\Downloads\flask-main"),
    Path(r"C:\Users\mansimaheshwari\Downloads\django-main"),
    Path(r"C:\Users\mansimaheshwari\Downloads\requests-main"),
]

TARGET = Path("human")
TARGET.mkdir(parents=True, exist_ok=True)

count = 0
MAX_FILES = 120

for src in SOURCE_DIRS:
    for py in src.rglob("*.py"):
        try:
            lines = py.read_text(encoding="utf-8", errors="ignore").splitlines()

            # filter junk
            if len(lines) < 15:
                continue

            dest = TARGET / f"human_{count:03d}.py"
            shutil.copy(py, dest)
            count += 1

            if count >= MAX_FILES:
                break
        except:
            pass

    if count >= MAX_FILES:
        break

print(f"✅ Extracted {count} human code files")
