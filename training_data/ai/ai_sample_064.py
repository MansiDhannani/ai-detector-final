62.# Copy Files with Progress Tracking
import shutil
import os

def copy_with_progress(src, dst, chunk_size=1024):
    total = os.path.getsize(src)
    copied = 0

    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        while chunk := fsrc.read(chunk_size):
            fdst.write(chunk)
            copied += len(chunk)
            print(f"Progress: {copied * 100 // total}%")