95.# Calculate File Checksums
import hashlib

def file_checksum(filename, algo="sha256"):
    h = hashlib.new(algo)
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()