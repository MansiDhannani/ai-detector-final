63.# Compress and Decompress Files
import gzip
import shutil

def compress_file(src, dst):
    with open(src, 'rb') as f, gzip.open(dst, 'wb') as gz:
        shutil.copyfileobj(f, gz)

def decompress_file(src, dst):
    with gzip.open(src, 'rb') as gz, open(dst, 'wb') as f:
        shutil.copyfileobj(gz, f)