59.# Merge Multiple CSV Files
import csv

def merge_csv_files(output_file, *input_files):
    with open(output_file, 'w', newline='') as out:
        writer = None
        for file in input_files:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(out)
                    writer.writerow(header)
                for row in reader:
                    writer.writerow(row)

60.# Context Manager for File Operations
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

61.# Recursively Search for Files
import os

def find_files(directory, extension):
    matches = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                matches.append(os.path.join(root, file))
    return matches

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

63.# Compress and Decompress Files
import gzip
import shutil

def compress_file(src, dst):
    with open(src, 'rb') as f, gzip.open(dst, 'wb') as gz:
        shutil.copyfileobj(f, gz)

def decompress_file(src, dst):
    with gzip.open(src, 'rb') as gz, open(dst, 'wb') as f:
        shutil.copyfileobj(gz, f)

64.# Log File Analyzer
from collections import Counter

def analyze_logs(filename):
    levels = Counter()
    with open(filename) as f:
        for line in f:
            if "ERROR" in line:
                levels["ERROR"] += 1
            elif "WARNING" in line:
                levels["WARNING"] += 1
            elif "INFO" in line:
                levels["INFO"] += 1
    return levels

65.# Safe File Writing (Atomic Operation)
import os
import tempfile

def atomic_write(filename, data):
    dir_name = os.path.dirname(filename)
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tf:
        tf.write(data)
        temp_name = tf.name
    os.replace(temp_name, filename)

66.# Simple REST API Client
import requests

def api_get(url, params=None):
    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()
    return response.json()

67.# Scrape Product Prices from a Website
import requests
from bs4 import BeautifulSoup

def scrape_prices(url, class_name):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [tag.text.strip() for tag in soup.find_all(class_=class_name)]


68.# Rate Limiter for API Calls
import time

class RateLimiter:
    def __init__(self, calls_per_second):
        self.delay = 1 / calls_per_second
        self.last_call = 0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

69.# Download Files with Retry Logic
import requests
import time

def download_with_retry(url, filename, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(r.content)
            return "Download successful"
        except Exception as e:
            if attempt == retries - 1:
                return "Download failed"
            time.sleep(2)