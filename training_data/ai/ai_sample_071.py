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