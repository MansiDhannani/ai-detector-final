87.# Retry Decorator with Exponential Backoff
import time

def retry(retries=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            d = delay
            for _ in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    time.sleep(d)
                    d *= 2
        return wrapper
    return decorator