99.# Rate-Limited Function Decorator
import time

def rate_limit(calls_per_second):
    interval = 1 / calls_per_second
    last_called = [0]

    def decorator(func):
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator