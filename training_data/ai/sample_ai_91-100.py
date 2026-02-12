91.# Logger with Different Severity Levels
from datetime import datetime

class Logger:
    LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

    def log(self, level, message):
        if level in self.LEVELS:
            print(f"{datetime.now()} [{level}] {message}")

    def debug(self, msg): self.log("DEBUG", msg)
    def info(self, msg): self.log("INFO", msg)
    def warning(self, msg): self.log("WARNING", msg)
    def error(self, msg): self.log("ERROR", msg)

92.# Validate Credit Card Numbers (Luhn Algorithm)
def validate_credit_card(number):
    digits = [int(d) for d in str(number)]
    checksum = 0

    for i in range(len(digits)-2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9

    checksum = sum(digits)
    return checksum % 10 == 0

93.# Date/Time Parser for Multiple Formats
from datetime import datetime

def parse_datetime(date_str):
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None

94.# Simple Dependency Injection Container
class Container:
    def __init__(self):
        self.services = {}

    def register(self, name, service):
        self.services[name] = service

    def resolve(self, name):
        return self.services.get(name)

95.# Calculate File Checksums
import hashlib

def file_checksum(filename, algo="sha256"):
    h = hashlib.new(algo)
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

96.# Configuration Manager
import json

class ConfigManager:
    def __init__(self, file):
        with open(file) as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

97.# Event Emitter / Listener Pattern
class EventEmitter:
    def __init__(self):
        self.listeners = {}

    def on(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)

    def emit(self, event, data=None):
        for callback in self.listeners.get(event, []):
            callback(data)

98.# Secure Random Password Generator
import secrets
import string

def generate_password(length=12):
    chars = string.ascii_letters + string.digits + "!@#$%"
    return ''.join(secrets.choice(chars) for _ in range(length))

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

100.# Simple State Machine
class StateMachine:
    def __init__(self, state):
        self.state = state

    def transition(self, new_state):
        print(f"{self.state} -> {new_state}")
        self.state = new_state