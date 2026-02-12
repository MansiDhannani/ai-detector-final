96.# Configuration Manager
import json

class ConfigManager:
    def __init__(self, file):
        with open(file) as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)