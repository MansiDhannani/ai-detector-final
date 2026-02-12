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