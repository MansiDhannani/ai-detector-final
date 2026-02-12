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