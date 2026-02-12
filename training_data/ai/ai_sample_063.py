61.# Recursively Search for Files
import os

def find_files(directory, extension):
    matches = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                matches.append(os.path.join(root, file))
    return matches