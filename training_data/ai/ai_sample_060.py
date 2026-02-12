58.# File Watcher (Directory Change Monitor)
import os
import time

def watch_directory(path, interval=2):
    previous = set(os.listdir(path))
    while True:
        time.sleep(interval)
        current = set(os.listdir(path))

        added = current - previous
        removed = previous - current

        if added:
            print("Added:", added)
        if removed:
            print("Removed:", removed)

        previous = current