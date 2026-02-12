48.# Extract All Email Addresses from Text
import re

def extract_emails(text):
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

49.# Generate All Possible Substrings
def all_substrings(s):
    substrings = []
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            substrings.append(s[i:j])
    return substrings

50.# Edit Distance (Levenshtein Distance)
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # delete
                    dp[i][j-1],      # insert
                    dp[i-1][j-1]     # replace
                )
    return dp[m][n]

51.# Parse a CSV String with Quoted Fields
import csv
from io import StringIO

def parse_csv(csv_text):
    reader = csv.reader(StringIO(csv_text))
    return list(reader)

52.# Validate Email Address Using Regex
import re

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

53.# Find the Most Frequent Word
from collections import Counter
import re

def most_frequent_word(text):
    words = re.findall(r'\w+', text.lower())
    return Counter(words).most_common(1)[0]

54.# Check if Strings Are Rotations
def are_rotations(s1, s2):
    if len(s1) != len(s2):
        return False
    return s2 in s1 + s1

55.# Generate Acronym from a Phrase
def generate_acronym(phrase):
    return ''.join(word[0].upper() for word in phrase.split())

56.# Read a Large File Line by Line Efficiently
def read_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()

57.# Parse JSON File with Error Handling
import json

def parse_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return "File not found"
    except json.JSONDecodeError:
        return "Invalid JSON format"

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