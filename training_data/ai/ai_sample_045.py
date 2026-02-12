43.# Find All Anagrams
from collections import Counter

def find_anagrams(word, words):
    target = Counter(word)
    return [w for w in words if Counter(w) == target]