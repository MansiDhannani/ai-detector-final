53.# Find the Most Frequent Word
from collections import Counter
import re

def most_frequent_word(text):
    words = re.findall(r'\w+', text.lower())
    return Counter(words).most_common(1)[0]