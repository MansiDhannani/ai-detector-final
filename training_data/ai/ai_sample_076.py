74.# Validate URLs
import re

def is_valid_url(url):
    pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))