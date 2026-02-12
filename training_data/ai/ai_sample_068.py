66.# Simple REST API Client
import requests

def api_get(url, params=None):
    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()
    return response.json()