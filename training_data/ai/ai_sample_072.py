70.# Web Scraper using BeautifulSoup
import requests
from bs4 import BeautifulSoup

def scrape_titles(url):
    response = requests.get(url, timeout=5)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [h.text.strip() for h in soup.find_all('h1')]