67.# Scrape Product Prices from a Website
import requests
from bs4 import BeautifulSoup

def scrape_prices(url, class_name):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [tag.text.strip() for tag in soup.find_all(class_=class_name)]