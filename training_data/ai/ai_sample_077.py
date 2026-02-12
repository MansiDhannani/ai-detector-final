75.# Simple Web Crawler
import requests
from bs4 import BeautifulSoup

def crawl(url, visited=set()):
    if url in visited:
        return
    visited.add(url)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for link in soup.find_all('a', href=True):
        print(link['href'])