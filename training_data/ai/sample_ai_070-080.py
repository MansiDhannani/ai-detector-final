70.# Web Scraper using BeautifulSoup
import requests
from bs4 import BeautifulSoup

def scrape_titles(url):
    response = requests.get(url, timeout=5)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [h.text.strip() for h in soup.find_all('h1')]

71.# Simple HTTP Server
from http.server import SimpleHTTPRequestHandler, HTTPServer

def run_server(port=8000):
    server = HTTPServer(('', port), SimpleHTTPRequestHandler)
    print(f"Server running on port {port}")
    server.serve_forever()

72.# Parse RSS Feeds
import feedparser

def parse_rss(url):
    feed = feedparser.parse(url)
    return [(entry.title, entry.link) for entry in feed.entries]

73.# Async API Client using aiohttp
import aiohttp
import asyncio

async def fetch_json(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()



74.# Validate URLs
import re

def is_valid_url(url):
    pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

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

76.# SQLite Database Wrapper Class
import sqlite3

class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def execute(self, query, params=()):
        self.cursor.execute(query, params)
        self.conn.commit()

    def fetchall(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

77.# Bulk Inserts Efficiently
def bulk_insert(cursor, query, data):
    cursor.executemany(query, data)

78.# Simple ORM Mapper
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def save(self, db):
        db.execute(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            (self.name, self.age)
        )

79.# Database Connection Pool Manager
import queue
import sqlite3

class ConnectionPool:
    def __init__(self, db, size=5):
        self.pool = queue.Queue(size)
        for _ in range(size):
            self.pool.put(sqlite3.connect(db))

    def acquire(self):
        return self.pool.get()

    def release(self, conn):
        self.pool.put(conn)

80.# Database Schema Migration
def migrate_schema(db, migration_sql):
    db.execute(migration_sql)