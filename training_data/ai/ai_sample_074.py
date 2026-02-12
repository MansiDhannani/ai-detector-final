72.# Parse RSS Feeds
import feedparser

def parse_rss(url):
    feed = feedparser.parse(url)
    return [(entry.title, entry.link) for entry in feed.entries]