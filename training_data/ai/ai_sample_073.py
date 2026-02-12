71.# Simple HTTP Server
from http.server import SimpleHTTPRequestHandler, HTTPServer

def run_server(port=8000):
    server = HTTPServer(('', port), SimpleHTTPRequestHandler)
    print(f"Server running on port {port}")
    server.serve_forever()