57.# Parse JSON File with Error Handling
import json

def parse_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return "File not found"
    except json.JSONDecodeError:
        return "Invalid JSON format"