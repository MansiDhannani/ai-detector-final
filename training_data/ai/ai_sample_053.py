51.# Parse a CSV String with Quoted Fields
import csv
from io import StringIO

def parse_csv(csv_text):
    reader = csv.reader(StringIO(csv_text))
    return list(reader)