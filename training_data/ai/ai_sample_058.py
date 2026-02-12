56.# Read a Large File Line by Line Efficiently
def read_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()