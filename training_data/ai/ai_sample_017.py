15.# Sparse Matrix Class
class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = {}  # store only non-zero elements as {(row, col): value}

    def set(self, row, col, value):
        if value != 0:
            self.data[(row, col)] = value
        elif (row, col) in self.data:
            del self.data[(row, col)]

    def get(self, row, col):
        return self.data.get((row, col), 0)