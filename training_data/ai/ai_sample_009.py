7.# Circular Buffer
class CircularBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.start = 0
        self.end = 0
        self.size = 0

    def append(self, value):
        self.buffer[self.end] = value
        self.end = (self.end + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.start = (self.start + 1) % self.capacity

    def get(self):
        result = []
        idx = self.start
        for _ in range(self.size):
            result.append(self.buffer[idx])
            idx = (idx + 1) % self.capacity
        return result