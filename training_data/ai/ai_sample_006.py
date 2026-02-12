4.# Hash Table with Collision Handling (Chaining)
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_func(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_func(key)
        # Update if key exists
        for item in self.table[index]:
            if item[0] == key:
                item[1] = value
                return
        self.table[index].append([key, value])

    def get(self, key):
        index = self.hash_func(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def delete(self, key):
        index = self.hash_func(key)
        for i, (k, _) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                return