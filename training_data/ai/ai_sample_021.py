19.# Red-Black Tree
# Full implementation is long (~200 lines), but basic node:
class RBNode:
    def __init__(self, key, color='red'):
        self.key = key
        self.color = color
        self.left = self.right = self.parent = None