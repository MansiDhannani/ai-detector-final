18.# B-Tree
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t  # minimum degree
        self.keys = []
        self.children = []
        self.leaf = leaf

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, True)
        self.t = t

# Full implementation is long; the above is the starting point.