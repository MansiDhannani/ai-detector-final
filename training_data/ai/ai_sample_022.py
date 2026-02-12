20.# Suffix Tree
class SuffixTreeNode:
    def __init__(self):
        self.children = {}

class SuffixTree:
    def __init__(self):
        self.root = SuffixTreeNode()

    def insert(self, s):
        for i in range(len(s)):
            current = self.root
            for char in s[i:]:
                if char not in current.children:
                    current.children[char] = SuffixTreeNode()
                current = current.children[char]