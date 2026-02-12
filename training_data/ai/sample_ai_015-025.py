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



16.# Segment Tree for Range Sum Queries
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)
        self.build(arr)

    def build(self, arr):
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[i << 1] + self.tree[i << 1 | 1]

    def update(self, index, value):
        index += self.n
        self.tree[index] = value
        while index > 1:
            index >>= 1
            self.tree[index] = self.tree[index << 1] + self.tree[index << 1 | 1]

    def query(self, left, right):
        result = 0
        left += self.n
        right += self.n
        while left < right:
            if left & 1:
                result += self.tree[left]
                left += 1
            if right & 1:
                right -= 1
                result += self.tree[right]
            left >>= 1
            right >>= 1
        return result



17.# Skip List
import random

class Node:
    def __init__(self, val, level):
        self.val = val
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = Node(-1, max_level)
        self.level = 0

    def random_level(self):
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, val):
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < val:
                current = current.forward[i]
            update[i] = current
        lvl = self.random_level()
        if lvl > self.level:
            for i in range(self.level + 1, lvl + 1):
                update[i] = self.header
            self.level = lvl
        node = Node(val, lvl)
        for i in range(lvl + 1):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node

    def search(self, val):
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < val:
                current = current.forward[i]
        current = current.forward[0]
        return current is not None and current.val == val

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

19.# Red-Black Tree
# Full implementation is long (~200 lines), but basic node:
class RBNode:
    def __init__(self, key, color='red'):
        self.key = key
        self.color = color
        self.left = self.right = self.parent = None




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


21.# Quicksort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))

22.# Merge Sort
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
arr = [12,11,13,5,6,7]
merge_sort(arr)
print(arr)

23.# Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr)-1
    while left <= right:
        mid = left + (right-left)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

24.# Breadth-First Search (Graph)
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        print(node, end=" ")
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

graph = {0:[1,2], 1:[2], 2:[0,3], 3:[3]}
bfs(graph, 2)

25.# Depth-First Search (Tree)
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None

def dfs(root):
    if root:
        print(root.val, end=" ")
        dfs(root.left)
        dfs(root.right)

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
dfs(root)