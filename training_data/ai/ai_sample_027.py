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