33.# Detect Cycles in a Graph (DFS)
def has_cycle(graph):
    visited = set()
    rec_stack = set()

    def dfs(v):
        visited.add(v)
        rec_stack.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(v)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

graph = {0:[1],1:[2],2:[0,3],3:[]}
print(has_cycle(graph))