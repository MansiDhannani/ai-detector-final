35.# Topological Sorting (DFS)
def topological_sort(graph):
    visited = set()
    stack = []

    def dfs(v):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)

    for node in graph:
        if node not in visited:
            dfs(node)
    return stack[::-1]

graph = {5:[2,0],4:[0,1],2:[3],3:[1],0:[],1:[]}
print(topological_sort(graph))