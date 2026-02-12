28.# Bellman-Ford Algorithm
def bellman_ford(edges, V, start):
    dist = [float('inf')] * V
    dist[start] = 0

    for _ in range(V-1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return "Graph contains negative weight cycle"
    return dist

edges = [(0,1,4), (0,2,5), (1,2,-3), (2,3,4)]
V = 4
print(bellman_ford(edges, V, 0))