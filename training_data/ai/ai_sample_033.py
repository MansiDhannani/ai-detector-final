31.# Kruskal's Algorithm (Minimum Spanning Tree)
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

def kruskal(edges, V):
    edges.sort(key=lambda x: x[2])
    dsu = DSU(V)
    mst = []
    for u,v,w in edges:
        if dsu.find(u) != dsu.find(v):
            mst.append((u,v,w))
            dsu.union(u,v)
    return mst

edges = [(0,1,10),(0,2,6),(0,3,5),(1,3,15),(2,3,4)]
V = 4
print(kruskal(edges, V))