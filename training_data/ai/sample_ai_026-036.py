26.# Dijkstra's Algorithm (Shortest Path)
import heapq

def dijkstra(graph, start):
    pq = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while pq:
        current_dist, node = heapq.heappop(pq)
        if current_dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}

print(dijkstra(graph, 0))  # Shortest distance from 0

27.# 0/1 Knapsack (Dynamic Programming)
def knapsack(weights, values, W):
    n = len(values)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(W+1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w - weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]

weights = [1,2,3]
values = [10, 20, 30]
W = 5
print(knapsack(weights, values, W))

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

29.# All Permutations of a String
def string_permutations(s):
    if len(s) <= 1:
        return [s]
    perms = []
    for i, char in enumerate(s):
        for perm in string_permutations(s[:i]+s[i+1:]):
            perms.append(char + perm)
    return perms

print(string_permutations("abc"))

30.# N-Queens Problem
def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i]-col) == row-i:
                return False
        return True

    def solve(board, row, solutions):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(board, row+1, solutions)

    solutions = []
    solve([-1]*n, 0, solutions)
    return solutions

print(solve_n_queens(4))

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

32.# Longest Common Subsequence (LCS)
def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j]+1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

print(lcs("AGGTAB","GXTXAYB"))

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

34.# Floyd-Warshall Algorithm
def floyd_warshall(graph):
    V = len(graph)
    dist = [row[:] for row in graph]
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])
    return dist

graph = [
    [0, 5, float('inf'), 10],
    [float('inf'), 0, 3, float('inf')],
    [float('inf'), float('inf'), 0, 1],
    [float('inf'), float('inf'), float('inf'), 0]
]

print(floyd_warshall(graph))

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

36.# KMP Pattern Matching Algorithm
def kmp_search(text, pattern):
    def compute_lps(pattern):
        lps = [0]*len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length-1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(pattern)
    i = j = 0
    matches = []
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            matches.append(i-j)
            j = lps[j-1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return matches

print(kmp_search("ABABDABACDABABCABAB","ABABCABAB"))