37. #A* Pathfinding Algorithm
import heapq

def a_star(start, goal, graph, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))

    g_cost = {start: 0}
    parent = {start: None}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for neighbor, cost in graph[current]:
            new_cost = g_cost[current] + cost
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic(neighbor)
                heapq.heappush(open_set, (f_cost, neighbor))
                parent[neighbor] = current

    return None

38.# Matrix Chain Multiplication
def matrix_chain_order(p):
    n = len(p) - 1
    dp = [[0]*n for _ in range(n)]

    for l in range(2, n+1):
        for i in range(n-l+1):
            j = i + l - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + p[i]*p[k+1]*p[j+1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]

39.# Rabin–Karp String Matching
def rabin_karp(text, pattern):
    d, q = 256, 101
    n, m = len(text), len(pattern)
    h = pow(d, m-1) % q
    p = t = 0
    result = []

    for i in range(m):
        p = (d*p + ord(pattern[i])) % q
        t = (d*t + ord(text[i])) % q

    for i in range(n-m+1):
        if p == t and text[i:i+m] == pattern:
            result.append(i)
        if i < n-m:
            t = (d*(t - ord(text[i])*h) + ord(text[i+m])) % q

    return result

40.# Optimized Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

41.# Reverse a String (No Built-ins)
def reverse_string(s):
    result = ""
    for ch in s:
        result = ch + result
    return result

42.# Check Palindrome
def is_palindrome(s):
    return s == s[::-1]

43.# Find All Anagrams
from collections import Counter

def find_anagrams(word, words):
    target = Counter(word)
    return [w for w in words if Counter(w) == target]

44.# String Compression
def compress_string(s):
    result = ""
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result += s[i-1] + str(count)
            count = 1

    result += s[-1] + str(count)
    return result

45.# Validate Parentheses
def is_valid_parentheses(s):
    stack = []
    pairs = {')':'(', '}':'{', ']':'['}

    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False

    return not stack

46.# Longest Substring Without Repeating Characters
def longest_unique_substring(s):
    seen = {}
    start = max_len = 0

    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        max_len = max(max_len, i - start + 1)

    return max_len

47.# Convert String to snake_case
import re

def to_snake_case(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
    return s.lower()
