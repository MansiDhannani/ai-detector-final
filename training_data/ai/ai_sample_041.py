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