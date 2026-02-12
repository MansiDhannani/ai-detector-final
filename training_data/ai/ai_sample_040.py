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