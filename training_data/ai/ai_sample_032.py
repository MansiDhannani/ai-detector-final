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