import joblib

# ------------------------------------
# HUMAN-WRITTEN CODE SAMPLES
# ------------------------------------
human_codes = [
    """
def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
""",
    """
def search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1
""",
    """
def sum_list(lst):
    s = 0
    for x in lst:
        s += x
    return s
"""
]

# ------------------------------------
# AI-GENERATED CODE SAMPLES
# ------------------------------------
ai_codes = [
    """
def calculate_fibonacci(n):
    \"\"\"
    Calculate the nth Fibonacci number using an efficient approach.

    Args:
        n (int): Index of Fibonacci number

    Returns:
        int: Fibonacci value
    \"\"\"
    if n <= 1:
        return n

    prev, curr = 0, 1
    for _ in range(n - 1):
        prev, curr = curr, prev + curr

    return curr
""",
    """
def binary_search(arr, target):
    \"\"\"Perform binary search on a sorted list.\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
]

# ------------------------------------
# BUILD DATASET
# ------------------------------------
data = []

for code in human_codes:
    data.append({"code": code, "label": 0})

for code in ai_codes:
    data.append({"code": code, "label": 1})

# ------------------------------------
# SAVE
# ------------------------------------
joblib.dump(data, "training_data.pkl")

print(f"✅ Saved training_data.pkl with {len(data)} samples")
