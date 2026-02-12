from hybrid_ai_detector import HybridAIDetector
import json
import sys

# ============================================================
# LOAD PRETRAINED DETECTOR
# ============================================================

try:
    detector = HybridAIDetector(use_gpu=False)
    detector.load_pretrained()
except Exception as e:
    print("❌ Failed to load pretrained model.")
    print("Reason:", e)
    sys.exit(1)

# ============================================================
# TEST CASES
# ============================================================

tests = [
    {
        "name": "AI Code (ChatGPT style)",
        "label": 1,  # 1 = AI, 0 = Human
        "code": """
def calculate_fibonacci(n):
    \"\"\"
    Calculate the nth Fibonacci number using dynamic programming.
    
    This function implements an efficient iterative approach to
    calculate Fibonacci numbers.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
    \"\"\"
    # Initialize base cases
    if n <= 1:
        return n
    
    fib = [0] * (n + 1)
    fib[1] = 1
    
    for i in range(2, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
    
    return fib[n]
"""
    },
    {
        "name": "Human Code (casual style)",
        "label": 0,
        "code": """
def fib(n):
    # quick fib calc
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
"""
    },
    {
        "name": "AI Code (GitHub Copilot style)",
        "label": 1,
        "code": """
def binary_search(arr, target):
    \"\"\"Perform binary search on sorted array.\"\"\"
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
    },
    {
        "name": "Human Code (messy)",
        "label": 0,
        "code": """
def search(lst,x):
    for i in range(len(lst)):
        if lst[i]==x:
            return i
    return -1
"""
    }
]

# ============================================================
# RUN TESTS
# ============================================================

print("=" * 70)
print("AI CODE DETECTOR - TEST SUITE")
print("=" * 70)

correct = 0
total = len(tests)

for test in tests:
    print(f"\n🧪 Test: {test['name']}")
    print("-" * 70)

    result = detector.analyze(test['code'])

    predicted_label = 1 if result['prediction'] == "AI-generated" else 0
    is_correct = predicted_label == test["label"]

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Language   : {result.get('language', 'unknown')}")

    if is_correct:
        print("✅ CORRECT")
        correct += 1
    else:
        print("❌ INCORRECT")

    # Optional detailed debug
    print("Components:")
    print(json.dumps(result["components"], indent=2))

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("=" * 70)
