"""
Calculate Fibonacci numbers efficiently.

This function implements the Fibonacci sequence calculation
using dynamic programming approach for optimal performance.

Args:
    n (int): The position in Fibonacci sequence
    
Returns:
    int: The nth Fibonacci number
    
Example:
    >>> fibonacci(10)
    55
"""

def fibonacci(n):
    """Helper function to calculate Fibonacci."""
    # Initialize base cases
    if n <= 1:
        return n
    
    # Create array to store previously calculated values
    fib = [0] * (n + 1)
    fib[1] = 1
    
    # Calculate Fibonacci iteratively
    for i in range(2, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
    
    return fib[n]
