#!/usr/bin/env python3
"""
AI Code Generator for Training Data
Generates AI code samples using ChatGPT/Claude
"""

from pathlib import Path
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')


class AICodeGenerator:
    """Generate AI code training samples"""
    
    def __init__(self, output_dir='training_data/ai'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 100 diverse prompts covering different scenarios
        self.prompts = self.generate_prompts()
    
    def generate_prompts(self):
        """Generate 100 diverse coding prompts"""
        
        prompts = []
        
        # Category 1: Data Structures (20 prompts)
        ds_prompts = [
            "Write a Python class for a binary search tree with insert, search, and delete methods",
            "Implement a stack using a linked list in Python",
            "Create a queue class with enqueue, dequeue, and peek operations",
            "Write a function to implement a hash table with collision handling",
            "Implement a min heap data structure in Python",
            "Create a doubly linked list with insert and delete operations",
            "Write a class for a circular buffer",
            "Implement a trie (prefix tree) for string searching",
            "Create a graph class with adjacency list representation",
            "Write a priority queue using heapq",
            "Implement a disjoint set (union-find) data structure",
            "Create a balanced binary search tree (AVL tree)",
            "Write a function to implement LRU cache",
            "Implement a bloom filter in Python",
            "Create a sparse matrix class",
            "Write a segment tree for range queries",
            "Implement a skip list data structure",
            "Create a B-tree structure",
            "Write a red-black tree implementation",
            "Implement a suffix tree"
        ]
        prompts.extend(ds_prompts)
        
        # Category 2: Algorithms (20 prompts)
        algo_prompts = [
            "Write a function to perform quicksort on a list",
            "Implement merge sort algorithm in Python",
            "Create a function for binary search",
            "Write a breadth-first search algorithm for graphs",
            "Implement depth-first search for tree traversal",
            "Create a function for finding the shortest path using Dijkstra's algorithm",
            "Write a dynamic programming solution for the knapsack problem",
            "Implement the Bellman-Ford algorithm",
            "Create a function to find all permutations of a string",
            "Write a function to solve the N-queens problem",
            "Implement Kruskal's algorithm for minimum spanning tree",
            "Create a function for longest common subsequence",
            "Write an algorithm to detect cycles in a graph",
            "Implement Floyd-Warshall algorithm",
            "Create a function for topological sorting",
            "Write a KMP pattern matching algorithm",
            "Implement A* pathfinding algorithm",
            "Create a function for matrix chain multiplication",
            "Write a Rabin-Karp string matching algorithm",
            "Implement bubble sort with optimization"
        ]
        prompts.extend(algo_prompts)
        
        # Category 3: String Manipulation (15 prompts)
        string_prompts = [
            "Write a function to reverse a string without using built-in reverse",
            "Create a function to check if a string is a palindrome",
            "Implement a function to find all anagrams of a word",
            "Write a function to compress a string (e.g., 'aaabb' -> 'a3b2')",
            "Create a function to validate parentheses in an expression",
            "Implement a function to find the longest substring without repeating characters",
            "Write a function to convert a string to snake_case",
            "Create a function to extract all email addresses from text",
            "Implement a function to generate all possible substrings",
            "Write a function to calculate edit distance between two strings",
            "Create a function to parse a CSV string with quoted fields",
            "Implement a function to validate email addresses with regex",
            "Write a function to find the most frequent word in text",
            "Create a function to check if strings are rotations of each other",
            "Implement a function to generate acronyms from phrases"
        ]
        prompts.extend(string_prompts)
        
        # Category 4: File & I/O Operations (10 prompts)
        io_prompts = [
            "Write a function to read a large file line by line efficiently",
            "Create a function to parse JSON files with error handling",
            "Implement a file watcher that monitors directory changes",
            "Write a function to merge multiple CSV files",
            "Create a context manager for file operations",
            "Implement a function to recursively search for files",
            "Write a function to copy files with progress tracking",
            "Create a function to compress and decompress files",
            "Implement a log file analyzer",
            "Write a function to safely write to files with atomic operations"
        ]
        prompts.extend(io_prompts)
        
        # Category 5: Web & API (10 prompts)
        web_prompts = [
            "Create a simple REST API client with requests",
            "Write a function to scrape product prices from a website",
            "Implement a rate limiter for API calls",
            "Create a function to download files from URLs with retry logic",
            "Write a web scraper with BeautifulSoup",
            "Implement a simple HTTP server",
            "Create a function to parse RSS feeds",
            "Write an async API client using aiohttp",
            "Implement a function to validate URLs",
            "Create a simple web crawler"
        ]
        prompts.extend(web_prompts)
        
        # Category 6: Database & ORM (10 prompts)
        db_prompts = [
            "Write a SQLite database wrapper class",
            "Create a function to perform bulk inserts efficiently",
            "Implement a simple ORM mapper",
            "Write a database connection pool manager",
            "Create a function to migrate database schema",
            "Implement a query builder class",
            "Write a function to backup and restore database",
            "Create a simple transaction manager",
            "Implement a database caching layer",
            "Write a function to sanitize SQL inputs"
        ]
        prompts.extend(db_prompts)
        
        # Category 7: Testing & Utilities (15 prompts)
        util_prompts = [
            "Write a decorator to measure function execution time",
            "Create a retry decorator with exponential backoff",
            "Implement a memoization decorator",
            "Write a function to generate random test data",
            "Create a custom exception class hierarchy",
            "Implement a logger with different severity levels",
            "Write a function to validate credit card numbers",
            "Create a date/time parser for various formats",
            "Implement a simple dependency injection container",
            "Write a function to calculate file checksums",
            "Create a configuration manager for app settings",
            "Implement a simple event emitter/listener pattern",
            "Write a function to generate secure random passwords",
            "Create a rate-limited function decorator",
            "Implement a simple state machine"
        ]
        prompts.extend(util_prompts)
        
        return prompts[:100]  # Ensure exactly 100
    
    def create_collection_template(self):
        """Create a template for manual collection"""
        
        template_path = self.output_dir / 'COLLECTION_INSTRUCTIONS.txt'
        
        template = f"""
# AI CODE COLLECTION INSTRUCTIONS
# ================================

GOAL: Collect 100-200 AI-generated code samples

METHOD 1: ChatGPT (Recommended)
--------------------------------
1. Open ChatGPT (chat.openai.com)
2. Copy each prompt below
3. Paste into ChatGPT
4. Copy ENTIRE code response
5. Save as ai_sample_XXX.py in this folder

METHOD 2: Claude
--------------------------------
Same process as ChatGPT using claude.ai

METHOD 3: GitHub Copilot
--------------------------------
1. Open VS Code with Copilot
2. Type prompt as comment
3. Let Copilot generate code
4. Save suggestion

IMPORTANT TIPS:
--------------
✅ Copy the COMPLETE code response
✅ Include comments and docstrings  
✅ Don't edit or clean the code
✅ Save exactly as AI generated
✅ Name files: ai_sample_001.py, ai_sample_002.py, etc.

PROMPTS (100 total):
===================

"""
        
        for i, prompt in enumerate(self.prompts, 1):
            template += f"\n{i}. {prompt}\n"
        
        template += """

COLLECTION CHECKLIST:
====================
[ ] Got ChatGPT/Claude account
[ ] Created training_data/ai/ folder
[ ] Started with prompt #1
[ ] Saving each response as ai_sample_XXX.py
[ ] Target: 100 samples minimum
[ ] Current progress: ___/100

QUALITY CHECK:
=============
- Each file should be 20-100 lines
- Code should be complete and runnable
- Include AI-typical features (docstrings, comments, error handling)
- Diverse types (algorithms, classes, functions)

After collecting 100 samples, run:
  python check_dataset.py
"""
        
        template_path.write_text(template)
        print(f"✅ Template saved to: {template_path}")
        
        return template_path
    
    def create_bulk_prompts_file(self):
        """Create file with all prompts for bulk generation"""
        
        prompts_file = self.output_dir / 'all_prompts.json'
        
        prompt_data = {
            'total': len(self.prompts),
            'prompts': [
                {
                    'id': i,
                    'prompt': prompt,
                    'filename': f'ai_sample_{i:03d}.py'
                }
                for i, prompt in enumerate(self.prompts, 1)
            ]
        }
        
        prompts_file.write_text(json.dumps(prompt_data, indent=2))
        print(f"✅ Prompts JSON saved to: {prompts_file}")
        
        return prompts_file
    
    def generate_sample_ai_code(self):
        """Generate a few sample AI-style codes"""
        
        # Sample 1: Typical ChatGPT style
        sample1 = '''"""
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
'''
        
        (self.output_dir / 'sample_ai_001.py').write_text(sample1)
        
        # Sample 2: Copilot style
        sample2 = '''def binary_search(arr, target):
    """
    Perform binary search on a sorted array.
    
    Args:
        arr: Sorted list of integers
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
    """
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

# Example usage
if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11, 13]
    result = binary_search(arr, 7)
    print(f"Found at index: {result}")
'''
        
        (self.output_dir / 'sample_ai_002.py').write_text(sample2)
        
        print(f"✅ Created 2 sample AI code files")
        print("   Use these as reference for what AI code looks like")

def main():
    """Main execution"""
    
    print("=" * 70)
    print("AI CODE GENERATOR - Training Data Collection")
    print("=" * 70)
    print()
    
    generator = AICodeGenerator()
    
    # Create collection template
    print("📝 Creating collection template...")
    generator.create_collection_template()
    print()
    
    # Create prompts JSON
    print("📋 Creating prompts file...")
    generator.create_bulk_prompts_file()
    print()
    
    # Create samples
    print("🔨 Creating sample AI code...")
    generator.generate_sample_ai_code()
    print()
    
    print("=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print()
    print("NEXT STEPS:")
    print("-----------")
    print("1. Open training_data/ai/COLLECTION_INSTRUCTIONS.txt")
    print("2. Follow the instructions to collect AI code")
    print("3. Use ChatGPT/Claude to generate samples")
    print("4. Save each response as ai_sample_XXX.py")
    print()
    print("ESTIMATED TIME:")
    print("---------------")
    print("• 100 samples × 2 minutes each = 3-4 hours total")
    print("• Can do in batches (20-30 per session)")
    print()
    print("TIP: Open ChatGPT in one window, save files in another")
    print("=" * 70)

if __name__ == "__main__":
    main()
