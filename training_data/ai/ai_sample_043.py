41.# Reverse a String (No Built-ins)
def reverse_string(s):
    result = ""
    for ch in s:
        result = ch + result
    return result