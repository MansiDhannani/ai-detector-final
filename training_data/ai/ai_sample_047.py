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