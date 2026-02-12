54.# Check if Strings Are Rotations
def are_rotations(s1, s2):
    if len(s1) != len(s2):
        return False
    return s2 in s1 + s1