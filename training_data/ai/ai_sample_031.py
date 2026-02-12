29.# All Permutations of a String
def string_permutations(s):
    if len(s) <= 1:
        return [s]
    perms = []
    for i, char in enumerate(s):
        for perm in string_permutations(s[:i]+s[i+1:]):
            perms.append(char + perm)
    return perms

print(string_permutations("abc"))