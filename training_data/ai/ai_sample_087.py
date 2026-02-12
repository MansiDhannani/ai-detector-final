85.# Sanitize SQL Inputs
def sanitize(value):
    return value.replace("'", "''")