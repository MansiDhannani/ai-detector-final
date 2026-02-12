89.# Generate Random Test Data
import random
import string

def random_data(n=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))