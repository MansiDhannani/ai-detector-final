65.# Safe File Writing (Atomic Operation)
import os
import tempfile

def atomic_write(filename, data):
    dir_name = os.path.dirname(filename)
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tf:
        tf.write(data)
        temp_name = tf.name
    os.replace(temp_name, filename)