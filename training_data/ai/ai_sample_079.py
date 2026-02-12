77.# Bulk Inserts Efficiently
def bulk_insert(cursor, query, data):
    cursor.executemany(query, data)