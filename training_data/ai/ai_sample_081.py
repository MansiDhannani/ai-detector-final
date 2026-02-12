79.# Database Connection Pool Manager
import queue
import sqlite3

class ConnectionPool:
    def __init__(self, db, size=5):
        self.pool = queue.Queue(size)
        for _ in range(size):
            self.pool.put(sqlite3.connect(db))

    def acquire(self):
        return self.pool.get()

    def release(self, conn):
        self.pool.put(conn)