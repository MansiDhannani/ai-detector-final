83.# Simple Transaction Manager
class Transaction:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if exc:
            self.conn.rollback()
        else:
            self.conn.commit()