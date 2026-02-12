76.# SQLite Database Wrapper Class
import sqlite3

class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def execute(self, query, params=()):
        self.cursor.execute(query, params)
        self.conn.commit()

    def fetchall(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()