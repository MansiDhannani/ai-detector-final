81.# Query Builder Class
class QueryBuilder:
    def __init__(self):
        self.query = "SELECT *"
        self.table = ""
        self.conditions = []

    def select(self, table):
        self.table = table
        return self

    def where(self, condition):
        self.conditions.append(condition)
        return self

    def build(self):
        q = f"{self.query} FROM {self.table}"
        if self.conditions:
            q += " WHERE " + " AND ".join(self.conditions)
        return q