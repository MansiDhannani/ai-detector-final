78.# Simple ORM Mapper
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def save(self, db):
        db.execute(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            (self.name, self.age)
        )