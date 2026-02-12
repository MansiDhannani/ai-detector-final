90.# Custom Exception Hierarchy
class AppError(Exception):
    pass

class DatabaseError(AppError):
    pass

class ValidationError(AppError):
    pass