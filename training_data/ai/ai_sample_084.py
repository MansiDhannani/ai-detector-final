82.# Backup and Restore Database (SQLite)
import shutil

def backup_db(db_file, backup_file):
    shutil.copy(db_file, backup_file)

def restore_db(backup_file, db_file):
    shutil.copy(backup_file, db_file)