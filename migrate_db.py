
import sqlite3
import os

# Define the expected columns and their SQLite types
# Note: we only need the ones that might be missing
columns_to_add = [
    ("image_path", "VARCHAR(255)"),
    ("tags", "VARCHAR(255)"),
    ("raw_fir_text", "TEXT"),
    ("evidence_summary", "TEXT"),
    ("weapon_used", "VARCHAR(120)"),
    ("victim_age", "INTEGER"),
    ("victim_gender", "VARCHAR(10)"),
    ("reported_at", "DATETIME")
]

db_path = 'ciris.db'

if not os.path.exists(db_path):
    print(f"Database {db_path} not found.")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get existing columns
cursor.execute("PRAGMA table_info(firs)")
existing_cols = [row[1] for row in cursor.fetchall()]

for col_name, col_type in columns_to_add:
    if col_name not in existing_cols:
        print(f"Adding column {col_name} to firs table...")
        try:
            cursor.execute(f"ALTER TABLE firs ADD COLUMN {col_name} {col_type}")
            print(f"Column {col_name} added successfully.")
        except Exception as e:
            print(f"Failed to add column {col_name}: {e}")

conn.commit()
conn.close()
print("Migration check complete.")
