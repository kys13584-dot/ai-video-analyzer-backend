import sqlite3
conn = sqlite3.connect("sql_app.db")
cur = conn.cursor()
cur.execute("SELECT id, title, source_title FROM videos WHERE status='completed' ORDER BY id")
for row in cur.fetchall():
    print(f"ID={row[0]}")
    print(f"  title:        {row[1]}")
    print(f"  source_title: {row[2]}")
    print()
conn.close()
