import sqlite3

db_path = "sql_app.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("PRAGMA table_info(videos)")
cols = [row[1] for row in cur.fetchall()]
print("현재 컬럼:", cols)

if "source_title" not in cols:
    cur.execute("ALTER TABLE videos ADD COLUMN source_title TEXT")
    conn.commit()
    print("source_title 컬럼 추가 완료!")
else:
    print("source_title 컬럼 이미 존재")

conn.close()
