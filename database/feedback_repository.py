# database/feedback_repository.py

import sqlite3

def save(user_id, anime_id, rating):
    conn = sqlite3.connect("feedback.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback(user_id, anime_id, rating) VALUES(?, ?, ?)",
        (user_id, anime_id, rating)
    )
    conn.commit()
    conn.close()