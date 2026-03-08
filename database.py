import sqlite3
import pandas as pd

def init_db():

    conn = sqlite3.connect("maxcut.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS experiments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nodes INTEGER,
        edges INTEGER,
        brute_cut INTEGER,
        greedy_cut INTEGER,
        brute_time REAL,
        greedy_time REAL,
        approx_ratio REAL
    )
    """)

    conn.commit()
    conn.close()


def insert_experiment(data):

    conn = sqlite3.connect("maxcut.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO experiments(nodes,edges,brute_cut,greedy_cut,
    brute_time,greedy_time,approx_ratio)
    VALUES (?,?,?,?,?,?,?)
    """, data)

    conn.commit()
    conn.close()


def load_data():

    conn = sqlite3.connect("maxcut.db")
    df = pd.read_sql_query("SELECT * FROM experiments", conn)
    conn.close()

    return df