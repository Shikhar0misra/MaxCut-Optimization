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
        brute_cut REAL,
        greedy_cut REAL,
        brute_time REAL,
        greedy_time REAL,
        approx_ratio REAL,
        vqe_cut REAL
    )
    """)

    c.execute("PRAGMA table_info(experiments)")
    columns = [column[1] for column in c.fetchall()]

    if "vqe_cut" not in columns:
        c.execute("ALTER TABLE experiments ADD COLUMN vqe_cut REAL")

    conn.commit()
    conn.close()


def insert_experiment(data):

    conn = sqlite3.connect("maxcut.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO experiments(nodes,edges,brute_cut,greedy_cut,
    brute_time,greedy_time,approx_ratio,vqe_cut)
    VALUES (?,?,?,?,?,?,?,?)
    """, data)

    conn.commit()
    conn.close()


def load_data():

    conn = sqlite3.connect("maxcut.db")
    df = pd.read_sql_query("SELECT * FROM experiments", conn)
    conn.close()

    return df
