import csv
import sqlite3
import os

def save_results_csv(history, avg_history, std_history):
    with open("results.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoka", "Najlepszy", "Średnia", "Odchylenie std"])
        for epoch, (best, avg, std) in enumerate(zip(history, avg_history, std_history)):
            writer.writerow([epoch + 1, best, avg, std])

def save_results_db(history, avg_history, std_history, db_path="results.db"):
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER,
                best_value REAL,
                average_value REAL,
                std_deviation REAL
            )
        """)
        conn.commit()
        conn.close()

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for epoch, (best, avg, std) in enumerate(zip(history, avg_history, std_history)):
        c.execute("INSERT INTO results (epoch, best_value, average_value, std_deviation) VALUES (?, ?, ?, ?)",
                  (epoch + 1, best, avg, std))
    conn.commit()
    conn.close()
