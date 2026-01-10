import sqlite3
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Tuple

DB_NAME = "pipecraft.db"

class DatabaseRepository:
    @staticmethod
    def init_db():
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        iso TEXT, naht TEXT, datum TEXT, 
                        dimension TEXT, bauteil TEXT, laenge REAL, 
                        charge TEXT, charge_apz TEXT, schweisser TEXT,
                        project_id INTEGER)''')
            c.execute('''CREATE TABLE IF NOT EXISTS projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        created_at TEXT,
                        archived INTEGER DEFAULT 0)''') 
            c.execute("PRAGMA table_info(rohrbuch)")
            cols = [info[1] for info in c.fetchall()]
            if 'charge_apz' not in cols:
                try: c.execute("ALTER TABLE rohrbuch ADD COLUMN charge_apz TEXT")
                except: pass
            if 'project_id' not in cols:
                try: c.execute("ALTER TABLE rohrbuch ADD COLUMN project_id INTEGER")
                except: pass
            c.execute("PRAGMA table_info(projects)")
            p_cols = [info[1] for info in c.fetchall()]
            if 'archived' not in p_cols:
                try: c.execute("ALTER TABLE projects ADD COLUMN archived INTEGER DEFAULT 0")
                except: pass
            c.execute("INSERT OR IGNORE INTO projects (id, name, created_at, archived) VALUES (1, 'Standard Baustelle', ?, 0)", 
                      (datetime.now().strftime("%d.%m.%Y"),))
            c.execute("UPDATE rohrbuch SET project_id = 1 WHERE project_id IS NULL")
            conn.commit()

    @staticmethod
    def get_projects() -> List[tuple]:
        with sqlite3.connect(DB_NAME) as conn:
            return conn.cursor().execute("SELECT id, name, archived FROM projects ORDER BY id ASC").fetchall()

    @staticmethod
    def create_project(name: str):
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.cursor().execute("INSERT INTO projects (name, created_at, archived) VALUES (?, ?, 0)", (name, datetime.now().strftime("%d.%m.%Y")))
                conn.commit()
            return True, "Projekt erstellt."
        except sqlite3.IntegrityError:
            return False, "Name existiert bereits."

    @staticmethod
    def toggle_archive_project(project_id: int, archive: bool):
        val = 1 if archive else 0
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute("UPDATE projects SET archived = ? WHERE id = ?", (val, project_id))
            conn.commit()

    @staticmethod
    def add_entry(data: dict):
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            pid = data.get('project_id', 1)
            if pid is None: pid = 1
            c.execute('''INSERT INTO rohrbuch 
                         (iso, naht, datum, dimension, bauteil, laenge, charge, charge_apz, schweisser, project_id) 
                         VALUES (:iso, :naht, :datum, :dimension, :bauteil, :laenge, :charge, :charge_apz, :schweisser, :project_id)''', 
                         dict(data, project_id=pid))
            conn.commit()

    @staticmethod
    def get_logbook_by_project(project_id: int) -> pd.DataFrame:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query("SELECT * FROM rohrbuch WHERE project_id = ? ORDER BY id DESC", conn, params=(project_id,))
            if not df.empty: 
                df['✏️'] = False 
                df['Löschen'] = False
            else: 
                df = pd.DataFrame(columns=["id", "iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser", "project_id", "✏️", "Löschen"])
            return df

    @staticmethod
    def update_full_entry(entry_id: int, data: dict):
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''UPDATE rohrbuch 
                         SET iso = :iso, naht = :naht, datum = :datum, 
                             dimension = :dimension, bauteil = :bauteil, laenge = :laenge,
                             charge_apz = :charge_apz, schweisser = :schweisser
                         WHERE id = :id''', 
                         dict(data, id=entry_id))
            conn.commit()

    @staticmethod
    def delete_entries(ids: List[int]):
        if not ids: return
        with sqlite3.connect(DB_NAME) as conn:
            placeholders = ', '.join('?' for _ in ids)
            conn.cursor().execute(f"DELETE FROM rohrbuch WHERE id IN ({placeholders})", ids)
            conn.commit()

    @staticmethod
    def bulk_update(ids: List[int], field: str, value: str):
        if not ids: return
        allowed_map = {
            "Schweißer": "schweisser",
            "APZ / Charge": "charge_apz",
            "ISO": "iso",
            "Datum": "datum"
        }
        db_col = allowed_map.get(field)
        if not db_col: return

        with sqlite3.connect(DB_NAME) as conn:
            placeholders = ', '.join('?' for _ in ids)
            query = f"UPDATE rohrbuch SET {db_col} = ? WHERE id IN ({placeholders})"
            args = [value] + ids
            conn.cursor().execute(query, args)
            conn.commit()

    @staticmethod
    def get_known_values(column: str, project_id: int, limit: int = 50) -> List[str]:
        allowed = ['charge', 'charge_apz', 'schweisser', 'iso']
        if column not in allowed: return []
        with sqlite3.connect(DB_NAME) as conn:
            query = f'''SELECT {column} FROM rohrbuch WHERE project_id = ? AND {column} IS NOT NULL AND {column} != '' GROUP BY {column} ORDER BY MAX(id) DESC LIMIT ?'''
            rows = conn.cursor().execute(query, (project_id, limit)).fetchall()
            return [r[0] for r in rows]

    @staticmethod
    def export_project_to_json(project_id: int) -> str:
        with sqlite3.connect(DB_NAME) as conn:
            proj = conn.cursor().execute("SELECT name, created_at FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not proj: return None
            rows = conn.cursor().execute("SELECT iso, naht, datum, dimension, bauteil, laenge, charge, charge_apz, schweisser FROM rohrbuch WHERE project_id = ?", (project_id,)).fetchall()
            cols = ["iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser"]
            entries = [dict(zip(cols, r)) for r in rows]
            data = {"project_name": proj[0], "created_at": proj[1], "entries": entries, "version": "1.6"}
            return json.dumps(data, indent=2)

    @staticmethod
    def import_project_from_json(json_str: str) -> Tuple[bool, str]:
        try:
            data = json.loads(json_str)
            name = data.get("project_name") + " (Import)"
            entries = data.get("entries", [])
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                try:
                    c.execute("INSERT INTO projects (name, created_at, archived) VALUES (?, ?, 0)", (name, datetime.now().strftime("%d.%m.%Y")))
                    new_pid = c.lastrowid
                except sqlite3.IntegrityError:
                    name += f"_{int(time.time())}"
                    c.execute("INSERT INTO projects (name, created_at, archived) VALUES (?, ?, 0)", (name, datetime.now().strftime("%d.%m.%Y")))
                    new_pid = c.lastrowid
                for e in entries:
                    c.execute('''INSERT INTO rohrbuch (iso, naht, datum, dimension, bauteil, laenge, charge, charge_apz, schweisser, project_id) 
                                 VALUES (:iso, :naht, :datum, :dimension, :bauteil, :laenge, :charge, :charge_apz, :schweisser, :project_id)''',
                                 dict(e, project_id=new_pid))
                conn.commit()
            return True, f"Projekt '{name}' importiert!"
        except Exception as e:
            return False, f"Fehler: {str(e)}"
