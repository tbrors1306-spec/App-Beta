import time
import logging
import sqlite3
import math
import re
import json
from dataclasses import dataclass, field, asdict
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# 0. SICHERE IMPORTS
# -----------------------------------------------------------------------------
# Wir haben AgGrid komplett entfernt. Nur FPDF bleibt optional.
PDF_AVAILABLE = False
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PDF_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. KONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipeCraft_V3_Native")

st.set_page_config(
    page_title="PipeCraft v3.0 (Native)",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; background-color: #f8fafc; }
    h1, h2, h3, h4, h5 { font-family: 'Segoe UI', sans-serif; font-weight: 600; color: #1e293b; }
    
    /* Navigation Style */
    div.row-widget.stRadio > div { flex-direction: row; align-items: stretch; }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] { 
        background-color: #ffffff; border: 1px solid #e2e8f0; padding: 10px 20px; border-radius: 8px; margin-right: 10px;
    }
    
    .machine-header-saw { border-bottom: 4px solid #f97316; color: #f97316; padding: 5px 0; font-weight: 700; font-size: 1.2rem; margin-bottom: 15px; text-transform: uppercase; }
    .machine-header-geo { border-bottom: 4px solid #0ea5e9; color: #0ea5e9; padding: 5px 0; font-weight: 700; font-size: 1.2rem; margin-bottom: 15px; text-transform: uppercase; }
    .machine-header-doc { border-bottom: 4px solid #64748b; color: #64748b; padding: 5px 0; font-weight: 700; font-size: 1.2rem; margin-bottom: 15px; text-transform: uppercase; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; }
    div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px; }
    .project-tag { font-family: 'Segoe UI', sans-serif; font-weight: 600; color: #475569; padding: 8px 12px; background-color: #e2e8f0; border-radius: 6px; margin-bottom: 20px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATEN-SCHICHT
# -----------------------------------------------------------------------------

@st.cache_data
def get_pipe_data() -> pd.DataFrame:
    raw_data = {
        'DN':             [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600],
        'D_Aussen':       [33.7, 42.4, 48.3, 60.3, 76.1, 88.9, 114.3, 139.7, 168.3, 219.1, 273.0, 323.9, 355.6, 406.4, 457.0, 508.0, 610.0, 711.0, 813.0, 914.0, 1016.0, 1219.0, 1422.0, 1626.0],
        'Radius_BA3':     [38, 48, 57, 76, 95, 114, 152, 190, 229, 305, 381, 457, 533, 610, 686, 762, 914, 1067, 1219, 1372, 1524, 1829, 2134, 2438],
        'T_Stueck_H':     [25, 32, 38, 51, 64, 76, 105, 124, 143, 178, 216, 254, 279, 305, 343, 381, 432, 521, 597, 673, 749, 889, 1029, 1168],
        'Red_Laenge_L':   [38, 50, 64, 76, 89, 89, 102, 127, 140, 152, 178, 203, 330, 356, 381, 508, 508, 610, 660, 711, 800, 900, 1000, 1100], 
        'Flansch_b_16':   [38, 40, 42, 45, 45, 50, 52, 55, 55, 62, 70, 78, 82, 85, 85, 90, 95, 105, 115, 125, 135, 155, 175, 195],
        'LK_k_16':        [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 355, 410, 470, 525, 585, 650, 770, 840, 950, 1050, 1160, 1380, 1590, 1820],
        'Schraube_M_16': ["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M24", "M24", "M24", "M27", "M27", "M30", "M33", "M33", "M36", "M36", "M39", "M45", "M45", "M52"],
        'L_Fest_16':      [55, 60, 60, 65, 65, 70, 70, 75, 80, 85, 100, 110, 110, 120, 130, 130, 150, 160, 170, 180, 190, 220, 240, 260],
        'L_Los_16':       [60, 65, 65, 70, 70, 75, 80, 85, 90, 100, 115, 125, 130, 140, 150, 150, 170, 180, 190, 210, 220, 250, 280, 300],
        'Lochzahl_16':    [4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 16, 16, 20, 20, 20, 24, 24, 28, 28, 32, 36, 40],
        'Flansch_b_10':   [38, 40, 42, 45, 45, 50, 52, 55, 55, 62, 70, 78, 82, 85, 85, 90, 95, 105, 115, 125, 135, 155, 175, 195],
        'LK_k_10':        [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 350, 400, 460, 515, 565, 620, 725, 840, 950, 1050, 1160, 1380, 1590, 1820],
        'Schraube_M_10': ["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M20", "M20", "M20", "M24", "M24", "M24", "M27", "M27", "M30", "M30", "M33", "M36", "M39", "M45"],
        'L_Fest_10':      [55, 60, 60, 65, 65, 70, 70, 75, 80, 85, 90, 90, 90, 100, 110, 110, 120, 130, 140, 150, 160, 190, 210, 230],
        'L_Los_10':       [60, 65, 65, 70, 70, 75, 80, 85, 90, 100, 105, 105, 110, 120, 130, 130, 140, 150, 160, 170, 180, 210, 240, 260],
        'Lochzahl_10':    [4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 16, 16, 20, 20, 20, 20, 24, 28, 28, 32, 36, 40]
    }
    return pd.DataFrame(raw_data)

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

# -----------------------------------------------------------------------------
# 3. HELPER & LOGIK
# -----------------------------------------------------------------------------

@dataclass
class FittingItem:
    id: str
    name: str
    count: int
    deduction_single: float
    dn: int
    @property
    def total_deduction(self) -> float: return self.deduction_single * self.count

@dataclass
class SavedCut:
    id: int
    name: str
    raw_length: float
    cut_length: float
    details: str
    timestamp: str
    fittings: List[FittingItem] = field(default_factory=list)

class PipeCalculator:
    PN_MAP = {
        "PN 16": "_16",
        "PN 10": "_10",
        "PN 6": "_10",
        "PN 25": "_16", 
        "PN 40": "_16" 
    }

    def __init__(self, df: pd.DataFrame): self.df = df
    
    def get_row(self, dn: int) -> pd.Series:
        row = self.df[self.df['DN'] == dn]
        return row.iloc[0] if not row.empty else self.df.iloc[0]
        
    def get_deduction(self, f_type: str, dn: int, pn: str, angle: float = 90.0) -> float:
        row = self.get_row(dn)
        suffix = self.PN_MAP.get(pn, "_10")
        
        if "Bogen 90°" in f_type: return float(row['Radius_BA3'])
        if "Zuschnitt" in f_type: return float(row['Radius_BA3']) * math.tan(math.radians(angle / 2))
        if "Flansch" in f_type: return float(row[f'Flansch_b{suffix}'])
        if "T-Stück" in f_type: return float(row['T_Stueck_H'])
        if "Reduzierung" in f_type: return float(row['Red_Laenge_L'])
        return 0.0
        
    def calculate_bend_details(self, dn: int, angle: float) -> Dict[str, float]:
        row = self.get_row(dn)
        r = float(row['Radius_BA3'])
        da = float(row['D_Aussen'])
        rad = math.radians(angle)
        return {"vorbau": r * math.tan(rad / 2), "bogen_aussen": (r + da/2) * rad, "bogen_mitte": r * rad, "bogen_innen": (r - da/2) * rad}
        
    def calculate_stutzen_coords(self, dn_haupt: int, dn_stutzen: int) -> pd.DataFrame:
        r_main = self.get_row(dn_haupt)['D_Aussen'] / 2
        r_stub = self.get_row(dn_stutzen)['D_Aussen'] / 2
        if r_stub > r_main: raise ValueError("Stutzen > Hauptrohr")
        table_data = []
        for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
            try:
                term = r_stub * math.sin(math.radians(angle))
                t_val = r_main - math.sqrt(r_main**2 - term**2)
            except: t_val = 0
            u_val = (r_stub * 2 * math.pi) * (angle / 360)
            table_data.append({"Winkel": f"{angle}°", "Tiefe (mm)": round(t_val, 1), "Umfang (mm)": round(u_val, 1)})
        return pd.DataFrame(table_data)
        
    def calculate_2d_offset(self, dn: int, offset: float, angle: float) -> Dict[str, float]:
        row = self.get_row(dn)
        r = float(row['Radius_BA3'])
        rad = math.radians(angle)
        try:
            hypotenuse = offset / math.sin(rad)
            run = offset / math.tan(rad)
        except: return {"error": "Winkel 0"}
        z_mass = r * math.tan(rad / 2)
        return {"hypotenuse": hypotenuse, "run": run, "z_mass_single": z_mass, "cut_length": hypotenuse - (2*z_mass), "offset": offset, "angle": angle}
        
    def calculate_rolling_offset(self, dn: int, roll: float, set_val: float, height: float = 0.0) -> Dict[str, float]:
        diag_base = math.sqrt(roll**2 + set_val**2)
        travel = math.sqrt(diag_base**2 + height**2)
        try: required_angle = math.degrees(math.acos(diag_base / travel)) if travel != 0 else 0
        except: required_angle = 0
        return {"travel": travel, "diag_base": diag_base, "angle_calc": required_angle}
        
    def calculate_segment_bend(self, dn: int, radius: float, num_segments: int, total_angle: float = 90.0) -> Dict[str, float]:
        row = self.get_row(dn)
        od = float(row['D_Aussen'])
        if num_segments < 2: return {"error": "Min. 2 Segmente"}
        miter_angle = total_angle / (2 * (num_segments - 1))
        tan_alpha = math.tan(math.radians(miter_angle))
        len_center = 2 * radius * tan_alpha
        len_back = 2 * (radius + od/2) * tan_alpha
        len_belly = 2 * (radius - od/2) * tan_alpha
        end_back = (radius + od/2) * tan_alpha
        end_belly = (radius - od/2) * tan_alpha
        end_center = radius * tan_alpha
        return {"miter_angle": miter_angle, "mid_back": len_back, "mid_belly": len_belly, "mid_center": len_center, "end_back": end_back, "end_belly": end_belly, "end_center": end_center, "od": od}

class MaterialManager:
    @staticmethod
    def parse_dn(dim_str: str) -> int:
        if not dim_str: return 0
        try:
            match = re.search(r'\d+', str(dim_str))
            if match: return int(match.group())
            return 0
        except: return 0
    @staticmethod
    def generate_mto(df_log: pd.DataFrame) -> pd.DataFrame:
        if df_log.empty: return pd.DataFrame()
        df = df_log.copy()
        df['dn_clean'] = df['dimension'].apply(MaterialManager.parse_dn)
        linear_items = ['Rohrstoß', 'Passstück', 'Rohr']
        df_linear = df[df['bauteil'].isin(linear_items)].copy()
        if not df_linear.empty:
            df_linear['menge'] = pd.to_numeric(df_linear['laenge'], errors='coerce').fillna(0) / 1000.0
            mto_linear = df_linear.groupby(['dn_clean', 'bauteil'])['menge'].sum().reset_index()
            mto_linear['Einheit'] = 'm'
        else:
            mto_linear = pd.DataFrame(columns=['dn_clean', 'bauteil', 'menge', 'Einheit'])
        df_count = df[~df['bauteil'].isin(linear_items)].copy()
        if not df_count.empty:
            mto_count = df_count.groupby(['dn_clean', 'bauteil']).size().reset_index(name='menge')
            mto_count['Einheit'] = 'Stk'
        else:
            mto_count = pd.DataFrame(columns=['dn_clean', 'bauteil', 'menge', 'Einheit'])
        mto_final = pd.concat([mto_linear, mto_count], ignore_index=True)
        mto_final['Dimension'] = mto_final['dn_clean'].apply(lambda x: f"DN {x}")
        mto_final = mto_final.rename(columns={'bauteil': 'Beschreibung', 'menge': 'Menge'})
        mto_final = mto_final[['Dimension', 'Beschreibung', 'Menge', 'Einheit']].sort_values(['Dimension', 'Beschreibung'])
        return mto_final

class HandbookCalculator:
    BOLT_DATA = {"M12": [19, 85, 55], "M16": [24, 210, 135], "M20": [30, 410, 265], "M24": [36, 710, 460], "M27": [41, 1050, 680], "M30": [46, 1420, 920], "M33": [50, 1930, 1250], "M36": [55, 2480, 1600], "M39": [60, 3200, 2080], "M45": [70, 5000, 3250], "M52": [80, 7700, 5000]}
    @staticmethod
    def calculate_weight(od, wall, length):
        if wall <= 0: return {"steel": 0, "water": 0, "total": 0}
        id_mm = od - (2*wall)
        vol_s = (math.pi*(od**2 - id_mm**2)/4)/1000000
        vol_w = (math.pi*(id_mm**2)/4)/1000000
        return {"kg_per_m_steel": vol_s*7850, "total_steel": vol_s*7850*(length/1000), "total_filled": (vol_s*7850 + vol_w*1000)*(length/1000), "volume_l": vol_w*(length/1000)*1000}
    @staticmethod
    def get_bolt_length(t1, t2, bolt, washers=2, gasket=2.0):
        try:
            d = int(bolt.replace("M", ""))
            l = t1 + t2 + gasket + (washers*4) + (d*0.8) + max(6, d*0.4)
            rem = l % 5
            return int(l + (5-rem) if rem != 0 else l)
        except: return 0

class Visualizer:
    @staticmethod
    def plot_stutzen(dn_haupt, dn_stutzen, df_pipe):
        row_h = df_pipe[df_pipe['DN'] == dn_haupt].iloc[0]
        row_s = df_pipe[df_pipe['DN'] == dn_stutzen].iloc[0]
        r_main = row_h['D_Aussen'] / 2; r_stub = row_s['D_Aussen'] / 2
        
        if r_stub > r_main:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "FEHLER: Stutzen > Hauptrohr", ha='center', va='center', color='red', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.close(fig)
            return fig

        angles = range(0, 361, 5); depths = []
        for a in angles:
            try:
                term = r_stub * math.sin(math.radians(a))
                depths.append(r_main - math.sqrt(r_main**2 - term**2))
            except: depths.append(0)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(angles, depths, color='#3b82f6', lw=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360); ax.set_ylabel("Tiefe (mm)"); ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.close(fig) 
        return fig
    @staticmethod
    def plot_2d_offset(run: float, offset: float):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        x = [0, run, run*1.5] 
        y = [0, offset, offset]
        ax.plot([0, run], [0, offset], color='#dc2626', linewidth=3, label='Rohrachse') 
        ax.plot([run, run*1.5], [offset, offset], color='black', linewidth=3) 
        ax.plot([-50, 0], [0, 0], color='black', linewidth=3) 
        ax.plot([0, run], [0, 0], linestyle='--', color='gray', alpha=0.7) 
        ax.plot([run, run], [0, offset], linestyle='--', color='gray', alpha=0.7) 
        ax.text(run/2, -offset*0.1 if offset!=0 else -10, f"Länge: {run:.0f}", ha='center', color='blue')
        ax.text(run + (run*0.05), offset/2, f"H: {offset:.0f}", va='center', color='blue')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        plt.close(fig)
        return fig
    @staticmethod
    def plot_rolling_offset_3d_room(roll: float, run: float, set_val: float):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        P0 = np.array([0, 0, 0])
        P1 = np.array([roll, run, set_val])
        max_dim = max(abs(roll), abs(run), abs(set_val), 100)
        xx, yy = np.meshgrid(np.linspace(-max_dim*0.2, roll*1.2, 2), np.linspace(-max_dim*0.2, run*1.2, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
        ax.plot([0, 0], [-run*0.3, 0], [0, 0], color='gray', linewidth=4, alpha=0.6)
        ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], color='#dc2626', linewidth=5, label='Passstück')
        ax.plot([P1[0], P1[0]], [P1[1], P1[1]+run*0.3], [P1[2], P1[2]], color='gray', linewidth=4, alpha=0.6)
        ax.scatter([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], color='#1e3a8a', s=100, label='Naht/Flansch')
        ax.plot([P1[0], P1[0]], [P1[1], P1[1]], [0, P1[2]], 'b--', linewidth=1, label='Höhe (Set)')
        ax.plot([0, P1[0]], [P1[1], P1[1]], [0, 0], 'g--', linewidth=1, label='Seite (Roll)')
        ax.set_xlabel('Seite (Roll)')
        ax.set_ylabel('Länge (Run)')
        ax.set_zlabel('Höhe (Set)')
        try: 
            ax.set_box_aspect([roll if roll>10 else 100, run if run>10 else 100, set_val if set_val>10 else 100])
        except: 
            pass 
        ax.legend(loc='upper left', fontsize='small')
        plt.close(fig)
        return fig
    @staticmethod
    def plot_rotation_gauge(roll: float, set_val: float, rotation_angle: float):
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
        theta = math.radians(rotation_angle)
        ax.arrow(0, 0, theta, 0.9, head_width=0.1, head_length=0.1, fc='#ef4444', ec='#ef4444', length_includes_head=True)
        ax.set_theta_zero_location("N") 
        ax.set_theta_direction(-1)      
        ax.set_rticks([])               
        ax.set_rlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Verdrehung: {rotation_angle:.1f}°", va='bottom', fontsize=10, fontweight='bold')
        ax.text(math.radians(90), 1.2, "R", ha='center', fontweight='bold')
        ax.text(math.radians(270), 1.2, "L", ha='center', fontweight='bold')
        plt.close(fig)
        return fig
    @staticmethod
    def plot_segment_schematic(mid_back: float, mid_belly: float, od: float, angle: float):
        fig, ax = plt.subplots(figsize=(6, 3))
        height = od
        top_len = mid_back
        bot_len = mid_belly
        x_top = [-top_len/2, top_len/2]
        x_bot = [-bot_len/2, bot_len/2]
        y_top = [height/2, height/2]
        y_bot = [-height/2, -height/2]
        ax.plot(x_top, y_top, 'r-', linewidth=3, label='Rücken')
        ax.plot(x_bot, y_bot, 'b-', linewidth=3, label='Bauch')
        ax.plot([x_top[0], x_bot[0]], [y_top[0], y_bot[0]], 'k--', linewidth=1)
        ax.plot([x_top[1], x_bot[1]], [y_top[1], y_bot[1]], 'k--', linewidth=1)
        ax.annotate(f"{top_len:.1f}", xy=(0, height/2 + height*0.1), ha='center', color='red', fontweight='bold')
        ax.annotate(f"{bot_len:.1f}", xy=(0, -height/2 - height*0.2), ha='center', color='blue', fontweight='bold')
        ax.set_title(f"Mittelstück ({angle:.1f}° Schnitt)", fontsize=10)
        ax.set_xlim(-top_len/2 - 50, top_len/2 + 50)
        ax.set_ylim(-height, height)
        ax.axis('off')
        plt.close(fig)
        return fig

class Exporter:
    @staticmethod
    def clean_text_for_pdf(text: str) -> str:
        if not isinstance(text, str): return str(text)
        replacements = {
            "€": "EUR", "–": "-", "—": "-", "„": '"', "“": '"', "”": '"', "’": "'", "‘": "'"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    @staticmethod
    def to_excel(df):
        output = BytesIO()
        export_df = df.drop(columns=['✏️', 'Löschen', 'id', 'Auswahl', 'project_id', 'dn_clean', 'charge'], errors='ignore')
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Daten')
        return output.getvalue()

    @staticmethod
    def to_pdf_final_report(df_log, project_name, meta_data=None):
        if not PDF_AVAILABLE: return b""
        if meta_data is None: meta_data = {}
        
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, Exporter.clean_text_for_pdf("FERTIGUNGSBESCHEINIGUNG"), 0, 1, 'C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 6, "Rohrleitungsbau / Anlagenbau", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(0, 8, "1. PROJEKTDATEN", 1, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 10)
        
        def row_cell(lbl, val):
            pdf.cell(60, 8, Exporter.clean_text_for_pdf(lbl), 1)
            pdf.cell(0, 8, Exporter.clean_text_for_pdf(str(val)), 1, 1)

        row_cell("Projekt / Baustelle:", project_name)
        row_cell("Auftrags-Nr. / Ticket:", meta_data.get('order_no', '-'))
        row_cell("Anlagenteil / System:", meta_data.get('system_name', '-'))
        row_cell("Datum der Fertigstellung:", datetime.now().strftime('%d.%m.%Y'))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, Exporter.clean_text_for_pdf("2. PRÜFERGEBNISSE & QUALITÄTSSICHERUNG"), 1, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 10)
        
        rt_state = "JA / i.O." if meta_data.get('check_rt') else "Nicht gefordert"
        dim_state = "JA / i.O." if meta_data.get('check_dim') else "Nein"
        iso_state = "JA / i.O." if meta_data.get('check_iso') else "Nein"
        
        row_cell("Zerstörungsfreie Prüfung (RT):", rt_state)
        row_cell("Maßhaltigkeit geprüft:", dim_state)
        row_cell("Isometrie revidiert (As-Built):", iso_state)
        row_cell("Materialzeugnisse (APZ) vorh.:", "Siehe Anlage")
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, Exporter.clean_text_for_pdf("Hiermit wird bestätigt, dass die oben genannten Rohrleitungen fachgerecht nach den geltenden Regeln der Technik und den vorliegenden Isometrien gefertigt wurden. Alle Schweißnähte wurden, soweit gefordert, einer Röntgenprüfung (RT) unterzogen und für in Ordnung befunden."))
        pdf.ln(15)

        y_sig = pdf.get_y()
        pdf.line(10, y_sig, 200, y_sig)
        pdf.ln(2)
        
        col_w = 63
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(col_w, 5, "Ersteller / Fachfirma", 0, 0, 'C')
        pdf.cell(col_w, 5, "Bauleitung / Supervisor", 0, 0, 'C')
        pdf.cell(col_w, 5, "Abnahme / TÜV", 0, 1, 'C')
        
        pdf.ln(15) 
        
        pdf.cell(col_w, 0, "", "B") 
        pdf.cell(col_w, 0, "", "B")
        pdf.cell(col_w, 0, "", "B")
        pdf.ln(2)
        pdf.set_font("Arial", '', 7)
        pdf.cell(col_w, 4, "Datum / Unterschrift", 0, 0, 'C')
        pdf.cell(col_w, 4, "Datum / Unterschrift", 0, 0, 'C')
        pdf.cell(col_w, 4, "Datum / Unterschrift / Stempel", 0, 1, 'C')

        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "ANLAGE 1: Material-Rückverfolgbarkeit", 0, 1, 'L')
        pdf.ln(5)
        
        df_log['charge_apz'] = df_log['charge_apz'].fillna('OHNE NACHWEIS').replace('', 'OHNE NACHWEIS')
        groups = df_log.groupby('charge_apz')
        pdf.set_font("Arial", size=10)
        for apz, group in groups:
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, f"Charge / APZ: {Exporter.clean_text_for_pdf(apz)}", 1, 1, 'L', fill=True)
            pdf.set_font("Arial", size=9)
            agg = group.groupby(['dimension', 'bauteil']).size().reset_index(name='count')
            for _, row in agg.iterrows():
                txt = f"   {row['count']}x {row['bauteil']} {row['dimension']}"
                isos = group[(group['dimension']==row['dimension']) & (group['bauteil']==row['bauteil'])]['iso'].unique()
                iso_txt = ", ".join(isos[:3])
                if len(isos) > 3: iso_txt += "..."
                pdf.cell(90, 6, Exporter.clean_text_for_pdf(txt), 1)
                pdf.cell(0, 6, f"Verbaut in: {Exporter.clean_text_for_pdf(iso_txt)}", 1, 1)
            pdf.ln(2)

        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "ANLAGE 2: Detailliertes Rohrbuch", 0, 1, 'L')
        pdf.ln(5)
        
        cols = ["ISO", "Naht", "DN", "Bauteil", "Schweißer"]
        widths = [40, 30, 30, 60, 30]
        pdf.set_font("Arial", 'B', 9)
        for i, c in enumerate(cols): pdf.cell(widths[i], 8, c, 1)
        pdf.ln()
        
        pdf.set_font("Arial", size=9)
        for _, row in df_log.iterrows():
            vals = [str(row.get(k.lower(), '')) if k.lower() != 'dn' else str(row.get('dimension','')) for k in cols]
            for i, v in enumerate(vals):
                pdf.cell(widths[i], 7, Exporter.clean_text_for_pdf(v[:25]), 1)
            pdf.ln()

        return pdf.output(dest='S').encode('latin-1')

    @staticmethod
    def to_pdf_sawlist(df, project_name="Unbekannt"):
        if not PDF_AVAILABLE: return b""
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Saegeauftrag: {Exporter.clean_text_for_pdf(project_name)}", 0, 1, 'L')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 5, f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}", 0, 1, 'L')
        pdf.ln(5)
        cols = ["Bezeichnung", "Rohmass", "Saegemass", "Info", "Zeit"]
        keys = ["name", "raw_length", "cut_length", "details", "timestamp"]
        widths = [60, 30, 30, 80, 30]
        pdf.set_font("Arial", 'B', 10)
        for i, c in enumerate(cols): pdf.cell(widths[i], 8, c, 1)
        pdf.ln()
        pdf.set_font("Arial", size=10)
        for _, row in df.iterrows():
            for i, k in enumerate(keys):
                val = str(row.get(k, ''))
                if isinstance(row.get(k), float): val = f"{row.get(k):.1f}"
                try: pdf.cell(widths[i], 8, Exporter.clean_text_for_pdf(val), 1)
                except: pdf.cell(widths[i], 8, "?", 1)
            pdf.ln()
        return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 4. UI SEITEN (TABS)
# -----------------------------------------------------------------------------

def init_app_state():
    defaults = {
        'active_project_id': None,
        'active_project_name': "Kein Projekt",
        'project_archived': 0,
        'fitting_list': [],
        'saved_cuts': [],
        'editing_id': None,
        'bulk_edit_ids': [], 
        'last_iso': '',
        'last_naht': '',
        'last_apz': '',
        'last_schweisser': '',
        'last_datum': datetime.now(),
        'form_dn_red_idx': 0,
        'logbook_view_index': 0,
        'active_tab': "🪚 Smarte Säge" 
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_smart_input(label: str, db_column: str, current_value: str, key_prefix: str, active_pid: int) -> str:
    known_values = DatabaseRepository.get_known_values(db_column, active_pid)
    
    if known_values:
        options = ["✨ Neu / Manuell"] + known_values
        try: 
            sel_idx = options.index(current_value)
        except ValueError:
            sel_idx = 0 
            
        selection = st.selectbox(label, options, index=sel_idx, key=f"{key_prefix}_sel")
        
        if selection == "✨ Neu / Manuell":
            final_val = st.text_input(f"{label} (Eingabe)", value=current_value, key=f"{key_prefix}_txt")
        else:
            final_val = selection
    else:
        final_val = st.text_input(label, value=current_value, key=f"{key_prefix}_txt_only")
        
    return final_val

def render_sidebar_projects():
    st.sidebar.title("🏗️ PipeCraft")
    st.sidebar.caption("v3.0 (Native)")
    
    projects = DatabaseRepository.get_projects() 
    
    if st.session_state.active_project_id is None and projects:
        st.session_state.active_project_id = projects[0][0]
        st.session_state.active_project_name = projects[0][1]
        st.session_state.project_archived = projects[0][2]

    current_proj_data = next((p for p in projects if p[0] == st.session_state.active_project_id), None)
    if current_proj_data:
        st.session_state.project_archived = current_proj_data[2]
    
    project_names = []
    for p in projects:
        prefix = "🔒 " if p[2] == 1 else ""
        project_names.append(f"{prefix}{p[1]}")
    
    current_idx = 0
    for i, p in enumerate(projects):
        if p[0] == st.session_state.get('active_project_id'):
            current_idx = i
            break
            
    selected_display = st.sidebar.selectbox("Aktive Baustelle:", project_names, index=current_idx)
    
    sel_index = project_names.index(selected_display)
    new_id = projects[sel_index][0]
    new_name = projects[sel_index][1]
    is_archived = projects[sel_index][2]
    
    if new_id != st.session_state.active_project_id:
        st.session_state.active_project_id = new_id
        st.session_state.active_project_name = new_name
        st.session_state.project_archived = is_archived
        st.session_state.saved_cuts = [] 
        st.session_state.fitting_list = []
        st.rerun()

    if st.session_state.project_archived == 1:
        st.sidebar.warning("🔒 Projekt ist archiviert (Read-Only)")

    with st.sidebar.expander("➕ Neues Projekt"):
        new_proj = st.text_input("Name", placeholder="z.B. Halle 4")
        if st.button("Erstellen"):
            if new_proj:
                ok, msg = DatabaseRepository.create_project(new_proj)
                if ok: 
                    st.success(msg)
                    st.rerun()
                else: 
                    st.error(msg)
    st.sidebar.divider()
    
    with st.sidebar.expander("💾 Datensicherung"):
        if st.session_state.active_project_id:
            json_data = DatabaseRepository.export_project_to_json(st.session_state.active_project_id)
            if json_data:
                fname = f"Backup_{st.session_state.active_project_name.replace(' ', '_')}.json"
                st.download_button("📤 Projekt Exportieren", json_data, fname, "application/json")
        
        uploaded_file = st.file_uploader("📥 Projekt Importieren", type=["json"])
        if uploaded_file is not None:
            if st.button("Import Starten"):
                string_data = uploaded_file.getvalue().decode("utf-8")
                ok, msg = DatabaseRepository.import_project_from_json(string_data)
                if ok:
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

def render_smart_saw(calc: PipeCalculator, df: pd.DataFrame, current_dn: int, pn: str):
    st.markdown('<div class="machine-header-saw">🪚 SMARTE SÄGE</div>', unsafe_allow_html=True)
    
    proj_name = st.session_state.get('active_project_name', 'Unbekannt')
    active_pid = st.session_state.get('active_project_id', 1)
    is_archived = st.session_state.get('project_archived', 0)

    st.markdown(f"<div class='project-tag'>📍 PROJEKT: {proj_name}</div>", unsafe_allow_html=True)

    if is_archived:
        st.info("Projekt ist abgeschlossen. Keine neuen Schnitte möglich.")
        return

    if st.session_state.saved_cuts:
        try: _ = st.session_state.saved_cuts[0].fittings
        except AttributeError: st.session_state.saved_cuts = []

    default_raw = 0.0
    if 'transfer_cut_length' in st.session_state:
        default_raw = st.session_state.pop('transfer_cut_length')
        st.toast("✅ Maß aus Geometrie übernommen!", icon="📏")

    c_calc, c_list = st.columns([1.3, 1.7])

    with c_calc:
        with st.container(border=True):
            st.markdown("**1. Neuer Schnitt**")
            cut_name = st.text_input("Bezeichnung / Spool", placeholder="z.B. Strang A - 01", help="Name für die Liste")
            raw_len = st.number_input("Schnittmaß (Roh) [mm]", value=default_raw, min_value=0.0, step=10.0, format="%.1f")
            
            cg1, cg2, cg3 = st.columns(3)
            gap = cg1.number_input("Spalt (mm)", value=3.0, step=0.5)
            dicht_anz = cg2.number_input("Dichtungen", 0, 5, 0)
            dicht_thk = cg3.number_input("Dicke (mm)", 0.0, 5.0, 2.0, disabled=(dicht_anz==0))
            st.divider()
            st.markdown("**Bauteil-Abzüge (Fittings)**")
            ca1, ca2, ca3, ca4 = st.columns([2, 1.5, 1, 1])
            f_type = ca1.selectbox("Typ", ["Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung"], label_visibility="collapsed")
            f_dn = ca2.selectbox("DN", df['DN'], index=df['DN'].tolist().index(current_dn), label_visibility="collapsed")
            f_cnt = ca3.number_input("Anz.", 1, 10, 1, label_visibility="collapsed")
            f_ang = 90.0
            if "Zuschnitt" in f_type: f_ang = st.slider("Winkel", 0, 90, 45)

            if ca4.button("➕", type="primary"):
                deduct = calc.get_deduction(f_type, f_dn, pn, f_ang)
                uid = f"{len(st.session_state.fitting_list)}_{datetime.now().timestamp()}"
                nm = f"{f_type} DN{f_dn}" + (f" ({f_ang}°)" if "Zuschnitt" in f_type else "")
                st.session_state.fitting_list.append(FittingItem(uid, nm, f_cnt, deduct, f_dn))
                st.rerun()

            if st.session_state.fitting_list:
                st.markdown("###### Aktuelle Abzüge:")
                for i, item in enumerate(st.session_state.fitting_list):
                    cr1, cr2, cr3 = st.columns([3, 1.5, 0.5])
                    cr1.text(f"{item.count}x {item.name}")
                    cr2.text(f"-{item.total_deduction:.1f}")
                    if cr3.button("x", key=f"d_{item.id}"):
                        st.session_state.fitting_list.pop(i)
                        st.rerun()
                if st.button("Reset Abzüge", type="secondary"):
                    st.session_state.fitting_list = []
                    st.rerun()

        sum_fit = sum(i.total_deduction for i in st.session_state.fitting_list)
        sum_gap = sum(i.count for i in st.session_state.fitting_list) * gap
        sum_gskt = dicht_anz * dicht_thk
        total = sum_fit + sum_gap + sum_gskt
        final = raw_len - total

        if final < 0: 
            st.error(f"Negativmaß! ({final:.1f} mm)")
        else:
            st.metric("Sägelänge", f"{final:.1f} mm", delta=None)
            st.caption(f"Abzüge: Teile -{sum_fit:.1f} | Spalte -{sum_gap:.1f} | Dicht. -{sum_gskt:.1f}")
            if st.button("💾 SPEICHERN", type="primary", use_container_width=True):
                if raw_len > 0:
                    final_name = cut_name if cut_name.strip() else f"Schnitt"
                    current_fittings_copy = list(st.session_state.fitting_list)
                    new_id = int(time.time() * 1000) 
                    new_cut = SavedCut(new_id, final_name, raw_len, final, f"{len(current_fittings_copy)} Teile", datetime.now().strftime("%H:%M"), current_fittings_copy)
                    st.session_state.saved_cuts.append(new_cut)
                    st.session_state.fitting_list = [] 
                    st.toast("✅ Schnitt gespeichert!", icon="💾")
                    time.sleep(0.5)
                    st.rerun()

    with c_list:
        st.markdown("#### 📋 Schnittliste")
        action_bar = st.container()

        if not st.session_state.saved_cuts:
            st.info("Noch keine Schnitte vorhanden.")
            with action_bar:
                st.button("🗑️ Löschen", disabled=True, use_container_width=True)
        else:
            data = [asdict(c) for c in st.session_state.saved_cuts]
            df_s = pd.DataFrame(data)
            if 'Auswahl' not in df_s.columns:
                df_s['Auswahl'] = False
            
            # DF für Editor
            df_display = df_s[['Auswahl', 'name', 'raw_length', 'cut_length', 'details', 'id']]
            
            # --- Native Editor statt AgGrid ---
            edited_df = st.data_editor(
                df_display, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Auswahl": st.column_config.CheckboxColumn("☑️", width="small", default=False),
                    "name": st.column_config.TextColumn("Bez.", width="medium"), 
                    "raw_length": st.column_config.NumberColumn("Roh", format="%.0f"), 
                    "cut_length": st.column_config.NumberColumn("Säge", format="%.1f", width="medium"), 
                    "details": st.column_config.TextColumn("Info", width="small"), 
                    "id": None
                },
                disabled=["name", "raw_length", "cut_length", "details", "id"], 
                key="saw_editor_v4"
            )
            
            # IDs holen
            selected_rows = edited_df[edited_df['Auswahl'] == True]
            selected_ids = selected_rows['id'].tolist()
            num_sel = len(selected_ids)
            
            with action_bar:
                btns_disabled = (num_sel == 0)
                col_del, col_trans, col_excel = st.columns([1, 1, 1])
                
                if col_del.button(f"🗑️ Löschen ({num_sel})", disabled=btns_disabled, type="secondary", use_container_width=True):
                    st.session_state.saved_cuts = [c for c in st.session_state.saved_cuts if c.id not in selected_ids]
                    st.toast(f"🗑️ {num_sel} Einträge gelöscht!", icon="🗑️")
                    time.sleep(0.5)
                    st.rerun()
                
                if col_trans.button(f"📝 Übertragen ({num_sel})", disabled=btns_disabled, type="primary", use_container_width=True, help="Überträgt Rohr + Anbauteile ins Rohrbuch"):
                    count_pipes = 0
                    count_fits = 0
                    for cut in st.session_state.saved_cuts:
                        if cut.id in selected_ids:
                            DatabaseRepository.add_entry({
                                "iso": cut.name, "naht": "", "datum": datetime.now().strftime("%d.%m.%Y"),
                                "dimension": f"DN {current_dn}", "bauteil": "Rohrstoß", "laenge": cut.cut_length,
                                "charge": "", "charge_apz": "", "schweisser": "", "project_id": active_pid
                            })
                            count_pipes += 1
                            for fit in cut.fittings:
                                fit_name_clean = fit.name.split(" DN")[0]
                                for _ in range(fit.count):
                                    DatabaseRepository.add_entry({
                                        "iso": cut.name, "naht": "", "datum": datetime.now().strftime("%d.%m.%Y"),
                                        "dimension": f"DN {fit.dn}", "bauteil": fit_name_clean, "laenge": 0.0,
                                        "charge": "", "charge_apz": "", "schweisser": "", "project_id": active_pid
                                    })
                                    count_fits += 1
                    
                    st.toast(f"✅ {count_pipes} Rohre und {count_fits} Fittings übertragen!", icon="📦")
                    time.sleep(0.5)

                fname_base = f"Saege_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
                excel_data = Exporter.to_excel(df_s)
                col_excel.download_button("📥 Excel (Alle)", excel_data, f"{fname_base}.xlsx", use_container_width=True)

            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            if st.button("Alles Reset (Liste leeren)", type="secondary"):
                st.session_state.saved_cuts = []
                st.rerun()

def render_geometry_tools(calc: PipeCalculator, df: pd.DataFrame):
    st.markdown('<div class="machine-header-geo">📐 GEOMETRIE & BERECHNUNG</div>', unsafe_allow_html=True)
    geo_tabs = st.tabs(["2D Etage (S-Schlag)", "3D Raum-Etage (Rolling)", "Bogen (Standard)", "🦞 Segment-Bogen", "Stutzen"])
    
    with geo_tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            with st.container(border=True):
                dn = st.selectbox("Nennweite", df['DN'], index=5, key="2d_dn")
                offset = st.number_input("Versprung (H) [mm]", value=500.0, step=10.0, key="2d_off")
                angle = st.selectbox("Fittings (°)", [30, 45, 60], index=1, key="2d_ang")
                if st.button("Berechnen 2D", type="primary", use_container_width=True):
                    res = calc.calculate_2d_offset(dn, offset, angle)
                    st.session_state.calc_res_2d = res 
        
        with c2:
            if 'calc_res_2d' in st.session_state:
                res = st.session_state.calc_res_2d
                if "error" in res: st.error(res["error"])
                else:
                    st.markdown("**Ergebnis**")
                    m1, m2 = st.columns(2)
                    m1.metric("Zuschnitt (Rohr)", f"{res['cut_length']:.1f} mm")
                    m2.metric("Etagenlänge", f"{res['hypotenuse']:.1f} mm")
                    st.info(f"Benötigter Platz (Länge): {res['run']:.1f} mm")
                    
                    if st.button("➡️ An Säge (2D)", key="btn_2d_saw"):
                        st.session_state.transfer_cut_length = res['cut_length']
                        st.toast("✅ Maß an Säge übertragen!", icon="📏")
                    
                    fig_2d = Visualizer.plot_2d_offset(res['run'], res['offset'])
                    st.pyplot(fig_2d, use_container_width=False)

    with geo_tabs[1]:
        col_in, col_out, col_vis = st.columns([1, 1, 1.5]) 
        with col_in:
            with st.container(border=True):
                st.markdown("**Eingabe**")
                dn_roll = st.selectbox("Nennweite", df['DN'], index=5, key="3d_dn")
                fit_angle = st.selectbox("Fitting Typ", [45, 60, 90], index=0, key="3d_ang")
                set_val = st.number_input("Versprung Höhe (Set)", value=300.0, min_value=0.0, step=10.0)
                roll_val = st.number_input("Versprung Seite (Roll)", value=400.0, min_value=0.0, step=10.0)
                
                true_offset = math.sqrt(set_val**2 + roll_val**2)
                rad_angle = math.radians(fit_angle)
                if rad_angle > 0:
                    travel_center = true_offset / math.sin(rad_angle)
                    run_length = true_offset / math.tan(rad_angle)
                else:
                    travel_center = 0; run_length = 0
                deduct_single = calc.get_deduction(f"Bogen (Zuschnitt)", dn_roll, "PN 16", fit_angle) 
                cut_len = travel_center - (2 * deduct_single)
                if set_val == 0 and roll_val == 0: rot_angle = 0.0
                elif set_val == 0: rot_angle = 90.0
                else: rot_angle = math.degrees(math.atan(roll_val / set_val))

        with col_out:
            st.markdown("**Ergebnis**")
            st.metric("Zuschnitt (Rohr)", f"{cut_len:.1f} mm")
            st.caption(f"Einbaumaß (Mitte-Mitte): {travel_center:.1f} mm")
            st.metric("Baulänge (Run)", f"{run_length:.1f} mm", help="Platzbedarf in Längsrichtung")
            st.metric("Verdrehung", f"{rot_angle:.1f} °", "aus der Senkrechten")
            if cut_len < 0: st.error("Nicht baubar! Fittings stoßen zusammen.")
            else:
                if st.button("➡️ An Säge (3D)", key="btn_3d_saw"):
                    st.session_state.transfer_cut_length = cut_len
                    st.toast("✅ Maß an Säge übertragen!", icon="📏")

        with col_vis:
            st.markdown("**3D Simulation**")
            fig_3d = Visualizer.plot_rolling_offset_3d_room(roll_val, run_length, set_val)
            st.pyplot(fig_3d, use_container_width=False)
            with st.expander("Verdrehung (Wasserwaage)"):
                fig_gauge = Visualizer.plot_rotation_gauge(roll_val, set_val, rot_angle)
                st.pyplot(fig_gauge, use_container_width=False)

    with geo_tabs[2]:
        with st.container(border=True):
            st.markdown("#### Standard Bogen-Rechner")
            cb1, cb2 = st.columns(2)
            angle = cb1.slider("Winkel", 0, 90, 45, key="gb_ang_std")
            dn_b = cb2.selectbox("DN", df['DN'], index=6, key="gb_dn_std")
            details = calc.calculate_bend_details(dn_b, angle)
        
        c_v, c_l = st.columns([1, 2])
        with c_v: st.metric("Vorbau (Z-Maß)", f"{details['vorbau']:.1f} mm")
        with c_l:
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Rücken", f"{details['bogen_aussen']:.1f}")
            cm2.metric("Mitte", f"{details['bogen_mitte']:.1f}") 
            cm3.metric("Bauch", f"{details['bogen_innen']:.1f}")

    with geo_tabs[3]:
        st.info("🦞 Berechnung für Segmentbögen (Lobster Back) ohne Standard-Fittings.")
        c1, c2, c3 = st.columns(3)
        with c1:
            dn_seg = st.selectbox("Nennweite", df['DN'], index=8, key="seg_dn")
            radius_seg = st.number_input("Radius (R) [mm]", value=1000.0, step=10.0)
        with c2:
            num_seg = st.number_input("Anzahl Segmente (Ganze)", value=3, min_value=2, step=1)
            total_ang = st.number_input("Gesamtwinkel", value=90.0, step=5.0)
        with c3:
            if st.button("Berechnen 🦞", type="primary"):
                res = calc.calculate_segment_bend(dn_seg, radius_seg, num_seg, total_ang)
                st.session_state.calc_res_seg = res
        
        if 'calc_res_seg' in st.session_state:
            res = st.session_state.calc_res_seg
            if "error" in res:
                st.error(res["error"])
            else:
                st.divider()
                with st.container(border=True):
                    c_res1, c_res2 = st.columns([1, 1])
                    with c_res1:
                        st.markdown("#### Mittelstück (Ganz)")
                        st.metric("Rücken (L_aussen)", f"{res['mid_back']:.1f} mm")
                        st.metric("Bauch (L_innen)", f"{res['mid_belly']:.1f} mm")
                        st.caption(f"Schnittwinkel: {res['miter_angle']:.2f}°")
                    with c_res2:
                        st.markdown("#### Endstück (Halb)")
                        st.metric("Rücken bis Schnitt", f"{res['end_back']:.1f} mm")
                        st.metric("Bauch bis Schnitt", f"{res['end_belly']:.1f} mm")
                fig_seg = Visualizer.plot_segment_schematic(res['mid_back'], res['mid_belly'], res['od'], res['miter_angle'])
                st.pyplot(fig_seg, use_container_width=False)

    with geo_tabs[4]:
        c1, c2 = st.columns(2)
        dn_stub = c1.selectbox("DN Stutzen", df['DN'], index=5, key="gs_dn1")
        dn_main = c2.selectbox("DN Hauptrohr", df['DN'], index=8, key="gs_dn2")
        if c1.button("Berechnen Stutzen"):
            try:
                df_c = calc.calculate_stutzen_coords(dn_main, dn_stub)
                fig = Visualizer.plot_stutzen(dn_main, dn_stub, df)
                c_r1, c_r2 = st.columns([1, 2])
                with c_r1: st.table(df_c)
                # FIX #2: Check ob Plot existiert (verhindert Absturz bei None)
                with c_r2: 
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error("⚠️ Geometrie ungültig")
            except ValueError as e: st.error(str(e))

def render_mto_tab(active_pid: int, proj_name: str):
    st.markdown('<div class="machine-header-doc">📦 MATERIAL MANAGER</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='project-tag'>📍 PROJEKT: {proj_name}</div>", unsafe_allow_html=True)
    df_log = DatabaseRepository.get_logbook_by_project(active_pid)
    if df_log.empty:
        st.info("Keine Daten im Rohrbuch. Das Materiallager ist leer.")
        return
    mto_df = MaterialManager.generate_mto(df_log)
    if not mto_df.empty:
        with st.container(border=True):
            total_items = len(mto_df)
            total_meters = mto_df[mto_df['Einheit']=='m']['Menge'].sum()
            c1, c2 = st.columns(2)
            c1.metric("Positionen", total_items, "verschiedene Artikel")
            c2.metric("Verrohrung gesamt", f"{total_meters:.1f} m", "Laufmeter")
        
        st.divider()
        fname = f"MTO_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            mto_df.to_excel(writer, index=False, sheet_name='Materialauszug')
        st.download_button("📥 MTO als Excel herunterladen", output.getvalue(), fname, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.dataframe(mto_df, use_container_width=True, hide_index=True)

def render_logbook(df_pipe: pd.DataFrame):
    st.markdown('<div class="machine-header-doc">📝 ROHRBUCH</div>', unsafe_allow_html=True)
    
    proj_name = st.session_state.get('active_project_name', 'Unbekannt')
    active_pid = st.session_state.get('active_project_id', 1)
    is_archived = st.session_state.get('project_archived', 0)

    st.markdown(f"<div class='project-tag'>📍 PROJEKT: {proj_name} (ID: {active_pid})</div>", unsafe_allow_html=True)

    bulk_ids = st.session_state.get('bulk_edit_ids', [])
    
    if not is_archived:
        # --- MASSEN-BEARBEITUNG BLOCK ---
        if len(bulk_ids) > 1:
            st.warning(f"⚡ MASSEN-BEARBEITUNG: {len(bulk_ids)} Einträge ausgewählt")
            with st.container(border=True):
                c_bulk1, c_bulk2, c_bulk3 = st.columns([1, 2, 1])
                target_field = c_bulk1.selectbox("Feld ändern:", ["Schweißer", "APZ / Charge", "ISO", "Datum"])
                
                new_value = ""
                if target_field == "Datum":
                    d_val = c_bulk2.date_input("Neues Datum", datetime.now())
                    new_value = d_val.strftime("%d.%m.%Y")
                elif target_field == "Schweißer":
                    new_value = render_smart_input("Neuer Schweißer", "schweisser", "", "bulk_sch", active_pid)
                elif target_field == "APZ / Charge":
                    new_value = render_smart_input("Neue APZ / Charge", "charge_apz", "", "bulk_apz", active_pid)
                elif target_field == "ISO":
                    new_value = render_smart_input("Neue ISO", "iso", "", "bulk_iso", active_pid)
                
                if c_bulk3.button("🚀 Alle ändern", type="primary"):
                    DatabaseRepository.bulk_update(bulk_ids, target_field, new_value)
                    st.toast(f"🚀 {len(bulk_ids)} Einträge aktualisiert!", icon="✅")
                    time.sleep(0.5)
                    st.session_state.bulk_edit_ids = []
                    st.rerun()
                
                if st.button("Abbrechen (Auswahl aufheben)"):
                    st.session_state.bulk_edit_ids = []
                    st.rerun()

        # --- EINZEL-BEARBEITUNG BLOCK ---
        else:
            if len(bulk_ids) == 1:
                if st.session_state.editing_id != bulk_ids[0]:
                    st.session_state.editing_id = bulk_ids[0]
            
            header_text = "Eintrag bearbeiten ✏️" if st.session_state.editing_id else "Neuer Eintrag ➕"
            
            with st.container(border=True):
                st.markdown(f"#### {header_text}")
                
                # Logic to determine default values...
                def_iso = st.session_state.last_iso if not st.session_state.editing_id else ""
                def_sch = st.session_state.last_schweisser if not st.session_state.editing_id else ""
                def_apz = st.session_state.last_apz if not st.session_state.editing_id else ""
                def_dat = st.session_state.last_datum if not st.session_state.editing_id else datetime.now()

                c1, c2, c3 = st.columns(3)
                
                current_iso = st.session_state.form_iso if st.session_state.editing_id else def_iso
                iso_val = render_smart_input("ISO / Bez.", "iso", current_iso, "main_iso", active_pid)

                if 'form_naht' not in st.session_state: st.session_state.form_naht = ""
                naht_val = c2.text_input("Naht", value=st.session_state.form_naht, key="inp_naht")
                
                if 'form_datum' not in st.session_state: st.session_state.form_datum = def_dat
                if isinstance(st.session_state.form_datum, str):
                      try: st.session_state.form_datum = datetime.strptime(st.session_state.form_datum, "%d.%m.%Y").date()
                      except: st.session_state.form_datum = datetime.now().date()
                dat_val = c3.date_input("Datum", value=st.session_state.form_datum, key="inp_dat")
                
                c4, c5, c6 = st.columns(3)
                
                if 'form_bauteil_idx' not in st.session_state: st.session_state.form_bauteil_idx = 0
                bt_idx = st.session_state.form_bauteil_idx
                bt_options = ["Rohrstoß", "Bogen", "Flansch", "T-Stück", "Reduzierung", "Stutzen", "Passstück", "Nippel", "Muffe"]
                if bt_idx >= len(bt_options): bt_idx = 0
                bt_val = c4.selectbox("Bauteil", bt_options, index=bt_idx, key="inp_bt")
                
                if 'form_dn_idx' not in st.session_state: st.session_state.form_dn_idx = 8
                dn_idx = st.session_state.form_dn_idx
                if dn_idx >= len(df_pipe): dn_idx = 8
                dn_val = c5.selectbox("Dimension", df_pipe['DN'], index=dn_idx, key="inp_dn")
                
                if bt_val == "Reduzierung":
                    if 'form_dn_red_idx' not in st.session_state: st.session_state.form_dn_red_idx = 0
                    r_idx = st.session_state.form_dn_red_idx
                    if r_idx >= len(df_pipe): r_idx = 0
                    dn_red_val = c5.selectbox("Auf DN", df_pipe['DN'], index=r_idx, key="inp_dn_red")
                    final_dim_str = f"DN {dn_val} x DN {dn_red_val}"
                else:
                    final_dim_str = f"DN {dn_val}"

                if 'form_len' not in st.session_state: st.session_state.form_len = 0.0
                len_val = c6.number_input("Länge (mm)", value=float(st.session_state.form_len), step=1.0, key="inp_len") 
                
                c7, c8 = st.columns(2)
                
                current_apz = st.session_state.form_apz if st.session_state.editing_id else def_apz
                apz_val = render_smart_input("APZ / Zeugnis", "charge_apz", current_apz, "main_apz", active_pid)
                
                current_sch = st.session_state.form_schweisser if st.session_state.editing_id else def_sch
                sch_val = render_smart_input("Schweißer", "schweisser", current_sch, "main_sch", active_pid)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.session_state.editing_id:
                    col_save, col_cancel = st.columns([1, 1])
                    if col_save.button("🔄 ÄNDERUNG ÜBERNEHMEN", type="primary", use_container_width=True):
                        DatabaseRepository.update_full_entry(st.session_state.editing_id, {
                            "iso": iso_val, "naht": naht_val, "datum": dat_val.strftime("%d.%m.%Y"),
                            "dimension": final_dim_str, "bauteil": bt_val, "laenge": len_val,
                            "charge_apz": apz_val, "schweisser": sch_val
                        })
                        st.toast("✅ Eintrag aktualisiert!", icon="✏️")
                        time.sleep(0.5)
                        st.session_state.editing_id = None
                        st.session_state.bulk_edit_ids = []
                        st.session_state.form_iso = ""
                        st.session_state.form_naht = ""
                        st.session_state.form_len = 0.0
                        st.rerun()
                        
                    if col_cancel.button("Abbrechen", use_container_width=True):
                        st.session_state.editing_id = None
                        st.session_state.bulk_edit_ids = []
                        st.session_state.form_iso = "" 
                        st.rerun()
                else:
                    if st.button("SPEICHERN 💾", type="primary", use_container_width=True):
                        DatabaseRepository.add_entry({
                            "iso": iso_val, "naht": naht_val, "datum": dat_val.strftime("%d.%m.%Y"),
                            "dimension": final_dim_str, "bauteil": bt_val, "laenge": len_val,
                            "charge": "", "charge_apz": apz_val, "schweisser": sch_val,
                            "project_id": active_pid
                        })
                        st.session_state.last_iso = iso_val
                        st.session_state.last_apz = apz_val
                        st.session_state.last_schweisser = sch_val
                        st.session_state.last_datum = dat_val
                        st.session_state.form_naht = "" 
                        st.session_state.form_len = 0.0
                        st.toast("✅ Gespeichert!", icon="💾")
                        time.sleep(0.5)
                        st.rerun()

    st.divider()
    
    df = DatabaseRepository.get_logbook_by_project(active_pid)
    
    if not df.empty:
        ce1, ce2, _ = st.columns([1,1,3])
        fname_base = f"Rohrbuch_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        ce1.download_button("📥 Excel", Exporter.to_excel(df), f"{fname_base}.xlsx")
        
        st.markdown("### 📋 Einträge")
        
        # --- NATIVE TABLE (REPLACEMENT FOR AGGRID) ---
        # Vorbereitung: Auswahl-Spalte hinzufügen (Checkbox)
        df_display = df.copy()
        df_display.insert(0, "Auswahl", False)
        
        # Spaltenkonfiguration
        edited_df = st.data_editor(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Auswahl": st.column_config.CheckboxColumn("☑️", width="small", default=False),
                "id": None, # ID verstecken
                "project_id": None,
                "✏️": None,
                "Löschen": None,
                "iso": st.column_config.TextColumn("ISO", width="medium"),
                "naht": st.column_config.TextColumn("Naht", width="small"),
                "datum": st.column_config.TextColumn("Datum", width="small"),
                "bauteil": st.column_config.TextColumn("Bauteil", width="medium"),
                "dimension": st.column_config.TextColumn("Dimension", width="small"),
                "schweisser": st.column_config.TextColumn("Schweißer", width="small"),
                "charge_apz": st.column_config.TextColumn("APZ/Charge", width="medium"),
            },
            disabled=["iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser", "id", "project_id"],
            key="logbook_editor_native"
        )
        
        # Logik: Welche IDs sind ausgewählt?
        selected_rows = edited_df[edited_df['Auswahl'] == True]
        selected_ids_list = selected_rows['id'].tolist()
        
        # State Update Logic
        current_bulk_set = set(st.session_state.bulk_edit_ids)
        new_bulk_set = set(selected_ids_list)
        
        if current_bulk_set != new_bulk_set:
            st.session_state.bulk_edit_ids = selected_ids_list
            
            # Wenn genau EINE Zeile ausgewählt ist -> ins Formular laden
            if len(selected_ids_list) == 1:
                sel_row = selected_rows.iloc[0].to_dict()
                
                st.session_state.editing_id = int(sel_row['id'])
                st.session_state.form_iso = sel_row.get('iso', '')
                st.session_state.form_naht = sel_row.get('naht', '')
                st.session_state.form_apz = sel_row.get('charge_apz', '')
                st.session_state.form_schweisser = sel_row.get('schweisser', '')
                l_val = sel_row.get('laenge', 0.0)
                st.session_state.form_len = float(l_val) if l_val else 0.0
                try: 
                    d_str = sel_row.get('datum', datetime.now().strftime("%d.%m.%Y"))
                    st.session_state.form_datum = datetime.strptime(d_str, "%d.%m.%Y").date()
                except: st.session_state.form_datum = datetime.now().date()
                
                bt_options = ["Rohrstoß", "Bogen", "Flansch", "T-Stück", "Reduzierung", "Stutzen", "Passstück", "Nippel", "Muffe"]
                try: st.session_state.form_bauteil_idx = bt_options.index(sel_row.get('bauteil', 'Rohrstoß'))
                except: st.session_state.form_bauteil_idx = 0
                
                dim_str = str(sel_row.get('dimension', ''))
                all_dns = re.findall(r'\d+', dim_str)
                if len(all_dns) > 0:
                    dn_int = int(all_dns[0])
                    match = df_pipe[df_pipe['DN'] == dn_int]
                    if not match.empty: st.session_state.form_dn_idx = int(match.index[0])

            # Wenn KEINE oder MEHRERE -> Formular leeren / Reset Edit Mode
            if len(selected_ids_list) != 1:
                st.session_state.editing_id = None
                
            st.rerun()

        # Lösch-Button für Auswahl (nur sichtbar wenn Auswahl > 1, bei 1 haben wir das Formular)
        if len(selected_ids_list) > 1:
             if st.button(f"🗑️ {len(selected_ids_list)} Einträge löschen", type="secondary"):
                DatabaseRepository.delete_entries(selected_ids_list)
                st.session_state.editing_id = None
                st.session_state.bulk_edit_ids = []
                st.toast(f"🗑️ {len(selected_ids_list)} Einträge gelöscht!")
                time.sleep(0.5)
                st.rerun()

    else:
        st.info(f"Keine Einträge für Projekt '{proj_name}'.")

def render_tab_handbook(calc: PipeCalculator, dn: int, pn: str):
    st.markdown('<div class="machine-header-doc">📚 SMART DATA</div>', unsafe_allow_html=True)
    row = calc.get_row(dn)
    suffix = "_16" if pn == "PN 16" else "_10"
    st.markdown(f"**DN {dn} / {pn}**")

    od = float(row['D_Aussen'])
    flange_b = float(row[f'Flansch_b{suffix}'])
    lk = float(row[f'LK_k{suffix}'])
    bolt = row[f'Schraube_M{suffix}']
    n_holes = int(row[f'Lochzahl{suffix}'])
    
    with st.container(border=True):
        st.markdown("##### 🏗️ Gewichte & Hydrotest")
        c_in1, c_in2 = st.columns([1, 2])
        with c_in1:
            wt_input = st.number_input("Wandstärke (mm)", value=6.3, min_value=1.0, step=0.1)
            len_input = st.number_input("Rohrlänge (m)", value=6.0, step=0.5)
        with c_in2:
            w_data = HandbookCalculator.calculate_weight(od, wt_input, len_input * 1000)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Leergewicht (Stahl)", f"{w_data['total_steel']:.1f} kg", f"{w_data['kg_per_m_steel']:.1f} kg/m")
            mc2.metric("Gewicht Gefüllt", f"{w_data['total_filled']:.1f} kg", "für Hydrotest")
            mc3.metric("Füllvolumen", f"{w_data['volume_l']:.0f} Liter", "Wasserbedarf")

    c_geo1, c_geo2 = st.columns(2)
    with c_geo1:
        with st.container(border=True):
            st.markdown("##### 📐 Flansch")
            st.write(f"**Blatt:** {flange_b} mm | **Lochkreis:** {lk} mm")
            st.write(f"**Bohrung:** {n_holes} x {bolt}")
            progress_val = min(lk / (od + 100), 1.0)
            st.progress(progress_val, text="Lochkreis Verhältnis")

    with c_geo2:
        with st.container(border=True):
            st.markdown("##### 🔘 Dichtung (Check)")
            d_innen = od - (2*wt_input) 
            d_aussen = lk - (int(bolt.replace("M","")) * 1.5)
            st.info(f"ID: ~{d_innen:.0f} mm | AD: ~{d_aussen:.0f} mm | 2.0mm")

    st.divider()
    
    with st.container(border=True):
        st.markdown("#### 🔧 Montage & Drehmomente (8.8)")
        
        cb_col1, cb_col2 = st.columns([1, 2.5])
        
        with cb_col1:
            st.caption("Konfiguration")
            conn_type = st.radio("Typ", ["Fest-Fest", "Fest-Los", "Fest-Blind"], index=0, label_visibility="collapsed")
            use_washers = st.checkbox("2x U-Scheibe", value=True)
            is_lubed = st.toggle("Geschmiert (MoS2)", value=True)
            gasket_thk = st.number_input("Dichtung", value=2.0, step=0.5)
            
        with cb_col2:
            bolt_info = HandbookCalculator.BOLT_DATA.get(bolt, [0, 0, 0])
            sw, nm_dry, nm_lube = bolt_info
            
            t1 = flange_b
            t2 = flange_b 
            if "Los" in conn_type: t2 = flange_b + 5 
            elif "Blind" in conn_type: t2 = flange_b + (dn * 0.02)
                
            n_washers = 2 if use_washers else 0
            calc_len = HandbookCalculator.get_bolt_length(t1, t2, bolt, n_washers, gasket_thk)
            torque = nm_lube if is_lubed else nm_dry
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Bolzen", f"{bolt} x {calc_len}", f"{n_holes} Stk.")
            m2.metric("Schlüsselweite", f"SW {sw} mm", "Nuss/Ring")
            m3.metric("Drehmoment", f"{torque} Nm", "Geschmiert" if is_lubed else "Trocken")

# -----------------------------------------------------------------------------
# 5. ABSCHLUSS & ARCHIV (TÜV-READY)
# -----------------------------------------------------------------------------

def render_closeout_tab(active_pid: int, proj_name: str, is_archived: int):
    st.markdown('<div class="machine-header-doc">🏁 FERTIGSTELLUNG (HANDOVER)</div>', unsafe_allow_html=True)
    
    if is_archived:
        st.warning(f"Projekt '{proj_name}' ist abgeschlossen und archiviert.")
        if st.button("🔓 Projekt wiedereröffnen (Reopen)"):
            DatabaseRepository.toggle_archive_project(active_pid, False)
            st.session_state.project_archived = 0
            st.rerun()
            
        df_log = DatabaseRepository.get_logbook_by_project(active_pid)
        if not df_log.empty and PDF_AVAILABLE:
            st.divider()
            st.markdown("#### Dokumentation (Abruf)")
            pdf_data = Exporter.to_pdf_final_report(df_log, proj_name, {"order_no": "Archiv", "system_name": "Archiv", "check_rt": True})
            st.download_button("📄 Fertigungsbescheinigung herunterladen", pdf_data, f"Fertigungsbescheinigung_{proj_name}.pdf", "application/pdf", type="primary")
        return

    st.info("Erstellung der Fertigungsbescheinigung für die Abnahme.")
    
    df_log = DatabaseRepository.get_logbook_by_project(active_pid)
    
    with st.container(border=True):
        st.markdown("#### 1. Projektdaten für Deckblatt")
        c1, c2 = st.columns(2)
        in_order = c1.text_input("Auftrags-Nr. / Ticket", placeholder="z.B. A-2024-55")
        in_sys = c2.text_input("Anlagenteil / System", placeholder="z.B. Kältewasser VL")
        
        st.markdown("#### 2. Qualitätssicherung (Bestätigung)")
        c_rt, c_dim, c_iso = st.columns(3)
        check_rt = c_rt.checkbox("ZfP: RT (Röntgen) durchgeführt & i.O.", value=True)
        check_dim = c_dim.checkbox("Maßhaltigkeit geprüft")
        check_iso = c_iso.checkbox("Isometrie revidiert (As-Built)")

    missing_apz = len(df_log[df_log['charge_apz'].astype(str).str.strip() == ''])
    missing_weld = len(df_log[df_log['schweisser'].astype(str).str.strip() == ''])
    
    ready = True
    if missing_apz > 0 or missing_weld > 0:
        st.warning(f"Hinweis: Es fehlen {missing_apz} APZs und {missing_weld} Schweißer-Einträge.")
        ready = False 

    st.divider()
    
    col_act, col_pdf = st.columns(2)
    
    meta_data = {
        "order_no": in_order,
        "system_name": in_sys,
        "check_rt": check_rt,
        "check_dim": check_dim,
        "check_iso": check_iso
    }

    with col_act:
        force_close = False
        if not ready:
            force_close = st.checkbox("⚠️ Trotz fehlender Daten abschließen")
        
        if ready or force_close:
            if st.button("🏁 BESCHEINIGUNG ERSTELLEN & ARCHIVIEREN", type="primary"):
                DatabaseRepository.toggle_archive_project(active_pid, True)
                st.session_state.project_archived = 1
                st.balloons()
                st.rerun()
        else:
            st.button("🏁 Abschließen", disabled=True)

    with col_pdf:
        if not df_log.empty and PDF_AVAILABLE:
            pdf_data = Exporter.to_pdf_final_report(df_log, proj_name, meta_data)
            st.download_button("📄 Vorschau: Fertigungsbescheinigung", pdf_data, f"Vorschau_Bescheinigung_{proj_name}.pdf", "application/pdf")

# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------

def main():
    init_app_state()
    DatabaseRepository.init_db()
    df_pipe = get_pipe_data()
    calc = PipeCalculator(df_pipe)

    render_sidebar_projects()

    with st.sidebar:
        st.subheader("⚙️ Setup")
        dn = st.selectbox("Nennweite", df_pipe['DN'], index=8)
        pn = st.radio("Druckstufe", ["PN 16", "PN 10"], horizontal=True)

    tabs = ["🪚 Smarte Säge", "📐 Geometrie", "📝 Rohrbuch", "📦 Material", "📚 Smart Data", "🏁 Handover"]
    
    if st.session_state.active_tab not in tabs:
        st.session_state.active_tab = tabs[0]
    
    selected_tab = st.radio("Menü", tabs, horizontal=True, label_visibility="collapsed", key="nav_radio", index=tabs.index(st.session_state.active_tab))
    
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
        st.rerun()

    if st.session_state.active_tab == "🪚 Smarte Säge":
        render_smart_saw(calc, df_pipe, dn, pn)
    elif st.session_state.active_tab == "📐 Geometrie":
        render_geometry_tools(calc, df_pipe)
    elif st.session_state.active_tab == "📝 Rohrbuch":
        render_logbook(df_pipe)
    elif st.session_state.active_tab == "📦 Material":
        render_mto_tab(st.session_state.active_project_id, st.session_state.active_project_name)
    elif st.session_state.active_tab == "📚 Smart Data":
        render_tab_handbook(calc, dn, pn)
    elif st.session_state.active_tab == "🏁 Handover":
        render_closeout_tab(st.session_state.active_project_id, st.session_state.active_project_name, st.session_state.project_archived)

if __name__ == "__main__":
    main()
