import streamlit as st
import pandas as pd
import math
import sqlite3
import logging
import re
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

# FPDF optional laden
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. KONFIGURATION & SETUP
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipeCraft_Pro_V7_9")

st.set_page_config(
    page_title="Rohrbau Profi 7.9",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e293b; font-family: 'Segoe UI', sans-serif; }
    .project-tag {
        background-color: #0ea5e9; color: white;
        padding: 6px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.9em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px; display: inline-block;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATEN-SCHICHT
# -----------------------------------------------------------------------------

@st.cache_data
def get_pipe_data() -> pd.DataFrame:
    raw_data = {
        'DN':            [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600],
        'D_Aussen':      [33.7, 42.4, 48.3, 60.3, 76.1, 88.9, 114.3, 139.7, 168.3, 219.1, 273.0, 323.9, 355.6, 406.4, 457.0, 508.0, 610.0, 711.0, 813.0, 914.0, 1016.0, 1219.0, 1422.0, 1626.0],
        'Radius_BA3':    [38, 48, 57, 76, 95, 114, 152, 190, 229, 305, 381, 457, 533, 610, 686, 762, 914, 1067, 1219, 1372, 1524, 1829, 2134, 2438],
        'T_Stueck_H':    [25, 32, 38, 51, 64, 76, 105, 124, 143, 178, 216, 254, 279, 305, 343, 381, 432, 521, 597, 673, 749, 889, 1029, 1168],
        'Red_Laenge_L':  [38, 50, 64, 76, 89, 89, 102, 127, 140, 152, 178, 203, 330, 356, 381, 508, 508, 610, 660, 711, 800, 900, 1000, 1100], 
        'Flansch_b_16':  [38, 40, 42, 45, 45, 50, 52, 55, 55, 62, 70, 78, 82, 85, 85, 90, 95, 105, 115, 125, 135, 155, 175, 195],
        'LK_k_16':       [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 355, 410, 470, 525, 585, 650, 770, 840, 950, 1050, 1160, 1380, 1590, 1820],
        'Schraube_M_16': ["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M24", "M24", "M24", "M27", "M27", "M30", "M33", "M33", "M36", "M36", "M39", "M45", "M45", "M52"],
        'L_Fest_16':     [55, 60, 60, 65, 65, 70, 70, 75, 80, 85, 100, 110, 110, 120, 130, 130, 150, 160, 170, 180, 190, 220, 240, 260],
        'L_Los_16':      [60, 65, 65, 70, 70, 75, 80, 85, 90, 100, 115, 125, 130, 140, 150, 150, 170, 180, 190, 210, 220, 250, 280, 300],
        'Lochzahl_16':   [4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 16, 16, 20, 20, 20, 24, 24, 28, 28, 32, 36, 40],
        'Flansch_b_10':  [38, 40, 42, 45, 45, 50, 52, 55, 55, 62, 70, 78, 82, 85, 85, 90, 95, 105, 115, 125, 135, 155, 175, 195],
        'LK_k_10':       [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 350, 400, 460, 515, 565, 620, 725, 840, 950, 1050, 1160, 1380, 1590, 1820],
        'Schraube_M_10': ["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M20", "M20", "M20", "M24", "M24", "M24", "M27", "M27", "M30", "M30", "M33", "M36", "M39", "M45"],
        'L_Fest_10':     [55, 60, 60, 65, 65, 70, 70, 75, 80, 85, 90, 90, 90, 100, 110, 110, 120, 130, 140, 150, 160, 190, 210, 230],
        'L_Los_10':      [60, 65, 65, 70, 70, 75, 80, 85, 90, 100, 105, 105, 110, 120, 130, 130, 140, 150, 160, 170, 180, 210, 240, 260],
        'Lochzahl_10':   [4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 16, 16, 20, 20, 20, 20, 24, 28, 28, 32, 36, 40]
    }
    return pd.DataFrame(raw_data)

DB_NAME = "rohrbau_profi.db"

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
                        created_at TEXT)''')
            c.execute("PRAGMA table_info(rohrbuch)")
            cols = [info[1] for info in c.fetchall()]
            if 'charge_apz' not in cols:
                try: c.execute("ALTER TABLE rohrbuch ADD COLUMN charge_apz TEXT")
                except: pass
            if 'project_id' not in cols:
                try: c.execute("ALTER TABLE rohrbuch ADD COLUMN project_id INTEGER")
                except: pass
            c.execute("INSERT OR IGNORE INTO projects (id, name, created_at) VALUES (1, 'Standard Baustelle', ?)", 
                      (datetime.now().strftime("%d.%m.%Y"),))
            c.execute("UPDATE rohrbuch SET project_id = 1 WHERE project_id IS NULL")
            conn.commit()

    @staticmethod
    def get_projects() -> List[tuple]:
        with sqlite3.connect(DB_NAME) as conn:
            return conn.cursor().execute("SELECT id, name FROM projects ORDER BY id ASC").fetchall()

    @staticmethod
    def create_project(name: str):
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.cursor().execute("INSERT INTO projects (name, created_at) VALUES (?, ?)", (name, datetime.now().strftime("%d.%m.%Y")))
                conn.commit()
            return True, "Projekt erstellt."
        except sqlite3.IntegrityError:
            return False, "Name existiert bereits."

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
            if not df.empty: df['Löschen'] = False 
            else: 
                df = pd.DataFrame(columns=["id", "iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser", "project_id", "Löschen"])
            return df

    @staticmethod
    def delete_entries(ids: List[int]):
        if not ids: return
        with sqlite3.connect(DB_NAME) as conn:
            placeholders = ', '.join('?' for _ in ids)
            conn.cursor().execute(f"DELETE FROM rohrbuch WHERE id IN ({placeholders})", ids)
            conn.commit()

    @staticmethod
    def get_known_values(column: str, project_id: int, limit: int = 50) -> List[str]:
        allowed = ['charge', 'charge_apz', 'schweisser', 'iso']
        if column not in allowed: return []
        with sqlite3.connect(DB_NAME) as conn:
            query = f'''SELECT {column} FROM rohrbuch WHERE project_id = ? AND {column} IS NOT NULL AND {column} != '' GROUP BY {column} ORDER BY MAX(id) DESC LIMIT ?'''
            rows = conn.cursor().execute(query, (project_id, limit)).fetchall()
            return [r[0] for r in rows]

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

class PipeCalculator:
    def __init__(self, df: pd.DataFrame): self.df = df
    def get_row(self, dn: int) -> pd.Series:
        row = self.df[self.df['DN'] == dn]
        return row.iloc[0] if not row.empty else self.df.iloc[0]
    def get_deduction(self, f_type: str, dn: int, pn: str, angle: float = 90.0) -> float:
        row = self.get_row(dn)
        suffix = "_16" if pn == "PN 16" else "_10"
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
    
    # SEGMENT BOGEN LOGIK (V7.8)
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
        if r_stub > r_main: return None
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
        plt.tight_layout(); return fig

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
        return fig

class Exporter:
    @staticmethod
    def to_excel(df):
        output = BytesIO()
        export_df = df.drop(columns=['Löschen', 'id', 'Auswahl', 'project_id'], errors='ignore')
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Daten')
        return output.getvalue()

    @staticmethod
    def to_pdf_logbook(df, project_name="Unbekannt"):
        if not PDF_AVAILABLE: return b""
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Rohrbuch: {project_name}", 0, 1, 'L')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 5, f"Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')}", 0, 1, 'L')
        pdf.ln(5)
        cols = ["ISO", "Naht", "Datum", "DN", "Bauteil", "Charge", "APZ", "Schweißer"]
        widths = [30, 20, 25, 20, 40, 35, 35, 30]
        pdf.set_font("Arial", 'B', 8)
        for i, c in enumerate(cols): pdf.cell(widths[i], 8, c, 1)
        pdf.ln()
        pdf.set_font("Arial", size=8)
        export_df = df.drop(columns=['Löschen', 'id', 'project_id'], errors='ignore')
        for _, row in export_df.iterrows():
            def g(k): 
                if k.lower() in row: return str(row[k.lower()])
                if k=="APZ" and 'charge_apz' in row: return str(row['charge_apz'])
                if k=="ISO" and 'iso' in row: return str(row['iso'])
                if k=="DN" and 'dimension' in row: return str(row['dimension'])
                return ""
            vals = [g(c) for c in cols]
            for i, v in enumerate(vals):
                try: pdf.cell(widths[i], 8, v[:20].encode('latin-1','replace').decode('latin-1'), 1)
                except: pdf.cell(widths[i], 8, "?", 1)
            pdf.ln()
        return pdf.output(dest='S').encode('latin-1')

    @staticmethod
    def to_pdf_sawlist(df, project_name="Unbekannt"):
        if not PDF_AVAILABLE: return b""
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Saegeauftrag: {project_name}", 0, 1, 'L')
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
                try: pdf.cell(widths[i], 8, val.encode('latin-1','replace').decode('latin-1'), 1)
                except: pdf.cell(widths[i], 8, "?", 1)
            pdf.ln()
        return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 4. UI SEITEN (TABS)
# -----------------------------------------------------------------------------

def render_sidebar_projects():
    st.sidebar.title("🏗️ Projekt")
    projects = DatabaseRepository.get_projects() 
    
    if 'active_project_id' not in st.session_state:
        if projects:
            st.session_state.active_project_id = projects[0][0]
            st.session_state.active_project_name = projects[0][1]
    
    current_ids = [p[0] for p in projects]
    if st.session_state.active_project_id not in current_ids and projects:
        st.session_state.active_project_id = projects[0][0]
        st.session_state.active_project_name = projects[0][1]

    project_names = [p[1] for p in projects]
    current_idx = 0
    for i, p in enumerate(projects):
        if p[0] == st.session_state.get('active_project_id'):
            current_idx = i
            break
            
    selected_name = st.sidebar.selectbox("Aktive Baustelle:", project_names, index=current_idx)
    new_id = [p[0] for p in projects if p[1] == selected_name][0]
    
    if new_id != st.session_state.active_project_id:
        st.session_state.active_project_id = new_id
        st.session_state.active_project_name = selected_name
        st.session_state.saved_cuts = [] 
        st.session_state.fitting_list = []
        st.rerun()

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

def render_smart_saw(calc: PipeCalculator, df: pd.DataFrame, current_dn: int, pn: str):
    st.subheader("🪚 Smarte Säge 6.4")
    
    proj_name = st.session_state.get('active_project_name', 'Unbekannt')
    st.markdown(f"<div class='project-tag'>📍 Projekt: {proj_name}</div>", unsafe_allow_html=True)

    if 'fitting_list' not in st.session_state: st.session_state.fitting_list = []
    if 'saved_cuts' not in st.session_state: st.session_state.saved_cuts = []
    if 'next_cut_id' not in st.session_state: st.session_state.next_cut_id = 1

    if st.session_state.saved_cuts:
        try: _ = st.session_state.saved_cuts[0].name
        except AttributeError: st.session_state.saved_cuts = []

    default_raw = 0.0
    if 'transfer_cut_length' in st.session_state:
        default_raw = st.session_state.pop('transfer_cut_length')
        st.toast("✅ Maß aus Geometrie übernommen!", icon="📏")

    c_calc, c_list = st.columns([1.3, 1.7])

    with c_calc:
        with st.container(border=True):
            st.markdown("#### 1. Neuer Schnitt")
            cut_name = st.text_input("Bezeichnung / Spool", placeholder="z.B. Strang A - 01", help="Name für die Liste")
            raw_len = st.number_input("Schnittmaß (Roh) [mm]", value=default_raw, min_value=0.0, step=10.0, format="%.1f")
            
            cg1, cg2, cg3 = st.columns(3)
            gap = cg1.number_input("Spalt (mm)", value=3.0, step=0.5)
            dicht_anz = cg2.number_input("Dichtungen", 0, 5, 0)
            dicht_thk = cg3.number_input("Dicke (mm)", 0.0, 5.0, 2.0, disabled=(dicht_anz==0))
            st.divider()
            st.caption("Bauteil-Abzüge:")
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

            st.divider()
            if final < 0: st.error(f"Negativmaß! ({final:.1f} mm)")
            else:
                st.success(f"Sägelänge: {final:.1f} mm")
                st.caption(f"Abzüge: Teile -{sum_fit:.1f} | Spalte -{sum_gap:.1f} | Dicht. -{sum_gskt:.1f}")
                if st.button("💾 In Schnittliste speichern", type="primary", use_container_width=True):
                    if raw_len > 0:
                        final_name = cut_name if cut_name.strip() else f"Schnitt #{st.session_state.next_cut_id}"
                        st.session_state.saved_cuts.append(SavedCut(st.session_state.next_cut_id, final_name, raw_len, final, f"{len(st.session_state.fitting_list)} Teile", datetime.now().strftime("%H:%M")))
                        st.session_state.next_cut_id += 1
                        st.session_state.fitting_list = [] 
                        st.success("Gespeichert!")
                        st.rerun()

    with c_list:
        st.markdown("#### 📋 Schnittliste & Export")
        if not st.session_state.saved_cuts:
            st.info("Noch keine Schnitte vorhanden.")
        else:
            data = [asdict(c) for c in st.session_state.saved_cuts]
            df_s = pd.DataFrame(data)
            df_s['Auswahl'] = False
            df_display = df_s[['Auswahl', 'name', 'raw_length', 'cut_length', 'details', 'id']]
            edited_df = st.data_editor(
                df_display, hide_index=True, use_container_width=True,
                column_config={"Auswahl": st.column_config.CheckboxColumn("☑️", width="small"), "name": st.column_config.TextColumn("Bez.", width="medium"), "raw_length": st.column_config.NumberColumn("Roh", format="%.0f"), "cut_length": st.column_config.NumberColumn("Säge", format="%.1f", width="medium"), "details": st.column_config.TextColumn("Info", width="small"), "id": None},
                disabled=["name", "raw_length", "cut_length", "details", "id"], key="saw_editor_v64"
            )
            selected_rows = edited_df[edited_df['Auswahl'] == True]
            selected_ids = selected_rows['id'].tolist()
            if selected_ids:
                st.info(f"{len(selected_ids)} Element(e) ausgewählt:")
                col_del, col_trans = st.columns(2)
                if col_del.button(f"🗑️ Löschen", type="primary", use_container_width=True):
                    st.session_state.saved_cuts = [c for c in st.session_state.saved_cuts if c.id not in selected_ids]
                    st.rerun()
                if col_trans.button(f"📝 Ins Rohrbuch", help="Kopiert ersten gewählten Schnitt ins Rohrbuch", use_container_width=True):
                    first_id = selected_ids[0]
                    cut_to_transfer = next((c for c in st.session_state.saved_cuts if c.id == first_id), None)
                    if cut_to_transfer:
                        st.session_state.logbook_defaults = {'iso': cut_to_transfer.name, 'len': cut_to_transfer.cut_length}
                        st.toast(f"'{cut_to_transfer.name}' kopiert! Bitte Tab wechseln.", icon="📋")

            st.divider()
            ce1, ce2 = st.columns(2)
            fname_base = f"Saege_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
            excel_data = Exporter.to_excel(df_s)
            ce1.download_button("📥 Excel", excel_data, f"{fname_base}.xlsx", use_container_width=True)
            if PDF_AVAILABLE:
                pdf_data = Exporter.to_pdf_sawlist(df_s, project_name=proj_name)
                ce2.download_button("📄 PDF", pdf_data, f"{fname_base}.pdf", use_container_width=True)
            if st.button("Alles Reset (Liste leeren)"):
                st.session_state.saved_cuts = []
                st.rerun()

def render_geometry_tools(calc: PipeCalculator, df: pd.DataFrame):
    st.subheader("📐 Geometrie V7.9 (Restored)")
    
    # JETZT 5 TABS - BOGEN (STANDARD) IST ZURÜCK
    geo_tabs = st.tabs(["2D Etage (S-Schlag)", "3D Raum-Etage (Rolling)", "Bogen (Standard)", "🦞 Segment-Bogen", "Stutzen"])
    
    # --- 1. 2D ETAGE ---
    with geo_tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            dn = st.selectbox("Nennweite", df['DN'], index=5, key="2d_dn")
            offset = st.number_input("Versprung (H) [mm]", value=500.0, step=10.0, key="2d_off")
            angle = st.selectbox("Fittings (°)", [30, 45, 60], index=1, key="2d_ang")
            if st.button("Berechnen 2D", type="primary"):
                res = calc.calculate_2d_offset(dn, offset, angle)
                st.session_state.calc_res_2d = res 
        
        with c2:
            if 'calc_res_2d' in st.session_state:
                res = st.session_state.calc_res_2d
                if "error" in res: st.error(res["error"])
                else:
                    st.markdown("#### Ergebnis")
                    m1, m2 = st.columns(2)
                    m1.metric("Zuschnitt (Rohr)", f"{res['cut_length']:.1f} mm")
                    m2.metric("Etagenlänge", f"{res['hypotenuse']:.1f} mm")
                    st.info(f"Benötigter Platz (Länge): {res['run']:.1f} mm")
                    
                    if st.button("➡️ An Säge (2D)", key="btn_2d_saw"):
                        st.session_state.transfer_cut_length = res['cut_length']
                        st.toast("Maß übertragen!", icon="📏")
                    
                    fig_2d = Visualizer.plot_2d_offset(res['run'], res['offset'])
                    st.pyplot(fig_2d, use_container_width=False)

    # --- 2. 3D RAUM ETAGE (ROLLING OFFSET) ---
    with geo_tabs[1]:
        st.info("💡 Berechnet eine Raum-Etage mit **Standard-Fittings**.")
        col_in, col_out, col_vis = st.columns([1, 1, 1.5]) 
        with col_in:
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
                    st.toast("Maß übertragen!", icon="📏")

        with col_vis:
            st.markdown("**3D Simulation**")
            fig_3d = Visualizer.plot_rolling_offset_3d_room(roll_val, run_length, set_val)
            st.pyplot(fig_3d, use_container_width=False)
            with st.expander("Verdrehung (Wasserwaage)"):
                fig_gauge = Visualizer.plot_rotation_gauge(roll_val, set_val, rot_angle)
                st.pyplot(fig_gauge, use_container_width=False)

    # --- 3. BOGEN RECHNER (RESTORED) ---
    with geo_tabs[2]:
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

    # --- 4. SEGMENT BOGEN (LOBSTER) ---
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

    # --- 5. STUTZEN ---
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
                with c_r2: st.pyplot(fig)
            except ValueError as e: st.error(str(e))

def render_logbook(df_pipe: pd.DataFrame):
    st.subheader("📝 Digitales Rohrbuch (V7.7.1)")
    
    proj_name = st.session_state.get('active_project_name', 'Unbekannt')
    active_pid = st.session_state.get('active_project_id', 1)
    st.markdown(f"<div class='project-tag'>📍 Projekt: {proj_name} (ID: {active_pid})</div>", unsafe_allow_html=True)

    defaults = st.session_state.get('logbook_defaults', {})
    def_iso = defaults.get('iso', '')
    def_len = defaults.get('len', 0.0)
    
    with st.expander("Eintrag hinzufügen", expanded=True):
        with st.form("lb_new"):
            c1, c2, c3 = st.columns(3)
            iso_known = DatabaseRepository.get_known_values('iso', active_pid)
            if iso_known:
                iso_sel = c1.selectbox("ISO / Bez.", ["✨ Neu / Manuell"] + iso_known)
                if iso_sel == "✨ Neu / Manuell":
                    iso = c1.text_input("ISO manuell", value=def_iso, label_visibility="collapsed")
                else:
                    iso = iso_sel
            else:
                iso = c1.text_input("ISO / Bez.", value=def_iso)

            naht = c2.text_input("Naht")
            dat = c3.date_input("Datum")
            
            c4, c5, c6 = st.columns(3)
            def_idx = 0 
            if def_iso: def_idx = 5 
            bt = c4.selectbox("Bauteil", ["Rohrstoß", "Bogen", "Flansch", "T-Stück", "Stutzen", "Passstück", "Nippel", "Muffe"], index=def_idx)
            dn = c5.selectbox("Dimension", df_pipe['DN'], index=8)
            laenge_in = c6.number_input("Länge (mm)", value=float(def_len if def_len > 0 else 0.0)) 
            
            ch_known = DatabaseRepository.get_known_values('charge', active_pid)
            if ch_known:
                ch_sel = st.selectbox("Charge (Auswahl)", ["✨ Neu / Manuell"] + ch_known)
                if ch_sel == "✨ Neu / Manuell":
                    ch = st.text_input("Charge (Eingabe)", label_visibility="collapsed")
                else:
                    ch = ch_sel
            else:
                ch = st.text_input("Charge")

            c7, c8 = st.columns(2)
            apz_known = DatabaseRepository.get_known_values('charge_apz', active_pid)
            with c7:
                if apz_known:
                    apz_sel = st.selectbox("APZ (Auswahl)", ["✨ Neu / Manuell"] + apz_known)
                    if apz_sel == "✨ Neu / Manuell":
                        apz = st.text_input("APZ (Eingabe)", label_visibility="collapsed")
                    else:
                        apz = apz_sel
                else:
                    apz = st.text_input("APZ / Zeugnis")

            sch_known = DatabaseRepository.get_known_values('schweisser', active_pid)
            with c8:
                if sch_known:
                    sch_sel = st.selectbox("Schweißer (Auswahl)", ["✨ Neu / Manuell"] + sch_known)
                    if sch_sel == "✨ Neu / Manuell":
                        sch = st.text_input("Schweißer (Eingabe)", label_visibility="collapsed")
                    else:
                        sch = sch_sel
                else:
                    sch = st.text_input("Schweißer")
            
            if st.form_submit_button("Speichern 💾", type="primary"):
                DatabaseRepository.add_entry({
                    "iso": iso, "naht": naht, "datum": dat.strftime("%d.%m.%Y"),
                    "dimension": f"DN {dn}", "bauteil": bt, "laenge": laenge_in,
                    "charge": ch, "charge_apz": apz, "schweisser": sch,
                    "project_id": active_pid
                })
                if 'logbook_defaults' in st.session_state: del st.session_state['logbook_defaults']
                st.success("Gespeichert")
                st.rerun()

    st.divider()
    
    df = DatabaseRepository.get_logbook_by_project(active_pid)
    
    if not df.empty:
        ce1, ce2, _ = st.columns([1,1,3])
        fname_base = f"Rohrbuch_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        ce1.download_button("📥 Excel", Exporter.to_excel(df), f"{fname_base}.xlsx")
        if PDF_AVAILABLE: 
            ce2.download_button("📄 PDF", Exporter.to_pdf_logbook(df, project_name=proj_name), f"{fname_base}.pdf")
            
        edited = st.data_editor(df, hide_index=True, use_container_width=True, column_config={"Löschen": st.column_config.CheckboxColumn(default=False)}, disabled=["id", "iso", "naht", "datum", "dimension", "bauteil", "charge", "charge_apz", "schweisser", "project_id"])
        to_del = edited[edited['Löschen'] == True]
        if not to_del.empty:
            if st.button(f"🗑️ {len(to_del)} löschen", type="primary"):
                DatabaseRepository.delete_entries(to_del['id'].tolist())
                st.rerun()
    else:
        st.info(f"Keine Einträge für Projekt '{proj_name}' gefunden. Beginne mit der ersten Naht oder importiere aus der Säge.")

def render_tab_handbook(calc: PipeCalculator, dn: int, pn: str):
    row = calc.get_row(dn)
    suffix = "_16" if pn == "PN 16" else "_10"
    st.subheader(f"📚 Smart Data: DN {dn} / {pn}")

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
# 5. MAIN
# -----------------------------------------------------------------------------

def main():
    DatabaseRepository.init_db()
    df_pipe = get_pipe_data()
    calc = PipeCalculator(df_pipe)

    # 1. RENDER SIDEBAR (PROJECTS V7)
    render_sidebar_projects()

    with st.sidebar:
        st.subheader("⚙️ Setup")
        dn = st.selectbox("Nennweite", df_pipe['DN'], index=8)
        pn = st.radio("Druckstufe", ["PN 16", "PN 10"], horizontal=True)

    t1, t2, t3, t4 = st.tabs(["🪚 Smarte Säge", "📐 Geometrie", "📝 Rohrbuch", "📚 Smart Data"])
    
    with t1: render_smart_saw(calc, df_pipe, dn, pn)
    with t2: render_geometry_tools(calc, df_pipe)
    with t3: render_logbook(df_pipe)
    with t4: render_tab_handbook(calc, dn, pn)

if __name__ == "__main__":
    main()
