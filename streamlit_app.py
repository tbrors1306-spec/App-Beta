"""
PipeCraft V33.1 (Stable Enterprise Edition)
-------------------------------------------
Version: 33.1.0 (Hotfix)
Date: 2023-12-21
Author: Senior Lead Software Engineer (AI)

Changelog:
- HOTFIX: Fixed TypeError in 'number_input' by enforcing explicit keyword arguments (value=...).
- REFAC: Applied strict parameter naming across all UI widgets for stability.
- FEAT: All previous features (3D, Inventory, Smart Cut) are preserved.
"""

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D 
import sqlite3
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import List, Tuple, Any, Optional, Union, Dict

# -----------------------------------------------------------------------------
# 0. SYSTEM CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipeCraft")

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("FPDF Library not found. PDF export disabled.")

st.set_page_config(
    page_title="PipeCraft V33.1 Enterprise",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #0f172a; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #1e293b !important; font-weight: 800; letter-spacing: -0.5px; }
    
    .result-card-blue { 
        background-color: #eff6ff; padding: 20px; border-radius: 10px; 
        border-left: 6px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        margin-bottom: 15px; color: #1e3a8a; font-size: 1rem; 
    }
    
    .result-card-green { 
        background: linear-gradient(to right, #f0fdf4, #ffffff); padding: 25px; 
        border-radius: 12px; border-left: 8px solid #22c55e; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-bottom: 15px; 
        text-align: center; font-size: 1.6rem; font-weight: 800; color: #14532d; 
    }
    
    .detail-box { 
        background-color: #f1f5f9; border: 1px solid #cbd5e1; padding: 15px; 
        border-radius: 8px; text-align: center; font-size: 0.9rem; 
        color: #334155; height: 100%; display: flex; flex-direction: column; 
        justify-content: center;
    }
    
    .weight-box { 
        background-color: #fff1f2; border: 1px solid #fecdd3; color: #be123c; 
        padding: 15px; border-radius: 8px; text-align: center; 
        font-weight: bold; font-size: 1.1rem; margin-top: 10px; 
    }
    
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input { 
        border-radius: 8px; border: 1px solid #cbd5e1; 
    }
    
    div.stButton > button { 
        width: 100%; border-radius: 8px; font-weight: 600; 
        border: 1px solid #cbd5e1; transition: all 0.2s; 
    }
    div.stButton > button:hover { 
        border-color: #3b82f6; color: #3b82f6; background-color: #eff6ff; 
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. DATA LAYER
# -----------------------------------------------------------------------------

RAW_DATA = {
    'DN':           [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600],
    'D_Aussen':     [33.7, 42.4, 48.3, 60.3, 76.1, 88.9, 114.3, 139.7, 168.3, 219.1, 273.0, 323.9, 355.6, 406.4, 457.0, 508.0, 610.0, 711.0, 813.0, 914.0, 1016.0, 1219.0, 1422.0, 1626.0],
    'Radius_BA3':   [38, 48, 57, 76, 95, 114, 152, 190, 229, 305, 381, 457, 533, 610, 686, 762, 914, 1067, 1219, 1372, 1524, 1829, 2134, 2438],
    'T_Stueck_H':   [25, 32, 38, 51, 64, 76, 105, 124, 143, 178, 216, 254, 279, 305, 343, 381, 432, 521, 597, 673, 749, 889, 1029, 1168],
    'Red_Laenge_L': [38, 50, 64, 76, 89, 89, 102, 127, 140, 152, 178, 203, 330, 356, 381, 508, 508, 610, 660, 711, 800, 900, 1000, 1100], 
    'Flansch_b_16': [38, 40, 42, 45, 45, 50, 52, 55, 55, 62, 70, 78, 82, 85, 85, 90, 95, 105, 115, 125, 135, 155, 175, 195],
    'LK_k_16':      [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 355, 410, 470, 525, 585, 650, 770, 840, 950, 1050, 1160, 1380, 1590, 1820],
    'Schraube_M_16':["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M24", "M24", "M24", "M27", "M27", "M30", "M33", "M33", "M36", "M36", "M39", "M45", "M45", "M52"],
    'L_Fest_16':    [55, 60, 60, 65, 65, 70, 70, 75, 80, 85, 100, 110, 110, 120, 130, 130, 150, 160, 170, 180, 190, 220, 240, 260],
    'L_Los_16':     [60, 65, 65, 70, 70, 75, 80, 85, 90, 100, 115, 125, 130, 140, 150, 150, 170, 180, 190, 210, 220, 250, 280, 300],
    'Lochzahl_16':  [4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 16, 16, 20, 20, 20, 24, 24, 28, 28, 32, 36, 40],
    'Flansch_b_10': [38, 40, 42, 45, 45, 50, 52, 55, 55, 62, 70, 78, 82, 85, 85, 90, 95, 105, 115, 125, 135, 155, 175, 195],
    'LK_k_10':      [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 350, 400, 460, 515, 565, 620, 725, 840, 950, 1050, 1160, 1380, 1590, 1820],
    'Schraube_M_10':["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M20", "M20", "M20", "M24", "M24", "M24", "M27", "M27", "M30", "M30", "M33", "M36", "M39", "M45"],
    'L_Fest_10':    [55, 60, 60, 65, 65, 70, 70, 75, 80, 85, 90, 90, 90, 100, 110, 110, 120, 130, 140, 150, 160, 190, 210, 230],
    'L_Los_10':     [60, 65, 65, 70, 70, 75, 80, 85, 90, 100, 105, 105, 110, 120, 130, 130, 140, 150, 160, 170, 180, 210, 240, 260],
    'Lochzahl_10':  [4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 16, 16, 20, 20, 20, 20, 24, 28, 28, 32, 36, 40]
}

try:
    df_pipe = pd.DataFrame(RAW_DATA)
except ValueError as e:
    st.error(f"FATAL ERROR: Inconsistent Data Arrays. {e}")
    st.stop()

SCHRAUBEN_DB = { 
    "M12": [18, 60], "M16": [24, 130], "M20": [30, 250], "M24": [36, 420], 
    "M27": [41, 600], "M30": [46, 830], "M33": [50, 1100], "M36": [55, 1400], 
    "M39": [60, 1800], "M45": [70, 2700], "M52": [80, 4200] 
}

WS_LISTE = [2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 8.8, 10.0, 11.0, 12.5, 14.2, 16.0]

WS_STD_MAP = {
    25: 3.2, 32: 3.6, 40: 3.6, 50: 3.9, 65: 5.2, 80: 5.5, 100: 6.0, 
    125: 6.6, 150: 7.1, 200: 8.2, 250: 9.3, 300: 9.5, 350: 9.5, 400: 9.5, 
    450: 9.5, 500: 9.5
}

DB_NAME = "pipecraft.db"

# -----------------------------------------------------------------------------
# 2. LOGIC LAYER (ENGINES & HELPERS)
# -----------------------------------------------------------------------------

def get_row_by_dn(dn: int) -> pd.Series:
    try:
        return df_pipe[df_pipe['DN'] == dn].iloc[0]
    except IndexError:
        return df_pipe.iloc[0]

def get_schrauben_info(gewinde: str) -> List[Union[int, str]]:
    return SCHRAUBEN_DB.get(gewinde, ["?", "?"])

def parse_abzuege(text: str) -> float:
    try:
        if not text:
            return 0.0
        clean_text = text.replace(",", ".").replace(" ", "")
        if not all(c in "0123456789.+-*/()" for c in clean_text):
            return 0.0
        return float(pd.eval(clean_text))
    except Exception:
        return 0.0

# UI Helpers
def get_ws_index(val: float) -> int:
    try: 
        return WS_LISTE.index(val)
    except ValueError: 
        return 6 

def get_verf_index(val: str) -> int:
    opts = ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"]
    if val in opts:
        return opts.index(val)
    return 0

def get_disc_idx(val: str) -> int:
    opts = ["125 mm", "180 mm", "230 mm"]
    if val in opts:
        return opts.index(val)
    return 0

def get_sys_idx(val: str) -> int:
    opts = ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"]
    if val in opts:
        return opts.index(val)
    return 0

# --- FITTING MANAGER (SMART CUT) ---

@dataclass
class SelectedFitting:
    type_name: str
    count: int
    deduction_single: float
    dn_spec: int

class FittingManager:
    @staticmethod
    def get_deduction(type_name: str, dn_target: int, pn_suffix: str = "_16", custom_angle: float = 45.0) -> float:
        row_data = get_row_by_dn(dn_target)
        
        if type_name == "Bogen 90° (BA3)":
            return float(row_data['Radius_BA3'])
        
        elif type_name == "Bogen (Zuschnitt)":
            radius = float(row_data['Radius_BA3'])
            angle_rad = math.radians(custom_angle / 2)
            return radius * math.tan(angle_rad)
            
        elif type_name == "Flansch (Vorschweiß)":
            return float(row_data[f'Flansch_b{pn_suffix}'])
            
        elif type_name == "T-Stück":
            return float(row_data['T_Stueck_H'])
            
        elif type_name == "Reduzierung (konz.)":
            return float(row_data['Red_Laenge_L'])
            
        return 0.0

# --- PHYSICS ENGINE ---

class PhysicsEngine:
    DENSITY_STEEL = 7.85
    DENSITY_CEMENT = 2.40
    
    @staticmethod
    def calculate_pipe_weight(dn_idx: int, ws: float, length_mm: float, is_zme: bool = False) -> float:
        try:
            da_mm = df_pipe.iloc[dn_idx]['D_Aussen']
            length_dm = length_mm / 100.0
            ra_dm = (da_mm / 2) / 100.0
            ri_stahl_dm = ra_dm - (ws / 100.0)
            
            vol_stahl = math.pi * (ra_dm**2 - ri_stahl_dm**2) * length_dm
            weight = vol_stahl * PhysicsEngine.DENSITY_STEEL
            
            if is_zme:
                dn_val = df_pipe.iloc[dn_idx]['DN']
                if dn_val < 300:
                    cem_th_mm = 6.0
                elif dn_val < 600:
                    cem_th_mm = 9.0
                else:
                    cem_th_mm = 12.0
                    
                ri_cem_dm = ri_stahl_dm - (cem_th_mm / 100.0)
                
                if ri_cem_dm > 0:
                    vol_cem = math.pi * (ri_stahl_dm**2 - ri_cem_dm**2) * length_dm
                    weight += (vol_cem * PhysicsEngine.DENSITY_CEMENT)
                    
            return round(weight, 1)
        except Exception:
            return 0.0

# --- COST ENGINE (V23.3 RESTORED) ---

class CostEngine:
    @staticmethod
    def calculate_welding_detailed(
        dn: int, 
        ws: float, 
        process: str, 
        pers_count: int, 
        anz: int, 
        factor: float,
        p_lohn: float,
        p_mach: float,
        p_draht: float,
        p_cel: float,
        p_gas: float
    ) -> Tuple[float, float, str]:
        
        zoll = dn / 25.0
        
        if process == "WIG":
            min_per_inch = 10.0
        elif "CEL" in process:
            min_per_inch = 3.5
        else:
            min_per_inch = 5.0 # MAG / E-Hand
            
        t_weld = zoll * min_per_inch
        t_fit = zoll * 2.5
        
        duration_per_seam = ((t_weld + t_fit) / pers_count) * factor
        total_hours = (duration_per_seam * anz) / 60
        labor_cost = total_hours * (pers_count * (p_lohn + p_mach))
        
        da = df_pipe[df_pipe['DN'] == dn].iloc[0]['D_Aussen']
        weld_cross_section_mm2 = ws**2 * 0.8
        weld_circumference_mm = da * math.pi
        weld_volume_mm3 = weld_circumference_mm * weld_cross_section_mm2
        
        kg_filler_per_seam = (weld_volume_mm3 * 7.85) / 1_000_000 * 1.5 
        
        mat_cost = 0.0
        mat_text = ""
        
        if "CEL" in process:
            kg_electrodes_total = (5.0 * kg_filler_per_seam) * anz
            mat_cost = kg_electrodes_total * p_cel
            mat_text = "CEL Elektroden"
        else:
            wire_cost = (kg_filler_per_seam * p_draht) * anz
            weld_time_only_min = (t_weld * factor) * anz
            gas_cost = weld_time_only_min * 15 * p_gas
            
            mat_cost = wire_cost + gas_cost
            mat_text = f"{round(kg_filler_per_seam * anz, 1)} kg Draht"
            
        return duration_per_seam, labor_cost + mat_cost, mat_text

    @staticmethod
    def calculate_cutting_detailed(
        dn: int,
        ws: float,
        disc_type: str,
        anz: int,
        is_zma: bool,
        is_iso: bool,
        factor: float,
        p_lohn: float,
        p_stahl_disc: float,
        p_dia_disc: float
    ) -> Tuple[float, float, str]:
        
        zoll = dn / 25.0
        
        if "230" in disc_type:
            cap = 14000
        elif "180" in disc_type:
            cap = 7000
        else:
            cap = 3500
            
        zma_fac = 3.0 if is_zma else 1.0
        iso_fac = 1.3 if is_iso else 1.0
        
        base_time = zoll * 0.5
        t_total = base_time * zma_fac * iso_fac * factor * anz
        
        da = df_pipe[df_pipe['DN'] == dn].iloc[0]['D_Aussen']
        area_mm2 = da * math.pi * ws
        
        wear_factor = 2.0 if is_zma else 1.0
        total_area = area_mm2 * anz * wear_factor
        
        n_disc = math.ceil(total_area / cap)
        disc_price = p_dia_disc if is_zma else p_stahl_disc
        
        mat_cost = n_disc * disc_price
        labor_cost = (t_total / 60) * p_lohn
        
        return t_total, labor_cost + mat_cost, f"{n_disc}x Scheiben"

# --- GEOMETRY ENGINE ---

class GeometryEngine:
    @staticmethod
    def solve_offset_3d(h: float, l: float, b: float) -> Tuple[float, float]:
        travel = math.sqrt(h**2 + l**2 + b**2)
        spread = math.sqrt(l**2 + b**2)
        
        if spread == 0:
            angle = 90.0
        else:
            angle = math.degrees(math.atan(h / spread))
            
        return travel, angle

# --- INVENTORY LOGIC ---

@dataclass
class InventoryItem:
    article_id: str
    name: str
    price_per_unit: float
    current_stock: int
    reorder_point: int
    target_stock: int

    def calculate_reorder_qty(self) -> int:
        if self.current_stock <= self.reorder_point:
            return self.target_stock - self.current_stock
        return 0
        
    @property
    def stock_value(self) -> float:
        return self.current_stock * self.price_per_unit

# -----------------------------------------------------------------------------
# 3. VISUALIZATION LAYER
# -----------------------------------------------------------------------------

class Visualizer:
    @staticmethod
    def plot_true_3d_pipe(length: float, width: float, height: float, azim: int, elev: int) -> plt.Figure:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        xs = [0, length]
        ys = [0, width]
        zs = [0, height]
        
        ax.plot(xs, ys, zs, color='#ef4444', linewidth=5, solid_capstyle='round')
        
        ax.plot([0, length], [0, 0], [0, 0], 'k--', alpha=0.2)
        ax.plot([length, length], [0, width], [0, 0], 'k--', alpha=0.2)
        ax.plot([0, length], [width, width], [0, 0], 'k--', alpha=0.1)
        ax.plot([length, length], [width, width], [0, height], 'k--', alpha=0.3)
        
        ax.scatter([0], [0], [0], color='black', s=50)
        ax.scatter([length], [width], [height], color='#10b981', s=50)
        
        ax.set_xlabel('L')
        ax.set_ylabel('B')
        ax.set_zlabel('H')
        
        max_dim = max(abs(length), abs(width), abs(height))
        if max_dim == 0:
            max_dim = 100
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_stutzen_curve(r_haupt: float, r_stutzen: float) -> plt.Figure:
        angles = range(0, 361, 5)
        try:
            depths = [r_haupt - math.sqrt(r_haupt**2 - (r_stutzen * math.sin(math.radians(a)))**2) for a in angles]
        except ValueError:
            return plt.figure()

        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

# -----------------------------------------------------------------------------
# 4. DATABASE REPOSITORY
# -----------------------------------------------------------------------------

class DatabaseRepository:
    @staticmethod
    def init_tables():
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (id INTEGER PRIMARY KEY AUTOINCREMENT, iso TEXT, naht TEXT, datum TEXT, dimension TEXT, bauteil TEXT, laenge REAL, charge TEXT, schweisser TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS kalkulation (id INTEGER PRIMARY KEY AUTOINCREMENT, typ TEXT, info TEXT, menge REAL, zeit_min REAL, kosten REAL, mat_text TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS inventory (article_id TEXT PRIMARY KEY, name TEXT, price REAL, current_stock INTEGER, reorder_point INTEGER, target_stock INTEGER)''')
            conn.commit()

    @staticmethod
    def add(table: str, data: Tuple):
        with sqlite3.connect(DB_NAME) as conn:
            p = ",".join(["?"] * len(data))
            if table == "rohrbuch":
                c = "iso, naht, datum, dimension, bauteil, laenge, charge, schweisser"
            else:
                c = "typ, info, menge, zeit_min, kosten, mat_text"
            conn.cursor().execute(f'INSERT INTO {table} ({c}) VALUES ({p})', data)
            conn.commit()

    @staticmethod
    def get_all(table: str) -> pd.DataFrame:
        with sqlite3.connect(DB_NAME) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)

    @staticmethod
    def delete(table: str, id: int):
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(f"DELETE FROM {table} WHERE id=?", (id,))
            conn.commit()

    @staticmethod
    def clear(table: str):
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(f"DELETE FROM {table}")
            conn.commit()

    # Inventory Specific
    @staticmethod
    def add_inventory(item: InventoryItem) -> bool:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.cursor().execute('INSERT INTO inventory VALUES (?,?,?,?,?,?)', (item.article_id, item.name, item.price_per_unit, item.current_stock, item.reorder_point, item.target_stock))
                return True
        except: return False

    @staticmethod
    def get_inventory() -> List[InventoryItem]:
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.cursor().execute('SELECT * FROM inventory').fetchall()
            return [InventoryItem(*r) for r in rows]

    @staticmethod
    def update_inventory(id: str, qty: int):
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute('UPDATE inventory SET current_stock=? WHERE article_id=?', (qty, id))
            conn.commit()

# --- Export Helpers ---
def export_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return out.getvalue()

def export_pdf(df):
    if not PDF_AVAILABLE: return b""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Report", 0, 1, 'C')
    pdf.ln(5)
    for _, r in df.iterrows():
        try:
            txt = f"{r.get('typ','')} | {r.get('info','')} | {r.get('kosten','')} | {r.get('mat_text','')}"
            pdf.cell(0, 8, txt.encode('latin-1','replace').decode('latin-1'), 1, 1)
        except: pass
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 5. INITIALIZATION
# -----------------------------------------------------------------------------

# Start DB
DatabaseRepository.init_tables()

# Init Session State
if 'store' not in st.session_state:
    st.session_state.store = {
        'saw_mass': 1000.0, 
        'saw_gap': 4.0, 
        'saw_deduct': "0", 
        'saw_zme': False,
        'kw_dn': 200, 
        'kw_ws': 6.3, 
        'kw_verf': "WIG", 
        'kw_pers': 1, 
        'kw_anz': 1, 
        'kw_split': False, 
        'kw_factor': 1.0,
        'cut_dn': 200, 
        'cut_ws': 6.3, 
        'cut_disc': "125 mm", 
        'cut_anz': 1, 
        'cut_zma': False, 
        'cut_iso': False, 
        'cut_factor': 1.0,
        'iso_sys': "Schrumpfschlauch (WKS)", 
        'iso_dn': 200, 
        'iso_anz': 1, 
        'iso_factor': 1.0,
        'mon_dn': 200, 
        'mon_type': "Schieber", 
        'mon_anz': 1, 
        'mon_factor': 1.0,
        'reg_min': 60, 
        'reg_pers': 2, 
        'bogen_winkel': 45, 
        'view_azim': 45, 
        'view_elev': 30,
        # Prices
        'p_lohn': 60.0, 'p_stahl': 2.5, 'p_dia': 45.0, 'p_cel': 0.40, 'p_draht': 15.0, 
        'p_gas': 0.05, 'p_wks': 25.0, 'p_kebu1': 15.0, 'p_kebu2': 12.0, 
        'p_primer': 12.0, 'p_machine': 15.0
    }

if 'fitting_list' not in st.session_state:
    st.session_state.fitting_list = []

# Callbacks
def save_val(key):
    st.session_state.store[key] = st.session_state[f"_{key}"]

def get_val(key):
    return st.session_state.store.get(key)

def update_kw_dn():
    st.session_state.store['kw_dn'] = st.session_state['_kw_dn']
    if st.session_state.store['kw_dn'] >= 300:
        st.session_state.store['kw_pers'] = 2
    else:
        st.session_state.store['kw_pers'] = 1

# -----------------------------------------------------------------------------
# 6. UI IMPLEMENTATION
# -----------------------------------------------------------------------------

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=50) 
st.sidebar.markdown("### Menü")
selected_dn_global = st.sidebar.selectbox("Nennweite (Global)", df_pipe['DN'], index=8, key="global_dn") 
selected_pn = st.sidebar.radio("Druckstufe", ["PN 16", "PN 10"], index=0, key="global_pn") 

# Global Context
row = get_row_by_dn(selected_dn_global)
standard_radius = float(row['Radius_BA3'])
suffix = "_16" if selected_pn == "PN 16" else "_10"

st.title("PipeCraft V33.1")
st.caption(f"🔧 Aktive Konfiguration: DN {selected_dn_global} | {selected_pn} | Radius: {standard_radius} mm")

# Tabs
tab_buch, tab_werk, tab_proj, tab_info, tab_lager = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch", "💰 Kalkulation", "📦 Lager"])

# -----------------------------------------------------------------------------
# TAB 1: TABELLENBUCH
# -----------------------------------------------------------------------------
with tab_buch:
    st.subheader("Rohr & Formstücke")
    
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='result-card-blue'><b>Außen-Ø:</b> {row['D_Aussen']} mm</div>", unsafe_allow_html=True)
    c1.markdown(f"<div class='result-card-blue'><b>Radius (3D):</b> {standard_radius} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>T-Stück (H):</b> {row['T_Stueck_H']} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>Reduzierung (L):</b> {row['Red_Laenge_L']} mm</div>", unsafe_allow_html=True)
    
    st.divider()
    st.subheader(f"Flansch & Montage ({selected_pn})")
    
    schraube = row[f'Schraube_M{suffix}']
    sw, nm = get_schrauben_info(schraube)
    
    mc1, mc2 = st.columns(2)
    mc1.markdown(f"<div class='result-card-blue'><b>Blattstärke:</b> {row[f'Flansch_b{suffix}']} mm</div>", unsafe_allow_html=True)
    mc2.markdown(f"<div class='result-card-blue'><b>Schraube:</b> {row[f'Lochzahl{suffix}']}x {schraube} (SW {sw})</div>", unsafe_allow_html=True)
    
    c_d1, c_d2, c_d3 = st.columns(3)
    c_d1.markdown(f"<div class='detail-box'>Länge (Fest-Fest)<br><span class='detail-value'>{row[f'L_Fest{suffix}']} mm</span></div>", unsafe_allow_html=True)
    c_d2.markdown(f"<div class='detail-box'>Länge (Fest-Los)<br><span class='detail-value'>{row[f'L_Los{suffix}']} mm</span></div>", unsafe_allow_html=True)
    c_d3.markdown(f"<div class='detail-box'>Drehmoment<br><span class='detail-value'>{nm} Nm</span></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: WERKSTATT
# -----------------------------------------------------------------------------
with tab_werk:
    tool_mode = st.radio("Werkzeug wählen:", ["📏 Säge (Passstück)", "🔄 Bogen (Zuschnitt)", "🔥 Stutzen (Schablone)", "📐 Etage (Versatz)"], horizontal=True, label_visibility="collapsed", key="tool_mode_nav")
    st.divider()
    
    # 2.1 SMART CUT (SÄGE)
    if "Säge" in tool_mode:
        st.subheader("Smart Cut System")
        
        c1, c2 = st.columns(2)
        iso_mass = c1.number_input("Gesamtmaß (Iso)", value=get_val('saw_mass'), step=10.0, key="_saw_mass", on_change=save_val, args=('saw_mass',))
        spalt = c2.number_input("Wurzelspalt (pro Naht)", value=get_val('saw_gap'), key="_saw_gap", on_change=save_val, args=('saw_gap',))
        
        st.divider()
        st.markdown("#### Bauteile hinzufügen")
        
        # --- Configurator mit Multi-DN ---
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1])
        
        f_type = col_f1.selectbox("Typ", ["Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung (konz.)"])
        
        # NEU: Dimension wählen (Standard = Globaler DN)
        f_dn = col_f2.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(selected_dn_global))
        
        # Winkel nur bei Bogen Zuschnitt
        f_angle = 45.0
        if "Zuschnitt" in f_type:
            f_angle = col_f3.number_input("Winkel", value=45.0, step=1.0)
        else:
            col_f3.write("-")
            
        f_count = col_f4.number_input("Anzahl", value=1, min_value=1, step=1)
        
        if st.button("Hinzufügen (+)", type="secondary"):
            # Hole Abzugsmaß für die SPEZIFISCH gewählte DN
            deduct = FittingManager.get_deduction(f_type, f_dn, suffix, f_angle)
            
            # Label
            name = f"{f_type} (DN {f_dn})"
            if "Zuschnitt" in f_type:
                name += f" [{f_angle}°]"
            
            # Speichern
            st.session_state.fitting_list.append(SelectedFitting(name, f_count, deduct, f_dn))
            st.rerun()
            
        # --- Liste & Berechnung ---
        total_deduct = 0.0
        total_gaps = 0
        
        if st.session_state.fitting_list:
            st.markdown("---")
            for i, item in enumerate(st.session_state.fitting_list):
                c_i1, c_i2, c_i3 = st.columns([4, 2, 1])
                sub = item.deduction_single * item.count
                total_deduct += sub
                total_gaps += item.count
                
                c_i1.write(f"**{item.count}x** {item.type_name}")
                c_i2.caption(f"-{round(sub, 1)} mm")
                if c_i3.button("🗑️", key=f"del_{i}"):
                    st.session_state.fitting_list.pop(i)
                    st.rerun()
            
            if st.button("Liste leeren", type="primary"):
                st.session_state.fitting_list = []
                st.rerun()
        
        # Final Calculation
        gap_deduct = total_gaps * spalt
        final_cut = iso_mass - total_deduct - gap_deduct
        
        st.markdown("---")
        
        if final_cut < 0:
            st.error(f"Fehler: Abzüge ({round(total_deduct+gap_deduct, 1)} mm) sind größer als das Maß!")
        else:
            c_res1, c_res2 = st.columns(2)
            c_res1.markdown(f"<div class='result-card-green'>Sägelänge: {round(final_cut, 1)} mm</div>", unsafe_allow_html=True)
            c_res2.info(f"Formteile: -{round(total_deduct, 1)} mm | Spalte: -{round(gap_deduct, 1)} mm")
            
            # Gewicht
            dn_idx = df_pipe[df_pipe['DN'] == selected_dn_global].index[0]
            std_ws = WS_STD_MAP.get(selected_dn_global, 4.0)
            c_zme = st.checkbox("ZME?", value=get_val('saw_zme'), key="_saw_zme", on_change=save_val, args=('saw_zme',))
            
            kg = PhysicsEngine.calculate_pipe_weight(dn_idx, std_ws, final_cut, c_zme)
            st.markdown(f"<div class='weight-box'>⚖️ Gewicht: ~ {kg} kg</div>", unsafe_allow_html=True)

    # 2.2 BOGEN
    elif "Bogen" in tool_mode:
        angle = st.slider("Winkel", 0, 90, 45, key="bogen_winkel")
        v = round(standard_radius * math.tan(math.radians(angle/2)), 1)
        st.markdown(f"<div class='result-card-green'>Vorbau: {v} mm</div>", unsafe_allow_html=True)

    # 2.3 STUTZEN
    elif "Stutzen" in tool_mode:
        c1, c2 = st.columns(2)
        dn1 = c1.selectbox("DN Stutzen", df_pipe['DN'], index=6)
        dn2 = c2.selectbox("DN Haupt", df_pipe['DN'], index=9)
        
        if dn1 > dn2:
            st.error("Stutzen größer als Hauptrohr!")
        else:
            rk = df_pipe[df_pipe['DN']==dn1].iloc[0]['D_Aussen']/2
            rg = df_pipe[df_pipe['DN']==dn2].iloc[0]['D_Aussen']/2
            st.pyplot(Visualizer.plot_stutzen_curve(rg, rk))

    # 2.4 ETAGE (3D)
    elif "Etage" in tool_mode:
        et_type = st.radio("Typ", ["2D", "Kastenmaß", "Fix-Winkel"], horizontal=True)
        c_c, c_v = st.columns([1, 1.5])
        
        with c_v:
            st.caption("📷 3D Ansicht")
            az = st.slider("H", 0, 360, get_val('view_azim'), key="_view_azim", on_change=save_val, args=('view_azim',))
            el = st.slider("V", 0, 90, get_val('view_elev'), key="_view_elev", on_change=save_val, args=('view_elev',))
        
        with c_c:
            if et_type == "2D":
                h = st.number_input("Höhe H", value=300)
                l = st.number_input("Länge L", value=400)
                t, a = GeometryEngine.solve_offset_3d(h, l, 0)
                st.markdown(f"<div class='result-card-green'>Säge: {round(t, 1)} mm</div>", unsafe_allow_html=True)
                st.pyplot(Visualizer.plot_true_3d_pipe(l, 0, h, az, el))
                
            elif et_type == "Kastenmaß":
                b = st.number_input("Breite B", value=200)
                h = st.number_input("Höhe H", value=300)
                l = st.number_input("Länge L", value=400)
                t, a = GeometryEngine.solve_offset_3d(h, l, b)
                st.markdown(f"<div class='result-card-green'>Säge: {round(t, 1)} mm</div>", unsafe_allow_html=True)
                st.pyplot(Visualizer.plot_true_3d_pipe(l, b, h, az, el))
                
            else:
                b = st.number_input("Breite B", value=200)
                h = st.number_input("Höhe H", value=300)
                w = st.selectbox("Winkel", [30, 45, 60])
                
                s = math.sqrt(b**2 + h**2)
                l_req = s / math.tan(math.radians(w))
                t = math.sqrt(l_req**2 + s**2)
                
                st.info(f"Länge L nötig: {round(l_req, 1)} mm")
                st.markdown(f"<div class='result-card-green'>Säge: {round(t, 1)} mm</div>", unsafe_allow_html=True)
                st.pyplot(Visualizer.plot_true_3d_pipe(l_req, b, h, az, el))

# -----------------------------------------------------------------------------
# TAB 3: ROHRBUCH
# -----------------------------------------------------------------------------
with tab_proj:
    with st.form("rb"):
        c1, c2, c3 = st.columns(3)
        iso = c1.text_input("ISO")
        naht = c2.text_input("Naht")
        datum = c3.date_input("Datum")
        c4, c5 = st.columns(2)
        dn_s = c4.selectbox("DN", df_pipe['DN'])
        len_s = c5.number_input("Länge", value=0)
        
        if st.form_submit_button("Speichern"):
            DatabaseRepository.add("rohrbuch", (iso, naht, datum.strftime("%d.%m.%Y"), f"DN {dn_s}", "Rohr", len_s, "-", "-"))
            st.rerun()
            
    st.dataframe(DatabaseRepository.get_all("rohrbuch"), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: KALKULATION (RESTORED V23.3 DETAIL LOGIC)
# -----------------------------------------------------------------------------
with tab_info:
    st.subheader("Kalkulation (Detail)")
    
    with st.expander("Globale Preise"):
        c1, c2, c3 = st.columns(3)
        st.session_state.store['p_lohn'] = c1.number_input("Lohn (€/h)", value=get_val('p_lohn'), key="_p_lohn", on_change=save_val, args=('p_lohn',))
        st.session_state.store['p_draht'] = c2.number_input("Draht (€/kg)", value=get_val('p_draht'), key="_p_draht", on_change=save_val, args=('p_draht',))
        st.session_state.store['p_gas'] = c3.number_input("Gas (€/L)", value=get_val('p_gas'), key="_p_gas", on_change=save_val, args=('p_gas',))
        
        c4, c5 = st.columns(2)
        st.session_state.store['p_stahl'] = c4.number_input("Scheibe Stahl (€)", value=get_val('p_stahl'), key="_p_stahl", on_change=save_val, args=('p_stahl',))
        st.session_state.store['p_dia'] = c5.number_input("Scheibe Dia (€)", value=get_val('p_dia'), key="_p_dia", on_change=save_val, args=('p_dia',))

    task = st.radio("Modul", ["Fügen (Schweißen)", "Trennen (Flexen)", "Montage"], horizontal=True)
    st.divider()
    
    # 4.1 FÜGEN
    if "Fügen" in task:
        c1, c2, c3 = st.columns(3)
        k_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('kw_dn')), key="_kw_dn", on_change=update_kw_dn)
        k_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('kw_ws')), key="_kw_ws", on_change=save_val, args=('kw_ws',))
        k_verf = c3.selectbox("Verfahren", ["WIG", "E-Hand (CEL 70)", "MAG"], index=get_verf_index(get_val('kw_verf')), key="_kw_verf", on_change=save_val, args=('kw_verf',))
        
        c4, c5 = st.columns(2)
        pers = c4.number_input("Personal", value=get_val('kw_pers'), key="_kw_pers", on_change=save_val, args=('kw_pers',))
        anz = c5.number_input("Anzahl Nähte", value=get_val('kw_anz'), key="_kw_anz", on_change=save_val, args=('kw_anz',))
        
        fac = st.slider("Erschwernis Faktor", 0.5, 2.0, 1.0)
        
        # Aufruf der detaillierten CostEngine
        dur_one, cost_total, mat_txt = CostEngine.calculate_welding_detailed(
            k_dn, k_ws, k_verf, pers, anz, fac,
            get_val('p_lohn'), get_val('p_machine'), get_val('p_draht'), get_val('p_cel'), get_val('p_gas')
        )
        
        m1, m2 = st.columns(2)
        m1.metric("Zeit Total", f"{int(dur_one * anz)} min")
        m2.metric("Kosten Total", f"{round(cost_total, 2)} €")
        st.caption(f"Verbrauch: {mat_txt}")
        
        if st.button("Hinzufügen"):
            DatabaseRepository.add("kalkulation", ("Fügen", f"DN {k_dn} {k_verf}", anz, dur_one*anz, cost_total, mat_txt))
            st.rerun()

    # 4.2 TRENNEN
    elif "Trennen" in task:
        c1, c2, c3 = st.columns(3)
        c_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('cut_dn')), key="_cut_dn", on_change=save_val, args=('cut_dn',))
        c_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('cut_ws')), key="_cut_ws", on_change=save_val, args=('cut_ws',))
        disc = c3.selectbox("Scheibe", ["125 mm", "230 mm"], index=get_disc_idx(get_val('cut_disc')), key="_cut_disc", on_change=save_val, args=('cut_disc',))
        
        c4, c5 = st.columns(2)
        anz = c4.number_input("Anzahl", value=1, key="cut_anz")
        zma = c5.checkbox("Beton (ZMA)?", key="cut_zma")
        
        # Aufruf CostEngine
        t_tot, c_tot, info = CostEngine.calculate_cutting_detailed(
            c_dn, c_ws, disc, anz, zma, False, 1.0,
            get_val('p_lohn'), get_val('p_stahl'), get_val('p_dia')
        )
        
        st.metric("Kosten", f"{round(c_tot, 2)} €")
        
        if st.button("Hinzufügen"):
            DatabaseRepository.add("kalkulation", ("Trennen", f"DN {c_dn} {disc}", anz, t_tot, c_tot, info))
            st.rerun()

    # LISTE
    st.divider()
    df_k = DatabaseRepository.get_all("kalkulation")
    if not df_k.empty:
        st.dataframe(df_k, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("Excel Export", export_excel(df_k), "kalk.xlsx")
        if c2.button("Reset"):
            DatabaseRepository.clear("kalkulation")
            st.rerun()

# -----------------------------------------------------------------------------
# TAB 5: LAGERHALTUNG
# -----------------------------------------------------------------------------
with tab_lager:
    st.subheader("📦 Lagerhaltung")
    
    c_l1, c_l2 = st.columns([1, 2])
    
    with c_l1:
        st.markdown("##### Artikel anlegen")
        with st.form("new_inv"):
            aid = st.text_input("Artikel ID")
            name = st.text_input("Name")
            pr = st.number_input("Preis", value=0.0)
            stk = st.number_input("Bestand", value=0)
            mn = st.number_input("Min", value=10)
            mx = st.number_input("Max", value=100)
            
            if st.form_submit_button("Speichern"):
                if DatabaseRepository.add_inventory(InventoryItem(aid, name, pr, stk, mn, mx)):
                    st.success("Gespeichert")
                else:
                    st.error("Fehler: ID existiert")
    
    with c_l2:
        st.markdown("##### Buchung")
        inv = DatabaseRepository.get_inventory()
        if inv:
            opts = {f"{x.article_id} {x.name}": x for x in inv}
            sel = st.selectbox("Artikel wählen", list(opts.keys()))
            amt = st.number_input("Menge", value=1)
            
            b1, b2 = st.columns(2)
            if b1.button("Eingang"):
                DatabaseRepository.update_inventory(opts[sel].article_id, opts[sel].current_stock + amt)
                st.rerun()
            if b2.button("Ausgang"):
                DatabaseRepository.update_inventory(opts[sel].article_id, max(0, opts[sel].current_stock - amt))
                st.rerun()
            
            # Tabelle
            data = []
            val = 0
            for i in inv:
                stat = "🟢" if i.current_stock > i.reorder_point else f"🔴 Order: {i.calculate_reorder_qty()}"
                data.append({"ID": i.article_id, "Name": i.name, "Bestand": i.current_stock, "Status": stat})
                val += i.stock_value
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            st.metric("Lagerwert", f"{val:,.2f} €")
