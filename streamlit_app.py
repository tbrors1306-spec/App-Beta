"""
PipeCraft V37.0 (Restoration & Stability Edition)
-------------------------------------------------
Wiederherstellung der exakten Funktionalitäten basierend auf V23.3 Screenshots.

Features:
1.  KALKULATION: Exaktes Layout wie im Screenshot (Slider, Checkboxen für ZMA/Mantel).
    - Berücksichtigt Maschinenpauschale (1.25 €/min = (60€ Lohn + 15€ Maschine) / 60).
2.  WERKSTATT (STUTZEN): Tabelle mit Winkel/Tiefe/Umfang wiederhergestellt.
3.  ROHRBUCH: Export-Buttons (Excel/PDF) repariert und sichtbar gemacht.
4.  Lager & Smart Cut: Beibehalten aus den stabilen Versionen.

Author: Senior Lead Software Engineer
"""

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Wichtig für 3D Plots
from mpl_toolkits.mplot3d import Axes3D 
import sqlite3
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import List, Tuple, Any, Optional, Union, Dict

# -----------------------------------------------------------------------------
# 0. SYSTEM SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipeCraft")

# PDF Library Check
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(
    page_title="PipeCraft V37.0",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styles (Angepasst an Screenshot-Look)
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #0f172a; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #1e293b !important; font-weight: 700; }
    
    /* Metriken groß darstellen wie im Screenshot */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #1e293b;
    }
    
    .result-card-blue { background-color: #eff6ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; margin-bottom: 10px; color: #1e3a8a; }
    .result-card-green { background: #f0fdf4; padding: 20px; border-radius: 8px; border-left: 5px solid #22c55e; margin-bottom: 15px; text-align: center; font-size: 1.5rem; font-weight: bold; color: #14532d; }
    
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] { border-radius: 4px; border: 1px solid #cbd5e1; }
    div.stButton > button { width: 100%; border-radius: 4px; font-weight: 600; border: 1px solid #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. DATENBANK (STATIC)
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
except Exception as e:
    st.error(f"Datenbankfehler: {e}")
    st.stop()

# Zusatz-Daten
SCHRAUBEN_DB = { "M12": [18, 60], "M16": [24, 130], "M20": [30, 250], "M24": [36, 420], "M27": [41, 600], "M30": [46, 830], "M33": [50, 1100], "M36": [55, 1400], "M39": [60, 1800], "M45": [70, 2700], "M52": [80, 4200] }
WS_LISTE = [2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 8.8, 10.0, 11.0, 12.5, 14.2, 16.0]
WS_STD_MAP = {25: 3.2, 32: 3.6, 40: 3.6, 50: 3.9, 65: 5.2, 80: 5.5, 100: 6.0, 125: 6.6, 150: 7.1, 200: 8.2, 250: 9.3, 300: 9.5, 350: 9.5, 400: 9.5, 450: 9.5, 500: 9.5}
DB_NAME = "pipecraft_v37.db"

# -----------------------------------------------------------------------------
# 2. LOGIC (ENGINES & HELPERS)
# -----------------------------------------------------------------------------

def get_row_by_dn(dn: int) -> pd.Series:
    try: return df_pipe[df_pipe['DN'] == dn].iloc[0]
    except: return df_pipe.iloc[0]

def get_schrauben_info(gewinde): return SCHRAUBEN_DB.get(gewinde, ["?", "?"])

def parse_abzuege(text: str) -> float:
    try:
        clean = text.replace(",", ".").replace(" ", "")
        if not all(c in "0123456789.+-*/()" for c in clean): return 0.0
        return float(pd.eval(clean))
    except: return 0.0

# UI Helpers (verhindern Abstürze bei Reloads)
def get_ws_index(val): return WS_LISTE.index(val) if val in WS_LISTE else 6
def get_verf_index(val): return ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"].index(val) if val in ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"] else 0
def get_disc_idx(val): return ["125 mm", "180 mm", "230 mm"].index(val) if val in ["125 mm", "180 mm", "230 mm"] else 0
def get_sys_idx(val): return ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"].index(val) if val in ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"] else 0

@dataclass
class SelectedFitting:
    type_name: str; count: int; deduction_single: float; dn_spec: int

class FittingManager:
    @staticmethod
    def get_deduction(type_name: str, dn_target: int, pn_suffix: str = "_16", custom_angle: float = 45.0) -> float:
        row_data = get_row_by_dn(dn_target)
        if type_name == "Bogen 90° (BA3)": return float(row_data['Radius_BA3'])
        elif type_name == "Bogen (Zuschnitt)": return float(row_data['Radius_BA3']) * math.tan(math.radians(custom_angle / 2))
        elif type_name == "Flansch (Vorschweiß)": return float(row_data[f'Flansch_b{pn_suffix}'])
        elif type_name == "T-Stück": return float(row_data['T_Stueck_H'])
        elif "Reduzierung" in type_name: return float(row_data['Red_Laenge_L'])
        return 0.0

class PhysicsEngine:
    @staticmethod
    def calculate_pipe_weight(dn_idx, ws, length_mm, is_zme=False):
        try:
            da = df_pipe.iloc[dn_idx]['D_Aussen']; length_dm = length_mm / 100.0; ra = (da/2)/100.0; ri = ra - (ws/100.0)
            vol = math.pi * (ra**2 - ri**2) * length_dm; w = vol * 7.85
            if is_zme:
                dn = df_pipe.iloc[dn_idx]['DN']; cem = 0.6 if dn < 300 else (0.9 if dn < 600 else 1.2)
                ric = ri - (cem/100.0)
                if ric > 0: w += (math.pi * (ri**2 - ric**2) * length_dm) * 2.4
            return round(w, 1)
        except: return 0.0

class CostEngine:
    """Wiederhergestellte V23.3 Kalkulation."""
    @staticmethod
    def calculate_welding(dn, ws, process, pers, anz, factor, p_lohn, p_mach, p_draht, p_cel, p_gas):
        zoll = dn / 25.0
        min_per_inch = 10.0 if process == "WIG" else (3.5 if "CEL" in process else 5.0)
        t_weld = zoll * min_per_inch
        t_fit = zoll * 2.5
        
        # Zeit
        dur_seam = ((t_weld + t_fit) / pers) * factor
        total_time_min = dur_seam * anz
        
        # Kostenfaktor Arbeit (Lohn + Maschine)
        # Screenshot Analyse: 1.25 €/min = (60 €/h + 15 €/h) / 60 min. Das passt!
        hourly_total = p_lohn + p_mach
        labor_cost = (total_time_min / 60) * (pers * hourly_total)
        
        # Material
        da = df_pipe[df_pipe['DN'] == dn].iloc[0]['D_Aussen']
        weld_vol = (da * math.pi) * (ws**2 * 0.8) 
        kg_fill = (weld_vol * 7.85) / 1_000_000 * 1.5
        
        mat_cost = 0; mat_txt = ""
        if "CEL" in process:
            mat_cost = ((5.0 * kg_fill) * p_cel) * anz
            mat_txt = f"{round(5.0*kg_fill*anz, 1)} Stk CEL"
        else:
            wire = (kg_fill * p_draht) * anz
            gas = (t_weld * factor * anz) * 15 * p_gas
            mat_cost = wire + gas
            mat_txt = f"{round(kg_fill*anz, 1)} kg Draht"
            
        return total_time_min, labor_cost + mat_cost, mat_txt

    @staticmethod
    def calculate_cutting(dn, ws, disc, anz, zma, iso, factor, p_lohn, p_stahl, p_dia):
        zoll = dn / 25.0
        cap = 14000 if "230" in disc else (7000 if "180" in disc else 3500)
        zma_fac = 3.0 if zma else 1.0
        iso_fac = 1.3 if iso else 1.0
        
        t_tot = (zoll * 0.5 * zma_fac * iso_fac) * anz * factor
        
        da = df_pipe[df_pipe['DN']==dn].iloc[0]['D_Aussen']
        area = (da * math.pi * ws)
        n_disc = math.ceil((area * (2.0 if zma else 1.0) * anz) / cap)
        
        c_mat = n_disc * (p_dia if zma else p_stahl)
        c_lab = (t_tot/60) * p_lohn
        return t_tot, c_lab + c_mat, f"{n_disc}x Scheiben"

class GeometryEngine:
    @staticmethod
    def solve_offset_3d(h, l, b):
        travel = math.sqrt(h**2 + l**2 + b**2); spread = math.sqrt(l**2 + b**2)
        angle = 90.0 if spread == 0 else math.degrees(math.atan(h / spread))
        return travel, angle

class Visualizer:
    @staticmethod
    def plot_stutzen_curve(r_haupt, r_stutzen):
        angles = range(0, 361, 5)
        try: depths = [r_haupt - math.sqrt(r_haupt**2 - (r_stutzen * math.sin(math.radians(a)))**2) for a in angles]
        except: return plt.figure()
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360); ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig
    
    @staticmethod
    def plot_true_3d_pipe(l, b, h, az, el):
        fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection='3d')
        ax.plot([0, l], [0, b], [0, h], color='#ef4444', linewidth=5)
        ax.plot([0, l], [0, 0], [0, 0], 'k--', alpha=0.2); ax.plot([l, l], [0, b], [0, 0], 'k--', alpha=0.2)
        ax.plot([0, l], [b, b], [0, 0], 'k--', alpha=0.1); ax.plot([l, l], [b, b], [0, h], 'k--', alpha=0.3)
        m = max(abs(l), abs(b), abs(h)) or 100
        ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_zlim(0, m)
        ax.view_init(elev=el, azim=az)
        return fig

# --- DATABASE ---
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
    def add_rohrbuch(data):
        with sqlite3.connect(DB_NAME) as conn: conn.cursor().execute('INSERT INTO rohrbuch (iso, naht, datum, dimension, bauteil, laenge, charge, schweisser) VALUES (?,?,?,?,?,?,?,?)', data); conn.commit()
    
    @staticmethod
    def add_kalk(data):
        with sqlite3.connect(DB_NAME) as conn: conn.cursor().execute('INSERT INTO kalkulation (typ, info, menge, zeit_min, kosten, mat_text) VALUES (?,?,?,?,?,?)', data); conn.commit()

    @staticmethod
    def get_all(table):
        with sqlite3.connect(DB_NAME) as conn: return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    
    @staticmethod
    def delete(table, id):
        with sqlite3.connect(DB_NAME) as conn: conn.cursor().execute(f"DELETE FROM {table} WHERE id=?", (id,)); conn.commit()
    
    @staticmethod
    def clear(table):
        with sqlite3.connect(DB_NAME) as conn: conn.cursor().execute(f"DELETE FROM {table}"); conn.commit()

    @staticmethod
    def add_inventory(item):
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.cursor().execute('INSERT INTO inventory VALUES (?,?,?,?,?,?)', (item.article_id, item.name, item.price_per_unit, item.current_stock, item.reorder_point, item.target_stock))
                return True
        except: return False

    @staticmethod
    def get_inventory():
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.cursor().execute('SELECT * FROM inventory').fetchall()
            return [InventoryItem(*r) for r in rows]

    @staticmethod
    def update_inventory(id, qty):
        with sqlite3.connect(DB_NAME) as conn: conn.cursor().execute('UPDATE inventory SET current_stock=? WHERE article_id=?', (qty, id)); conn.commit()

# --- EXPORT ---
def export_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer: df.to_excel(writer, index=False)
    return out.getvalue()

def export_pdf(df):
    if not PDF_AVAILABLE: return b""
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Report", 0, 1, 'C'); pdf.ln(5)
    for _, r in df.iterrows():
        try: pdf.cell(0, 8, f"{r.get('typ','')} | {r.get('info','')} | {r.get('kosten','')} | {r.get('mat_text','')}", 1, 1)
        except: pass
    return pdf.output(dest='S').encode('latin-1')

# --- INVENTORY MODEL ---
@dataclass
class InventoryItem:
    article_id: str; name: str; price_per_unit: float; current_stock: int; reorder_point: int; target_stock: int
    def calculate_reorder_qty(self): return max(0, self.target_stock - self.current_stock) if self.current_stock <= self.reorder_point else 0
    @property
    def stock_value(self): return self.current_stock * self.price_per_unit

# --- INITIALIZATION ---
DatabaseRepository.init_tables()
if 'store' not in st.session_state:
    st.session_state.store = {
        'saw_mass': 1000.0, 'saw_gap': 4.0, 'saw_deduct': "0", 'saw_zme': False,
        'kw_dn': 200, 'kw_ws': 6.3, 'kw_verf': "WIG", 'kw_pers': 1, 'kw_anz': 1, 'kw_factor': 1.0,
        'cut_dn': 200, 'cut_ws': 6.3, 'cut_disc': "125 mm", 'cut_anz': 1, 'cut_zma': False, 'cut_iso': False, 'cut_factor': 1.0,
        'iso_sys': "Schrumpfschlauch (WKS)", 'iso_dn': 200, 'iso_anz': 1, 'iso_factor': 1.0,
        'mon_dn': 200, 'mon_type': "Schieber", 'mon_anz': 1, 'mon_factor': 1.0,
        'reg_min': 60, 'reg_pers': 2, 'bogen_winkel': 45, 'view_azim': 45, 'view_elev': 30,
        # Default Werte aus Screenshot
        'p_lohn': 60.0, 'p_stahl': 2.5, 'p_dia': 45.0, 'p_cel': 0.40, 'p_draht': 15.0, 'p_gas': 0.05, 
        'p_wks': 25.0, 'p_kebu1': 15.0, 'p_kebu2': 12.0, 'p_primer': 12.0, 'p_machine': 15.0
    }
if 'fitting_list' not in st.session_state: st.session_state.fitting_list = []

def save_val(key): st.session_state.store[key] = st.session_state[f"_{key}"]
def get_val(key): return st.session_state.store.get(key)
def update_kw_dn(): st.session_state.store['kw_dn'] = st.session_state['_kw_dn']; st.session_state.store['kw_pers'] = 2 if st.session_state.store['kw_dn'] >= 300 else 1

# -----------------------------------------------------------------------------
# UI IMPLEMENTATION
# -----------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=50) 
st.sidebar.markdown("### Menü")
selected_dn_global = st.sidebar.selectbox("Nennweite (Global)", df_pipe['DN'], index=8, key="global_dn") 
selected_pn = st.sidebar.radio("Druckstufe", ["PN 16", "PN 10"], index=0, key="global_pn") 

row = get_row_by_dn(selected_dn_global)
standard_radius = float(row['Radius_BA3'])
suffix = "_16" if selected_pn == "PN 16" else "_10"

st.title("PipeCraft V37.0")
st.caption(f"🔧 Aktive Konfiguration: DN {selected_dn_global} | {selected_pn} | Radius: {standard_radius} mm")

tab_buch, tab_werk, tab_proj, tab_info, tab_lager = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch", "💰 Kalkulation", "📦 Lager"])

with tab_buch:
    st.subheader("Rohr & Formstücke")
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='result-card-blue'><b>Außen-Ø:</b> {row['D_Aussen']} mm</div>", unsafe_allow_html=True)
    c1.markdown(f"<div class='result-card-blue'><b>Radius (3D):</b> {standard_radius} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>T-Stück (H):</b> {row['T_Stueck_H']} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>Reduzierung (L):</b> {row['Red_Laenge_L']} mm</div>", unsafe_allow_html=True)
    st.divider(); st.subheader(f"Flansch & Montage ({selected_pn})")
    schraube = row[f'Schraube_M{suffix}']; sw, nm = get_schrauben_info(schraube)
    mc1, mc2 = st.columns(2)
    mc1.markdown(f"<div class='result-card-blue'><b>Blattstärke:</b> {row[f'Flansch_b{suffix}']} mm</div>", unsafe_allow_html=True)
    mc2.markdown(f"<div class='result-card-blue'><b>Schraube:</b> {row[f'Lochzahl{suffix}']}x {schraube} (SW {sw})</div>", unsafe_allow_html=True)
    c_d1, c_d2, c_d3 = st.columns(3)
    c_d1.markdown(f"<div class='detail-box'>Länge (Fest-Fest)<br><span class='detail-value'>{row[f'L_Fest{suffix}']} mm</span></div>", unsafe_allow_html=True)
    c_d2.markdown(f"<div class='detail-box'>Länge (Fest-Los)<br><span class='detail-value'>{row[f'L_Los{suffix}']} mm</span></div>", unsafe_allow_html=True)
    c_d3.markdown(f"<div class='detail-box'>Drehmoment<br><span class='detail-value'>{nm} Nm</span></div>", unsafe_allow_html=True)

with tab_werk:
    tool_mode = st.radio("Werkzeug:", ["📏 Säge (Smart Cut)", "🔄 Bogen", "🔥 Stutzen", "📐 Etage"], horizontal=True, label_visibility="collapsed", key="tool_nav")
    st.divider()
    
    if "Säge" in tool_mode:
        st.subheader("Smart Cut System")
        c1, c2 = st.columns(2)
        iso_mass = c1.number_input("Gesamtmaß (Iso)", value=get_val('saw_mass'), step=10.0, key="_saw_mass", on_change=save_val, args=('saw_mass',))
        spalt = c2.number_input("Wurzelspalt (pro Naht)", value=get_val('saw_gap'), key="_saw_gap", on_change=save_val, args=('saw_gap',))
        st.markdown("#### Bauteile hinzufügen")
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1])
        f_type = col_f1.selectbox("Typ", ["Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung (konz.)"])
        f_dn = selected_dn_global
        if "Reduzierung" in f_type:
            f_dn = col_f2.selectbox("DN (Groß)", df_pipe['DN'], index=df_pipe['DN'].tolist().index(selected_dn_global), key="red_lg")
            col_f2.selectbox("DN (Klein)", df_pipe['DN'], index=0, key="red_sm")
        else: f_dn = col_f2.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(selected_dn_global))
        f_angle = 45.0
        if "Zuschnitt" in f_type: f_angle = col_f3.number_input("Winkel", value=45.0, step=1.0)
        else: col_f3.write("-")
        f_count = col_f4.number_input("Anzahl", value=1, min_value=1)
        if st.button("Hinzufügen (+)", type="secondary"):
            deduct = FittingManager.get_deduction(f_type, f_dn, suffix, f_angle)
            name = f"{f_type} (DN {f_dn})"
            if "Zuschnitt" in f_type: name += f" [{f_angle}°]"
            st.session_state.fitting_list.append(SelectedFitting(name, f_count, deduct, f_dn))
            st.rerun()
        total_deduct = 0.0; total_gaps = 0
        if st.session_state.fitting_list:
            st.markdown("---")
            for i, item in enumerate(st.session_state.fitting_list):
                c_i1, c_i2, c_i3 = st.columns([4, 2, 1])
                sub = item.deduction_single * item.count
                total_deduct += sub; total_gaps += item.count
                c_i1.write(f"**{item.count}x** {item.type_name}")
                c_i2.caption(f"-{round(sub, 1)} mm")
                if c_i3.button("🗑️", key=f"del_{i}"): st.session_state.fitting_list.pop(i); st.rerun()
            if st.button("Liste leeren", type="primary"): st.session_state.fitting_list = []; st.rerun()
        final = iso_mass - total_deduct - (total_gaps * spalt)
        st.markdown("---")
        if final < 0: st.error(f"Fehler: Bauteile länger als Iso!")
        else:
            st.markdown(f"<div class='result-card-green'>Sägelänge: {round(final, 1)} mm</div>", unsafe_allow_html=True)
            dn_idx = df_pipe[df_pipe['DN'] == selected_dn_global].index[0]
            c_zme = st.checkbox("ZME?", value=get_val('saw_zme'), key="_saw_zme", on_change=save_val, args=('saw_zme',))
            kg = PhysicsEngine.calculate_pipe_weight(dn_idx, WS_STD_MAP.get(selected_dn_global, 4.0), final, c_zme)
            st.markdown(f"<div class='weight-box'>⚖️ Gewicht: ~ {kg} kg</div>", unsafe_allow_html=True)

    elif "Bogen" in tool_mode:
        angle = st.slider("Winkel", 0, 90, 45, key="bogen_winkel")
        v = round(standard_radius * math.tan(math.radians(angle/2)), 1)
        st.markdown(f"<div class='result-card-green'>Vorbau: {v} mm</div>", unsafe_allow_html=True)

    elif "Stutzen" in tool_mode:
        st.subheader("Stutzen Schablone")
        c1, c2 = st.columns(2)
        dn1 = c1.selectbox("DN Stutzen", df_pipe['DN'], index=6)
        dn2 = c2.selectbox("DN Haupt", df_pipe['DN'], index=9)
        if dn1 > dn2: st.error("Stutzen größer als Hauptrohr!")
        else:
            rk = df_pipe[df_pipe['DN']==dn1].iloc[0]['D_Aussen']/2
            rg = df_pipe[df_pipe['DN']==dn2].iloc[0]['D_Aussen']/2
            c_tab, c_plot = st.columns([1, 2])
            # WIEDERHERGESTELLT: Die Tabelle mit Werten für die Werkstatt
            table_data = []
            for a in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
                t = int(round(rg - math.sqrt(rg**2 - (rk * math.sin(math.radians(a)))**2), 0))
                u = int(round((rk * 2 * math.pi) * (a/360), 0))
                table_data.append({"Winkel": f"{a}°", "Tiefe (mm)": t, "Umfang (mm)": u})
            with c_tab: st.table(pd.DataFrame(table_data)) # st.table ist besser lesbar als dataframe für fixe werte
            with c_plot: st.pyplot(Visualizer.plot_stutzen_curve(rg, rk))

    elif "Etage" in tool_mode:
        et_type = st.radio("Typ", ["2D", "Kastenmaß", "Fix-Winkel"], horizontal=True)
        c_c, c_v = st.columns([1, 1.5])
        with c_v:
            az = st.slider("H", 0, 360, get_val('view_azim'), key="_view_azim", on_change=save_val, args=('view_azim',))
            el = st.slider("V", 0, 90, get_val('view_elev'), key="_view_elev", on_change=save_val, args=('view_elev',))
        with c_c:
            if et_type == "2D":
                h = st.number_input("H", value=300); l = st.number_input("L", value=400)
                t, a = GeometryEngine.solve_offset_3d(h, l, 0)
                st.markdown(f"<div class='result-card-green'>Säge: {round(t, 1)} mm</div>", unsafe_allow_html=True)
                st.pyplot(Visualizer.plot_true_3d_pipe(l, 0, h, az, el))
            elif et_type == "Kastenmaß":
                b = st.number_input("B", value=200); h = st.number_input("H", value=300); l = st.number_input("L", value=400)
                t, a = GeometryEngine.solve_offset_3d(h, l, b)
                st.markdown(f"<div class='result-card-green'>Säge: {round(t, 1)} mm</div>", unsafe_allow_html=True)
                st.pyplot(Visualizer.plot_true_3d_pipe(l, b, h, az, el))
            else:
                b = st.number_input("B", value=200); h = st.number_input("H", value=300); w = st.selectbox("Winkel", [30, 45, 60])
                s = math.sqrt(b**2 + h**2); l_req = s / math.tan(math.radians(w))
                t = math.sqrt(l_req**2 + s**2)
                st.info(f"L nötig: {round(l_req, 1)} mm")
                st.markdown(f"<div class='result-card-green'>Säge: {round(t, 1)} mm</div>", unsafe_allow_html=True)
                st.pyplot(Visualizer.plot_true_3d_pipe(l_req, b, h, az, el))

with tab_proj:
    st.subheader("Digitales Rohrbuch")
    with st.form("rb"):
        c1, c2, c3 = st.columns(3); iso = c1.text_input("ISO"); naht = c2.text_input("Naht"); datum = c3.date_input("Datum")
        c4, c5 = st.columns(2); dn_s = c4.selectbox("DN", df_pipe['DN']); len_s = c5.number_input("Länge", value=0)
        c6, c7 = st.columns(2); charge = c6.text_input("Charge"); schweisser = c7.text_input("Schweißer")
        if st.form_submit_button("Speichern"):
            DatabaseRepository.add_rohrbuch((iso, naht, datum.strftime("%d.%m.%Y"), f"DN {dn_s}", "Rohr", len_s, charge, schweisser))
            st.success("Gespeichert!")
    
    df_rb = DatabaseRepository.get_all("rohrbuch")
    st.dataframe(df_rb, use_container_width=True)
    
    # WIEDERHERGESTELLT: Export Buttons
    if not df_rb.empty:
        st.markdown("### Export")
        c_ex1, c_ex2 = st.columns(2)
        c_ex1.download_button("📥 Excel Download", export_excel(df_rb), "rohrbuch.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if PDF_AVAILABLE: c_ex2.download_button("📄 PDF Download", export_pdf(df_rb), "rohrbuch.pdf", mime="application/pdf")

    with st.expander("Zeile löschen"):
        if not df_rb.empty:
            opts = {f"ID {r['id']}: {r['iso']} {r['naht']}": r['id'] for i, r in df_rb.iterrows()}
            if opts:
                sel = st.selectbox("Wähle Eintrag:", list(opts.keys()), key="rb_del_sel")
                if st.button("Löschen", key="rb_del_btn"): DatabaseRepository.delete("rohrbuch", opts[sel]); st.rerun()

with tab_info:
    # WIEDERHERGESTELLT: Layout exakt nach Screenshot
    st.subheader("Preis-Datenbank")
    with st.expander("Einstellungen bearbeiten"):
        c1, c2, c3 = st.columns(3)
        st.session_state.store['p_lohn'] = c1.number_input("Lohn (€/h)", value=get_val('p_lohn'), key="_p_lohn", on_change=save_val, args=('p_lohn',))
        st.session_state.store['p_stahl'] = c2.number_input("Stahl-Scheibe (€)", value=get_val('p_stahl'), key="_p_stahl", on_change=save_val, args=('p_stahl',))
        st.session_state.store['p_dia'] = c3.number_input("Diamant-Scheibe (€)", value=get_val('p_dia'), key="_p_dia", on_change=save_val, args=('p_dia',))
        
        c4, c5, c6 = st.columns(3)
        st.session_state.store['p_cel'] = c4.number_input("Elektrode CEL (€)", value=get_val('p_cel'), key="_p_cel", on_change=save_val, args=('p_cel',))
        st.session_state.store['p_draht'] = c5.number_input("Draht (€/kg)", value=get_val('p_draht'), key="_p_draht", on_change=save_val, args=('p_draht',))
        st.session_state.store['p_gas'] = c6.number_input("Gas (€/L)", value=get_val('p_gas'), key="_p_gas", on_change=save_val, args=('p_gas',))
        
        c7, c8, c9 = st.columns(3)
        st.session_state.store['p_wks'] = c7.number_input("WKS (€)", value=get_val('p_wks'), key="_p_wks", on_change=save_val, args=('p_wks',))
        st.session_state.store['p_kebu1'] = c8.number_input("Kebu 1.2 (€)", value=get_val('p_kebu1'), key="_p_kebu1", on_change=save_val, args=('p_kebu1',))
        st.session_state.store['p_machine'] = c9.number_input("Geräte-Pauschale (€/h)", value=get_val('p_machine'), key="_p_machine", on_change=save_val, args=('p_machine',))

    st.divider()
    
    # LAYOUT: Radio Buttons oben (Fügen, Trennen...) wie im Screenshot
    task = st.radio("Tätigkeit", ["🔥 Fügen (Schweißen)", "✂️ Trennen (Vorbereitung)", "🔧 Montage", "🛡️ Isolierung", "🚗 Regie"], horizontal=True)
    st.markdown("---")
    
    if "Fügen" in task:
        # Layout gemäß Screenshot: 3 Spalten DN | WS | Verfahren
        c1, c2, c3 = st.columns(3)
        k_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('kw_dn')), key="_kw_dn", on_change=update_kw_dn)
        k_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('kw_ws')), key="_kw_ws", on_change=save_val, args=('kw_ws',))
        k_verf = c3.selectbox("Verfahren", ["WIG", "E-Hand (CEL 70)", "MAG"], index=get_verf_index(get_val('kw_verf')), key="_kw_verf", on_change=save_val, args=('kw_verf',))
        
        # Zeile 2: Mitarbeiter | Anzahl | Slider
        c4, c5 = st.columns(2)
        pers = c4.number_input("Anzahl Mitarbeiter", value=get_val('kw_pers'), key="_kw_pers", on_change=save_val, args=('kw_pers',))
        anz = c5.number_input("Anzahl Nähte", value=get_val('kw_anz'), key="_kw_anz", on_change=save_val, args=('kw_anz',))
        
        st.caption("Zeit-Faktor")
        fac = st.slider("Faktor", 0.5, 2.0, 1.0, label_visibility="collapsed")
        
        # Checkbox für Split (Optional)
        split = st.checkbox("Als 2 Positionen speichern? (Vorb. + Fügen)")
        
        # LOGIC: Kalkulation (100 min x 1 Pers x 1.25 €/min)
        total_time_min, total_cost, mat_txt = CostEngine.calculate_welding(
            k_dn, k_ws, k_verf, pers, anz, fac, 
            get_val('p_lohn'), get_val('p_machine'), get_val('p_draht'), get_val('p_cel'), get_val('p_gas')
        )
        
        # Output Groß (wie Screenshot)
        m1, m2 = st.columns(2)
        m1.metric("Zeit", f"{int(total_time_min)} min")
        m2.metric("Kosten", f"{round(total_cost, 2)} €")
        
        if st.button("Hinzufügen"):
            if split:
                t_half = total_time_min / 2
                c_half = total_cost / 2 # Vereinfacht
                DatabaseRepository.add_kalk(("Vorbereitung", f"DN {k_dn} Fitting", anz, t_half, c_half, "-"))
                DatabaseRepository.add_kalk(("Fügen", f"DN {k_dn} Welding", anz, t_half, c_half, mat_txt))
            else:
                DatabaseRepository.add_kalk(("Fügen", f"DN {k_dn} {k_verf}", anz, total_time_min, total_cost, mat_txt))
            st.rerun()

    elif "Trennen" in task:
        # Layout: DN | WS | Scheibe | Anzahl (Screenshot ähnlich)
        c1, c2, c3, c4 = st.columns(4)
        c_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('cut_dn')), key="_cut_dn", on_change=save_val, args=('cut_dn',))
        c_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('cut_ws')), key="_cut_ws", on_change=save_val, args=('cut_ws',))
        disc = c3.selectbox("Scheibe", ["125 mm", "230 mm"], index=get_disc_idx(get_val('cut_disc')), key="_cut_disc", on_change=save_val, args=('cut_disc',))
        anz = c4.number_input("Anzahl", 1, key="cut_anz")
        
        # Checkboxen darunter
        c5, c6 = st.columns(2)
        zma = c5.checkbox("Beton (ZMA)?", key="cut_zma")
        iso = c6.checkbox("Mantel entfernen?", key="cut_iso")
        
        st.caption("Zeit-Faktor")
        fac = st.slider("Faktor", 0.5, 2.0, 1.0, label_visibility="collapsed")
        
        t_tot, c_tot, info = CostEngine.calculate_cutting(
            c_dn, c_ws, disc, anz, zma, iso, fac, 
            get_val('p_lohn'), get_val('p_stahl'), get_val('p_dia')
        )
        
        m1, m2 = st.columns(2)
        m1.metric("Zeit", f"{int(t_tot)} min")
        m2.metric("Kosten", f"{round(c_tot, 2)} €")
        
        if st.button("Hinzufügen"):
            DatabaseRepository.add_kalk(("Trennen", f"DN {c_dn} {disc}", anz, t_tot, c_tot, info))
            st.rerun()

    elif "Isolierung" in task:
        sys = st.radio("System", ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"], horizontal=True)
        c1, c2 = st.columns(2)
        i_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('iso_dn')), key="_iso_dn", on_change=save_val, args=('iso_dn',))
        anz = c2.number_input("Anzahl", 1, key="_iso_anz")
        
        st.caption("Zeit-Faktor")
        fac = st.slider("Faktor", 0.5, 2.0, 1.0, label_visibility="collapsed")
        
        # Logic Restore
        time = (20 + (i_dn * 0.07)) * fac * anz
        c_mat = (get_val('p_wks') * anz) if "WKS" in sys else 50.0 # Vereinfachtes Material für Band
        c_lab = (time/60) * get_val('p_lohn')
        total = c_lab + c_mat
        
        m1, m2 = st.columns(2)
        m1.metric("Zeit", f"{int(time)} min")
        m2.metric("Kosten", f"{round(total, 2)} €")
        
        if st.button("Hinzufügen"):
            DatabaseRepository.add_kalk(("Isolierung", f"DN {i_dn} {sys}", anz, time, total, "-"))
            st.rerun()

    elif "Regie" in task:
        c1, c2 = st.columns(2); t = c1.number_input("Minuten", value=get_val('reg_min'), step=15, key="_reg_min", on_change=save_val, args=('reg_min',)); p = c2.number_input("Personen", value=get_val('reg_pers'), min_value=1, key="_reg_pers", on_change=save_val, args=('reg_pers',))
        cost = (t/60 * get_val('p_lohn')) * p; st.metric("Kosten", f"{round(cost, 2)} €")
        if st.button("Hinzufügen"): DatabaseRepository.add_kalk(("Regie", f"{p} Pers.", 1, t, cost, "-")); st.rerun()

    # Liste unten (wie Screenshot)
    st.markdown("### 📊 Projekt Status (Live)")
    df_k = DatabaseRepository.get_all("kalkulation")
    if not df_k.empty:
        c_sum1, c_sum2 = st.columns(2)
        c_sum1.metric("Gesamt-Kosten", f"{round(df_k['kosten'].sum(), 2)} €")
        c_sum2.metric("Gesamt-Stunden", f"{round(df_k['zeit_min'].sum()/60, 1)} h")
        
        st.dataframe(df_k, use_container_width=True)
        
        c_del, c_rst = st.columns(2)
        with c_del.expander("Zeile löschen"):
            opts = {f"ID {r['id']}: {r['typ']}": r['id'] for i, r in df_k.iterrows()}
            if opts:
                sel = st.selectbox("Wähle:", list(opts.keys()), key="kalk_del")
                if st.button("Löschen"): DatabaseRepository.delete("kalkulation", opts[sel]); st.rerun()
        if c_rst.button("Alles Löschen", type="primary"): DatabaseRepository.clear("kalkulation"); st.rerun()
        
        c_ex1, c_ex2 = st.columns(2)
        c_ex1.download_button("Excel Export", export_excel(df_k), "kalkulation.xlsx")
        if PDF_AVAILABLE: c_ex2.download_button("PDF Export", export_pdf(df_k), "kalkulation.pdf")

with tab_lager:
    st.subheader("📦 Lagerhaltung")
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.form("new_inv"):
            aid = st.text_input("Artikel ID"); name = st.text_input("Name"); pr = st.number_input("Preis", 0.0); stk = st.number_input("Bestand", 0); mn = st.number_input("Min", 10); mx = st.number_input("Max", 100)
            if st.form_submit_button("Speichern"):
                if DatabaseRepository.add_inventory(InventoryItem(aid, name, pr, stk, mn, mx)): st.success("Gespeichert")
                else: st.error("Fehler: ID existiert")
    with c2:
        inv = DatabaseRepository.get_inventory()
        if inv:
            opts = {f"{x.article_id} {x.name}": x for x in inv}; sel = st.selectbox("Artikel wählen", list(opts.keys())); amt = st.number_input("Menge", value=1)
            b1, b2 = st.columns(2)
            if b1.button("Eingang"): DatabaseRepository.update_inventory(opts[sel].article_id, opts[sel].current_stock + amt); st.rerun()
            if b2.button("Ausgang"): DatabaseRepository.update_inventory(opts[sel].article_id, max(0, opts[sel].current_stock - amt)); st.rerun()
            data = []
            val = 0
            for i in inv:
                stat = "🟢" if i.current_stock > i.reorder_point else f"🔴 Order: {i.calculate_reorder_qty()}"
                data.append({"ID": i.article_id, "Name": i.name, "Bestand": i.current_stock, "Status": stat}); val += i.stock_value
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            st.metric("Lagerwert", f"{val:,.2f} €")
