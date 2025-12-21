"""
PipeCraft Enterprise Edition (V24.1 - Hotfix)
---------------------------------------------
Fix: Re-added missing UI helper functions for index retrieval.
"""

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sqlite3
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime
from io import BytesIO
import logging

# Logging Konfiguration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optionale Importe
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("FPDF library not found. PDF export will be disabled.")

# -----------------------------------------------------------------------------
# 1. KONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PipeCraft V24.1", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #0f172a; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #1e293b !important; }
    .result-card-blue { background-color: #eff6ff; padding: 20px; border-radius: 12px; border-left: 6px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px; color: #1e3a8a; font-size: 1rem; }
    .result-card-green { background: linear-gradient(to right, #f0fdf4, #ffffff); padding: 25px; border-radius: 12px; border-left: 8px solid #22c55e; box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-bottom: 15px; text-align: center; font-size: 1.5rem; font-weight: 800; color: #14532d; }
    .detail-box { background-color: #f1f5f9; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; font-size: 0.9rem; color: #334155; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .weight-box { background-color: #fff1f2; border: 1px solid #fecdd3; color: #881337; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; margin-top: 10px; }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] { border-radius: 8px; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATEN-SCHICHT
# -----------------------------------------------------------------------------
PIPE_DATA = {
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

# DataFrame und Listen
try:
    df_pipe = pd.DataFrame(PIPE_DATA)
except ValueError as e:
    st.error("Kritischer Fehler: Datenbank inkonsistent.")
    st.stop()

SCHRAUBEN_DB = {
    "M12": [18, 60], "M16": [24, 130], "M20": [30, 250], "M24": [36, 420],
    "M27": [41, 600], "M30": [46, 830], "M33": [50, 1100], "M36": [55, 1400],
    "M39": [60, 1800], "M45": [70, 2700], "M52": [80, 4200]
}
WS_LISTE = [2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 8.8, 10.0, 11.0, 12.5, 14.2, 16.0]
WS_STD = {25: 3.2, 32: 3.6, 40: 3.6, 50: 3.9, 65: 5.2, 80: 5.5, 100: 6.0, 125: 6.6, 150: 7.1, 200: 8.2, 250: 9.3, 300: 9.5, 350: 9.5, 400: 9.5}

DB_NAME = "pipecraft.db"

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS (RESTORED)
# -----------------------------------------------------------------------------
# Diese Funktionen fehlten in der vorherigen Version und verursachten den NameError

def get_schrauben_info(gewinde: str) -> List[Union[int, str]]:
    return SCHRAUBEN_DB.get(gewinde, ["?", "?"])

def get_ws_index(val: float) -> int:
    try: return WS_LISTE.index(val)
    except ValueError: return 6 # Default index

def get_verf_index(val: str) -> int:
    opts = ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"]
    return opts.index(val) if val in opts else 0

def get_disc_idx(val: str) -> int:
    opts = ["125 mm", "180 mm", "230 mm"]
    return opts.index(val) if val in opts else 0

def get_sys_idx(val: str) -> int:
    opts = ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"]
    return opts.index(val) if val in opts else 0

def get_cel_idx(val: str) -> int:
    opts = ["2.5 mm", "3.2 mm", "4.0 mm", "5.0 mm"]
    return opts.index(val) if val in opts else 1

# -----------------------------------------------------------------------------
# 4. BUSINESS LOGIC CLASSES
# -----------------------------------------------------------------------------

class DatabaseManager:
    @staticmethod
    def init_db() -> None:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (id INTEGER PRIMARY KEY AUTOINCREMENT, iso TEXT, naht TEXT, datum TEXT, dimension TEXT, bauteil TEXT, laenge REAL, charge TEXT, schweisser TEXT)''')
                c.execute('''CREATE TABLE IF NOT EXISTS kalkulation (id INTEGER PRIMARY KEY AUTOINCREMENT, typ TEXT, info TEXT, menge REAL, zeit_min REAL, kosten REAL, mat_text TEXT)''')
                conn.commit()
        except sqlite3.Error as e: logger.error(f"DB Init Error: {e}")

    @staticmethod
    def add_entry(table: str, data: Tuple) -> None:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                placeholders = ",".join(["?"] * len(data))
                if table == "rohrbuch": c.execute(f'INSERT INTO {table} (iso, naht, datum, dimension, bauteil, laenge, charge, schweisser) VALUES ({placeholders})', data)
                elif table == "kalkulation": c.execute(f'INSERT INTO {table} (typ, info, menge, zeit_min, kosten, mat_text) VALUES ({placeholders})', data)
                conn.commit()
        except sqlite3.Error: st.error("Fehler beim Speichern")

    @staticmethod
    def get_df(table: str) -> pd.DataFrame:
        try:
            with sqlite3.connect(DB_NAME) as conn: return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception: return pd.DataFrame()

    @staticmethod
    def delete_entry(table: str, entry_id: int) -> None:
        with sqlite3.connect(DB_NAME) as conn: conn.cursor().execute(f"DELETE FROM {table} WHERE id=?", (entry_id,))

    @staticmethod
    def clear_table(table: str) -> None:
        with sqlite3.connect(DB_NAME) as conn: conn.cursor().execute(f"DELETE FROM {table}")

class Calculator:
    @staticmethod
    def parse_math_string(text: str) -> float:
        try:
            clean = text.replace(",", ".").replace(" ", "")
            if not all(c in "0123456789.+-*/()" for c in clean): return 0.0
            return float(pd.eval(clean))
        except: return 0.0

    @staticmethod
    def calculate_weight(dn_row_idx: int, ws: float, length_mm: float, is_zme: bool = False) -> float:
        try:
            da = df_pipe.iloc[dn_row_idx]['D_Aussen']
            di = da - (2 * ws)
            vol_stahl = (math.pi * ((da/100)**2 - (di/100)**2) / 4) * (length_mm/10)
            weight = vol_stahl * 7.85
            if is_zme:
                dn_val = df_pipe.iloc[dn_row_idx]['DN']
                cem_th = 0.6 if dn_val < 300 else (0.9 if dn_val < 600 else 1.2)
                di_cem = (di/10) - (2 * cem_th)
                if di_cem > 0:
                    vol_cem = (math.pi * ((di/100)**2 - (di_cem/10)**2) / 4) * (length_mm/10)
                    weight += (vol_cem * 2.4)
            return round(weight, 1)
        except: return 0.0

    @staticmethod
    def calculate_travel(h: float, l: float, b: float) -> Tuple[float, float]:
        travel = math.sqrt(h**2 + l**2 + b**2)
        spread = math.sqrt(l**2 + b**2)
        angle = 90.0 if spread == 0 else math.degrees(math.atan(h / spread))
        return travel, angle

class Visualizer:
    @staticmethod
    def plot_stutzen_curve(r_haupt: float, r_stutzen: float) -> plt.Figure:
        angles = range(0, 361, 5)
        try: depths = [r_haupt - math.sqrt(r_haupt**2 - (r_stutzen * math.sin(math.radians(a)))**2) for a in angles]
        except ValueError: return plt.figure()
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360); ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

    @staticmethod
    def plot_etage_sketch(h: float, l: float, is_3d: bool = False, b: float = 0) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(0, 0, 'o', color='black')
        if not is_3d:
            ax.plot([0, l], [0, 0], '--', color='gray'); ax.plot([l, l], [0, h], '--', color='gray')
            ax.plot([0, l], [0, h], '-', color='#ef4444', linewidth=3)
            ax.text(l/2, -h*0.1 if h!=0 else -10, f"L={l}", ha='center'); ax.text(l, h/2, f"H={h}", va='center')
        else:
            dx, dy = b * 0.5, b * 0.3
            ax.plot([0, l], [0, 0], 'k--', alpha=0.3); ax.plot([l, l], [0, h], 'k--', alpha=0.3)
            ax.plot([0, dx], [0, dy], 'k--', alpha=0.3); ax.plot([l, l+dx], [0, dy], 'k--', alpha=0.3)
            ax.plot([dx, l+dx], [dy, dy], 'k--', alpha=0.3); ax.plot([l+dx, l+dx], [dy, h+dy], 'k--', alpha=0.3)
            ax.plot([l, l+dx], [h, h+dy], 'k--', alpha=0.3); ax.plot([0, l+dx], [0, h+dy], '-', color='#ef4444', linewidth=4, solid_capstyle='round')
            ax.text(l/2, -20, f"L={l}", ha='center', fontsize=8); ax.text(l+dx+10, h/2+dy, f"H={h}", va='center', fontsize=8); ax.text(dx/2-10, dy/2, f"B={b}", ha='right', fontsize=8)
        ax.axis('equal'); ax.axis('off')
        return fig

# -----------------------------------------------------------------------------
# 5. STATE MANAGEMENT & INIT
# -----------------------------------------------------------------------------
DEFAULT_STATE = {
    'saw_mass': 1000.0, 'saw_gap': 4.0, 'saw_deduct': "0", 'saw_zme': False,
    'kw_dn': 200, 'kw_ws': 6.3, 'kw_verf': "WIG", 'kw_pers': 1, 'kw_anz': 1, 
    'kw_split': False, 'kw_factor': 1.0,
    'cut_dn': 200, 'cut_ws': 6.3, 'cut_disc': "125 mm", 'cut_anz': 1, 
    'cut_zma': False, 'cut_iso': False, 'cut_factor': 1.0,
    'iso_sys': "Schrumpfschlauch (WKS)", 'iso_dn': 200, 'iso_anz': 1, 'iso_factor': 1.0,
    'mon_dn': 200, 'mon_type': "Schieber", 'mon_anz': 1, 'mon_factor': 1.0,
    'reg_min': 60, 'reg_pers': 2, 'bogen_winkel': 45,
    'p_lohn': 60.0, 'p_stahl': 2.5, 'p_dia': 45.0, 'p_cel': 0.40, 'p_draht': 15.0,
    'p_gas': 0.05, 'p_wks': 25.0, 'p_kebu1': 15.0, 'p_kebu2': 12.0, 'p_primer': 12.0, 'p_machine': 15.0
}

if 'store' not in st.session_state:
    st.session_state.store = DEFAULT_STATE.copy()

def save_val(key: str) -> None:
    st.session_state.store[key] = st.session_state[f"_{key}"]

def get_val(key: str) -> Any:
    return st.session_state.store.get(key, DEFAULT_STATE.get(key))

def update_kw_dn() -> None:
    st.session_state.store['kw_dn'] = st.session_state['_kw_dn']
    if st.session_state.store['kw_dn'] >= 300:
        st.session_state.store['kw_pers'] = 2
        st.session_state['_kw_pers'] = 2

DatabaseManager.init_db()

# -----------------------------------------------------------------------------
# 6. UI IMPLEMENTATION
# -----------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=50) 
st.sidebar.markdown("### Menü")
selected_dn_global = st.sidebar.selectbox("Nennweite (Global)", df_pipe['DN'], index=8, key="global_dn") 
selected_pn = st.sidebar.radio("Druckstufe", ["PN 16", "PN 10"], index=0, key="global_pn") 

row = df_pipe[df_pipe['DN'] == selected_dn_global].iloc[0]
standard_radius = float(row['Radius_BA3'])
suffix = "_16" if selected_pn == "PN 16" else "_10"

st.title("PipeCraft V24.1")
st.caption(f"🔧 Aktive Konfiguration: DN {selected_dn_global} | {selected_pn} | Radius: {standard_radius} mm")

tab_buch, tab_werk, tab_proj, tab_info = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch", "💰 Kalkulation"])

# === TAB 1: TABELLENBUCH ===
with tab_buch:
    st.subheader("Rohr & Formstücke")
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='result-card-blue'><b>Außen-Ø:</b> {row['D_Aussen']} mm</div>", unsafe_allow_html=True)
    c1.markdown(f"<div class='result-card-blue'><b>Radius (3D):</b> {standard_radius} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>T-Stück (H):</b> {row['T_Stueck_H']} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>Reduzierung (L):</b> {row['Red_Laenge_L']} mm</div>", unsafe_allow_html=True)
    st.divider(); st.subheader(f"Flansch & Montage ({selected_pn})")
    schraube = row[f'Schraube_M{suffix}']; sw, nm = Calculator.get_schrauben_details(schraube)
    mc1, mc2 = st.columns(2)
    mc1.markdown(f"<div class='result-card-blue'><b>Blattstärke:</b> {row[f'Flansch_b{suffix}']} mm</div>", unsafe_allow_html=True)
    mc2.markdown(f"<div class='result-card-blue'><b>Schraube:</b> {row[f'Lochzahl{suffix}']}x {schraube} (SW {sw})</div>", unsafe_allow_html=True)
    c_d1, c_d2, c_d3 = st.columns(3)
    c_d1.markdown(f"<div class='detail-box'>Länge (Fest-Fest)<br><span class='detail-value'>{row[f'L_Fest{suffix}']} mm</span></div>", unsafe_allow_html=True)
    c_d2.markdown(f"<div class='detail-box'>Länge (Fest-Los)<br><span class='detail-value'>{row[f'L_Los{suffix}']} mm</span></div>", unsafe_allow_html=True)
    c_d3.markdown(f"<div class='detail-box'>Drehmoment<br><span class='detail-value'>{nm} Nm</span></div>", unsafe_allow_html=True)

# === TAB 2: WERKSTATT ===
with tab_werk:
    tool_mode = st.radio("Werkzeug wählen:", ["📏 Säge (Passstück)", "🔄 Bogen (Zuschnitt)", "🔥 Stutzen (Schablone)", "📐 Etage (Versatz)"], horizontal=True, label_visibility="collapsed", key="tool_mode_nav")
    st.divider()
    
    if "Säge" in tool_mode:
        st.subheader("Passstück Berechnung")
        c_s1, c_s2 = st.columns(2)
        iso_mass = c_s1.number_input("Gesamtmaß (Iso)", value=get_val('saw_mass'), step=10.0, key="_saw_mass", on_change=save_val, args=('saw_mass',))
        spalt = c_s2.number_input("Wurzelspalt", value=get_val('saw_gap'), key="_saw_gap", on_change=save_val, args=('saw_gap',))
        abzug_input = st.text_input("Abzüge (z.B. 52+30)", value=get_val('saw_deduct'), key="_saw_deduct", on_change=save_val, args=('saw_deduct',))
        
        abzuege = Calculator.parse_math_string(abzug_input)
        saege_erg = iso_mass - spalt - abzuege
        st.markdown(f"<div class='result-card-green'>Sägelänge: {round(saege_erg, 1)} mm</div>", unsafe_allow_html=True)
        
        dn_idx = df_pipe[df_pipe['DN'] == selected_dn_global].index[0]
        std_ws = WS_STD.get(selected_dn_global, 4.0) 
        c_zme = st.checkbox("ZME (Beton innen)?", value=get_val('saw_zme'), key="_saw_zme", on_change=save_val, args=('saw_zme',))
        kg = Calculator.calculate_weight(dn_idx, std_ws, saege_erg, c_zme)
        st.markdown(f"<div class='weight-box'>⚖️ Gewicht: ca. {kg} kg</div>", unsafe_allow_html=True)
        
        bogen_winkel = st.session_state.get('bogen_winkel', 45)
        vorbau_custom = int(round(standard_radius * math.tan(math.radians(bogen_winkel/2)), 0))
        with st.expander(f"ℹ️ Abzugsmaße (DN {selected_dn_global})", expanded=True):
            st.markdown(f"""
            * **Flansch:** {row[f'Flansch_b{suffix}']} mm
            * **Bogen 90°:** {standard_radius} mm
            * **Bogen {bogen_winkel}° (Zuschnitt):** {vorbau_custom} mm
            * **T-Stück:** {row['T_Stueck_H']} mm
            * **Reduzierung:** {row['Red_Laenge_L']} mm
            """)

    elif "Bogen" in tool_mode:
        st.subheader("Bogen Zuschnitt")
        angle = st.slider("Winkel (°)", 0, 90, 45, key="bogen_winkel")
        vorbau = round(standard_radius * math.tan(math.radians(angle/2)), 1)
        aussen = round((standard_radius + (row['D_Aussen']/2)) * angle * (math.pi/180), 1)
        innen = round((standard_radius - (row['D_Aussen']/2)) * angle * (math.pi/180), 1)
        st.markdown(f"<div class='result-card-green'>Vorbau: {vorbau} mm</div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2); b1.metric("Rücken", f"{aussen} mm"); b2.metric("Bauch", f"{innen} mm")

    elif "Stutzen" in tool_mode:
        st.subheader("Stutzen Schablone")
        c_st1, c_st2 = st.columns(2)
        dn_stutzen = c_st1.selectbox("DN Stutzen", df_pipe['DN'], index=6, key="stutz_dn1")
        dn_haupt = c_st2.selectbox("DN Hauptrohr", df_pipe['DN'], index=9, key="stutz_dn2")
        
        if dn_stutzen > dn_haupt: st.error("Fehler: Stutzen > Hauptrohr")
        else:
            r_k = df_pipe[df_pipe['DN'] == dn_stutzen].iloc[0]['D_Aussen'] / 2; r_g = df_pipe[df_pipe['DN'] == dn_haupt].iloc[0]['D_Aussen'] / 2
            c_tab, c_plot = st.columns([1, 2])
            table_data = []
            for a in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
                t = int(round(r_g - math.sqrt(r_g**2 - (r_k * math.sin(math.radians(a)))**2), 0))
                umfang_pos = int(round((r_k * 2 * math.pi) * (a/360), 0))
                table_data.append([f"{a}°", t, umfang_pos])
            with c_tab: st.dataframe(pd.DataFrame(table_data, columns=["Winkel", "Tiefe", "Umfang"]), hide_index=True)
            with c_plot: st.pyplot(Visualizer.plot_stutzen_curve(r_g, r_k))

    elif "Etage" in tool_mode:
        st.subheader("Etagen Berechnung")
        et_type = st.radio("Typ", ["2D (Einfach)", "3D (Kastenmaß)", "3D (Fix-Winkel)"], horizontal=True, key="et_type")
        spalt_et = st.number_input("Spalt", 4, key="et_gap")
        c_calc, c_vis = st.columns([1, 1])
        weight_l = 0.0
        
        if "2D" in et_type:
            with c_calc:
                h = st.number_input("Höhe H", 300, key="et2d_h"); l = st.number_input("Länge L", 400, key="et2d_l")
                travel, angle = Calculator.calculate_travel(h, l, 0)
                abzug = 2 * (standard_radius * math.tan(math.radians(angle/2)))
                erg = travel - abzug - spalt_et
                st.markdown(f"<div class='result-card-green'>Säge: {round(erg, 1)} mm</div>", unsafe_allow_html=True)
                weight_l = erg
            with c_vis: st.pyplot(Visualizer.plot_etage_sketch(h, l))
        
        elif "Kastenmaß" in et_type:
            with c_calc:
                b = st.number_input("Breite", 200, key="et3d_b"); h = st.number_input("Höhe", 300, key="et3d_h"); l = st.number_input("Länge", 400, key="et3d_l")
                travel, angle = Calculator.calculate_travel(h, l, b)
                abzug = 2 * (standard_radius * math.tan(math.radians(angle/2)))
                erg = travel - abzug - spalt_et
                st.markdown(f"<div class='result-card-green'>Säge: {round(erg, 1)} mm</div>", unsafe_allow_html=True)
                weight_l = erg
            with c_vis: st.pyplot(Visualizer.plot_etage_sketch(h, l, True, b))
                
        elif "Fix-Winkel" in et_type:
            with c_calc:
                b = st.number_input("Breite", 200, key="etfix_b"); h = st.number_input("Höhe", 300, key="etfix_h")
                fix_w = st.selectbox("Winkel", [15, 30, 45, 60, 90], index=2, key="etfix_w")
                spread = math.sqrt(b**2 + h**2); l_req = spread / math.tan(math.radians(fix_w))
                diag = math.sqrt(b**2 + h**2 + l_req**2); abzug = 2 * (standard_radius * math.tan(math.radians(fix_w/2)))
                erg = diag - abzug - spalt_et
                st.info(f"Benötigte Länge L: {round(l_req, 1)} mm")
                st.markdown(f"<div class='result-card-green'>Säge: {round(erg, 1)} mm</div>", unsafe_allow_html=True)
                weight_l = erg
            with c_vis: st.pyplot(Visualizer.plot_etage_sketch(h, l_req, True, b))
        
        if weight_l > 0:
            dn_idx = df_pipe[df_pipe['DN'] == selected_dn_global].index[0]
            std_ws = WS_STD.get(selected_dn_global, 4.0)
            c_zme_et = st.checkbox("ZME?", key="et_zme")
            kg = Calculator.calculate_weight(dn_idx, std_ws, weight_l, c_zme_et)
            st.markdown(f"<div class='weight-box'>⚖️ Gewicht: ca. {kg} kg</div>", unsafe_allow_html=True)

# === TAB 3: ROHRBUCH ===
with tab_proj:
    st.subheader("Digitales Rohrbuch")
    with st.form("rb_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3); iso = c1.text_input("ISO"); naht = c2.text_input("Naht"); datum = c3.date_input("Datum")
        c4, c5, c6 = st.columns(3); dn_sel = c4.selectbox("Dimension", df_pipe['DN'], index=8, key="rb_dn_sel"); bauteil = c5.selectbox("Bauteil", ["📏 Rohr", "⤵️ Bogen", "⭕ Flansch", "🔗 Muffe", "🔩 Nippel"])
        laenge = c6.number_input("Länge", value=0); c7, c8 = st.columns(2); charge = c7.text_input("Charge"); schweisser = c8.text_input("Schweißer")
        if st.form_submit_button("Speichern"):
            DatabaseManager.add_entry("rohrbuch", (iso, naht, datum.strftime("%d.%m.%Y"), f"DN {dn_sel}", bauteil, laenge, charge, schweisser))
            st.success("Gespeichert!")
    df_rb = DatabaseManager.get_df("rohrbuch")
    st.dataframe(df_rb, use_container_width=True)
    with st.expander("Zeile löschen"):
        opts = {f"ID {r['id']}: {r['iso']} {r['naht']}": r['id'] for i, r in df_rb.iterrows()}
        if opts:
            sel = st.selectbox("Wähle:", list(opts.keys()), key="rb_del")
            if st.button("Löschen", key="rb_del_btn"): DatabaseManager.delete_entry("rohrbuch", opts[sel]); st.rerun()

# === TAB 4: KALKULATION ===
with tab_info:
    with st.expander("💶 Preis-Datenbank (Einstellungen)"):
        c_io1, c_io2 = st.columns(2)
        try: json_data = json.dumps(st.session_state.store); c_io1.download_button("💾 Einstellungen speichern", data=json_data, file_name="pipecraft_config.json", mime="application/json")
        except: c_io1.error("Export Fehler")
        uploaded_file = c_io2.file_uploader("📂 Einstellungen laden", type=["json"])
        if uploaded_file is not None:
            try: data = json.load(uploaded_file); st.session_state.store.update(data); st.success("Geladen!"); st.rerun()
            except: st.error("Fehler beim Laden")
        st.divider()
        c_p1, c_p2, c_p3 = st.columns(3); st.session_state.store['p_lohn'] = c_p1.number_input("Lohn (€/h)", value=get_val('p_lohn'), key="_p_lohn", on_change=save_val, args=('p_lohn',))
        st.session_state.store['p_machine'] = c_p2.number_input("Maschine (€/h)", value=get_val('p_machine'), key="_p_machine", on_change=save_val, args=('p_machine',))

    kalk_sub_mode = st.radio("Ansicht:", ["Eingabe & Rechner", "📊 Projekt Status / Export"], horizontal=True, label_visibility="collapsed")
    st.divider()

    if kalk_sub_mode == "Eingabe & Rechner":
        calc_task = st.radio("Tätigkeit", ["🔥 Fügen", "✂️ Trennen", "🔧 Montage", "🛡️ Isolierung", "🚗 Regie"], horizontal=True, key="calc_mode")
        st.markdown("---")
        p_lohn = get_val('p_lohn'); p_machine = get_val('p_machine'); p_cel = get_val('p_cel'); p_draht = get_val('p_draht'); p_gas = get_val('p_gas'); p_stahl_disc = get_val('p_stahl'); p_dia_disc = get_val('p_dia'); p_wks = get_val('p_wks')

        if "Fügen" in calc_task:
            c1, c2, c3 = st.columns(3)
            k_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('kw_dn')), key="_kw_dn", on_change=update_kw_dn)
            k_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('kw_ws')), key="_kw_ws", on_change=save_val, args=('kw_ws',))
            verf_opts = ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"]
            k_verf = c3.selectbox("Verfahren", verf_opts, index=get_verf_index(get_val('kw_verf')), key="_kw_verf", on_change=save_val, args=('kw_verf',))
            c4, c5 = st.columns(2)
            if get_val('kw_dn') >= 300: st.info("ℹ️ Großrohr (≥ DN 300): Team-Größe automatisch auf 2 gesetzt.")
            pers_count = c4.number_input("Anzahl Mitarbeiter", value=get_val('kw_pers'), min_value=1, key="_kw_pers", on_change=save_val, args=('kw_pers',))
            anz = c5.number_input("Anzahl Nähte", value=get_val('kw_anz'), min_value=1, key="_kw_anz", on_change=save_val, args=('kw_anz',))
            factor = st.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('kw_factor'), 0.1, key="_kw_factor", on_change=save_val, args=('kw_factor',))
            split_entry = st.checkbox("2 Positionen (Vorb. + Fügen)?", value=get_val('kw_split'), key="_kw_split", on_change=save_val, args=('kw_split',))
            
            zoll = k_dn / 25.0; min_per_inch = 10.0 if "WIG" == k_verf else (3.5 if "CEL" in k_verf else 5.0)
            duration_per_seam = ((zoll * min_per_inch + zoll * 2.5) / pers_count) * factor
            crew_hourly = (pers_count * p_lohn) + (pers_count * p_machine)
            total_labor_cost = (duration_per_seam / 60 * crew_hourly) * anz
            
            da = df_pipe[df_pipe['DN'] == k_dn].iloc[0]['D_Aussen']
            kg = (da * math.pi * k_ws**2 * 0.7 / 1000 * 7.85 / 1000) * 1.5
            mat_cost = ((5.0 * kg) * 0.40 * anz) if "CEL" in k_verf else ((kg * p_draht + (duration_per_seam/60 * 15 * p_gas)) * anz)
            
            m1, m2 = st.columns(2); m1.metric("Zeit", f"{int(duration_per_seam*anz)} min"); m2.metric("Kosten", f"{round(total_labor_cost + mat_cost, 2)} €")
            if st.button("Hinzufügen", key="add_weld"):
                if split_entry:
                    DatabaseManager.add_entry("kalkulation", ("Vorbereitung", f"DN {k_dn}", anz, (duration_per_seam*anz)/2, total_labor_cost/2, "-"))
                    DatabaseManager.add_entry("kalkulation", ("Fügen", f"DN {k_dn}", anz, (duration_per_seam*anz)/2, (total_labor_cost/2)+mat_cost, "Material inkl."))
                else:
                    DatabaseManager.add_entry("kalkulation", ("Fügen", f"DN {k_dn} {k_verf}", anz, duration_per_seam*anz, total_labor_cost + mat_cost, "Material inkl."))
                st.rerun()

        elif "Trennen" in calc_task:
            c1, c2, c3, c4 = st.columns(4)
            c_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('cut_dn')), key="_cut_dn", on_change=save_val, args=('cut_dn',))
            c_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('cut_ws')), key="_cut_ws", on_change=save_val, args=('cut_ws',))
            disc = c3.selectbox("Scheibe", ["125 mm", "180 mm", "230 mm"], index=get_disc_idx(get_val('cut_disc')), key="_cut_disc", on_change=save_val, args=('cut_disc',))
            anz = c4.number_input("Anzahl", value=get_val('cut_anz'), min_value=1, key="_cut_anz", on_change=save_val, args=('cut_anz',))
            c5, c6 = st.columns(2); zma = c5.checkbox("Beton (ZMA)?", value=get_val('cut_zma'), key="_cut_zma", on_change=save_val, args=('cut_zma',)); iso = c6.checkbox("Mantel entfernen?", value=get_val('cut_iso'), key="_cut_iso", on_change=save_val, args=('cut_iso',))
            factor = st.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('cut_factor'), 0.1, key="_cut_factor", on_change=save_val, args=('cut_factor',))
            
            zoll = c_dn / 25.0; cap = 14000 if "230" in disc else (7000 if "180" in disc else 3500)
            t_total = zoll * 0.5 * (3.0 if zma else 1.0) * (1.3 if iso else 1.0) * factor * anz
            area = (math.pi * df_pipe[df_pipe['DN']==c_dn].iloc[0]['D_Aussen']) * c_ws
            n_disc = math.ceil((area * (2.0 if zma else 1.0) * anz) / cap)
            cost = ((t_total/60 * p_lohn) + (n_disc * (p_dia_disc if zma else p_stahl_disc)))
            
            m1, m2 = st.columns(2); m1.metric("Zeit", f"{int(t_total)} min"); m2.metric("Kosten", f"{round(cost, 2)} €")
            if st.button("Hinzufügen", key="cut_add"):
                DatabaseManager.add_entry("kalkulation", ("Vorbereitung", f"DN {c_dn} ({disc})", anz, t_total, cost, f"{n_disc}x Scheiben"))
                st.rerun()

        elif "Montage" in calc_task:
            c1, c2, c3 = st.columns(3)
            m_type = c1.selectbox("Bauteil", ["Schieber", "Klappe", "Hydrant", "Formstück"], key="mon_type")
            m_dn = c2.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('mon_dn')), key="_mon_dn", on_change=save_val, args=('mon_dn',))
            m_anz = c3.number_input("Anzahl", value=get_val('mon_anz'), min_value=1, key="_mon_anz", on_change=save_val, args=('mon_anz',))
            factor = st.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('mon_factor'), 0.1, key="_mon_factor", on_change=save_val, args=('mon_factor',))
            
            bolts = df_pipe[df_pipe['DN'] == m_dn].iloc[0][f'Lochzahl{suffix}']
            total_time = ((bolts * 2.5) + 20) * m_anz * factor
            total_cost = (total_time / 60) * (p_lohn + p_machine)
            m1, m2 = st.columns(2); m1.metric("Zeit", f"{int(total_time)} min"); m2.metric("Kosten", f"{round(total_cost, 2)} €")
            if st.button("Hinzufügen", key="mon_add"):
                DatabaseManager.add_entry("kalkulation", ("Montage", f"DN {m_dn} {m_type}", m_anz, total_time, total_cost, f"{bolts*2}x Schrauben"))
                st.rerun()

        elif "Isolierung" in calc_task:
            sys = st.radio("System", ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"], index=get_sys_idx(get_val('iso_sys')), key="_iso_sys", on_change=save_val, args=('iso_sys',))
            c1, c2 = st.columns(2)
            i_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('iso_dn')), key="_iso_dn", on_change=save_val, args=('iso_dn',))
            i_anz = c2.number_input("Anzahl", value=get_val('iso_anz'), min_value=1, key="_iso_anz", on_change=save_val, args=('iso_anz',))
            factor = st.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('iso_factor'), 0.1, key="_iso_factor", on_change=save_val, args=('iso_factor',))
            time = (20 + (i_dn * 0.07)) * factor * i_anz
            cost = ((time/60 * p_lohn) + (p_wks * i_anz))
            m1, m2 = st.columns(2); m1.metric("Zeit", f"{int(time)} min"); m2.metric("Kosten", f"{round(cost, 2)} €")
            if st.button("Hinzufügen", key="iso_add"): DatabaseManager.add_entry("kalkulation", ("Iso", f"DN {i_dn} {sys}", i_anz, time, cost, "-")); st.rerun()

        elif "Regie" in calc_task:
            c1, c2 = st.columns(2); t = c1.number_input("Minuten", value=get_val('reg_min'), step=15, key="_reg_min", on_change=save_val, args=('reg_min',)); p = c2.number_input("Personen", value=get_val('reg_pers'), min_value=1, key="_reg_pers", on_change=save_val, args=('reg_pers',))
            cost = (t/60 * p_lohn) * p; st.metric("Kosten", f"{round(cost, 2)} €")
            if st.button("Hinzufügen", key="reg_add"): DatabaseManager.add_entry("kalkulation", ("Regie", f"{p} Pers.", 1, t, cost, "-")); st.rerun()

        st.markdown("### 📊 Projekt Status")
        df_k = DatabaseManager.get_df("kalkulation")
        if not df_k.empty:
            sc1, sc2 = st.columns(2); sc1.metric("Gesamt", f"{round(df_k['kosten'].sum(), 2)} €"); sc2.metric("Stunden", f"{round(df_k['zeit_min'].sum()/60, 1)} h")
            st.dataframe(df_k, use_container_width=True)
            c_del, c_rst = st.columns(2)
            with c_del.expander("Zeile löschen"):
                opts = {f"ID {r['id']}: {r['typ']}": r['id'] for i, r in df_k.iterrows()}
                if opts:
                    sel = st.selectbox("Wähle:", list(opts.keys()), key="kalk_del")
                    if st.button("Löschen"): DatabaseManager.delete_entry("kalkulation", opts[sel]); st.rerun()
            if c_rst.button("Alles Löschen", type="primary"): DatabaseManager.clear_table("kalkulation"); st.rerun()
            
            c_xls, c_pdf = st.columns(2)
            c_xls.download_button("Excel Export", convert_df_to_excel(df_k), "kalkulation.xlsx")
            if PDF_AVAILABLE: c_pdf.download_button("PDF Export", create_pdf(df_k), "bericht.pdf")
        else: st.info("Projekt ist leer.")
