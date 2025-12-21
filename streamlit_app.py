"""
PipeCraft V44.0 (Pure Engineering & Documentation)
--------------------------------------------------
Fokus: Technische Berechnung und Schweißnaht-Dokumentation.
Entfernt: Kommerzielle Kalkulation & Lagerbestände.

Features:
1.  Smart Cut: Sägelisten-Erstellung (inkl. Reduzier-Logik nach DN Groß).
2.  Engineering: 3D-Etagen, Stutzen-Abwicklung (Tabelle + Plot), Gewichte.
3.  Rohrbuch: Detaillierte Erfassung mit Export (Excel/PDF).

Author: Senior Lead Software Engineer
"""

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D 
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import List, Tuple, Any, Optional, Union

# -----------------------------------------------------------------------------
# 0. SYSTEM CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipeCraft")

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(
    page_title="PipeCraft V44.0",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #0f172a; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #1e293b !important; font-weight: 700; }
    
    .result-card-blue { 
        background-color: #eff6ff; padding: 20px; border-radius: 10px; 
        border-left: 6px solid #3b82f6; margin-bottom: 15px; color: #1e3a8a; 
    }
    
    .result-card-green { 
        background: linear-gradient(to right, #f0fdf4, #ffffff); padding: 25px; 
        border-radius: 12px; border-left: 8px solid #22c55e; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-bottom: 15px; 
        text-align: center; font-size: 1.6rem; font-weight: 800; color: #14532d; 
    }
    
    .detail-box { 
        background-color: #f1f5f9; border: 1px solid #cbd5e1; padding: 15px; 
        border-radius: 8px; text-align: center; font-size: 0.9rem; color: #334155; 
    }
    
    .weight-box { 
        background-color: #fff1f2; border: 1px solid #fecdd3; color: #be123c; 
        padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; 
        margin-top: 15px; 
    }
    
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] { 
        border-radius: 4px; border: 1px solid #cbd5e1; 
    }
    
    div.stButton > button { 
        width: 100%; border-radius: 4px; font-weight: 600; 
        border: 1px solid #cbd5e1; transition: all 0.2s; 
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
    st.error(f"Datenbankfehler: {e}")
    st.stop()

SCHRAUBEN_DB = { "M12": [18, 60], "M16": [24, 130], "M20": [30, 250], "M24": [36, 420], "M27": [41, 600], "M30": [46, 830], "M33": [50, 1100], "M36": [55, 1400], "M39": [60, 1800], "M45": [70, 2700], "M52": [80, 4200] }
WS_STD_MAP = {25: 3.2, 32: 3.6, 40: 3.6, 50: 3.9, 65: 5.2, 80: 5.5, 100: 6.0, 125: 6.6, 150: 7.1, 200: 8.2, 250: 9.3, 300: 9.5, 350: 9.5, 400: 9.5, 450: 9.5, 500: 9.5}
DB_NAME = "pipecraft_v44.db"

# -----------------------------------------------------------------------------
# 2. LOGIC LAYER
# -----------------------------------------------------------------------------

def get_row_by_dn(dn: int) -> pd.Series:
    try: return df_pipe[df_pipe['DN'] == dn].iloc[0]
    except: return df_pipe.iloc[0]

def get_schrauben_info(gewinde): return SCHRAUBEN_DB.get(gewinde, ["?", "?"])

# --- SMART CUT MANAGER ---
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
        if type_name == "Bogen 90° (BA3)": return float(row_data['Radius_BA3'])
        elif type_name == "Bogen (Zuschnitt)": return float(row_data['Radius_BA3']) * math.tan(math.radians(custom_angle / 2))
        elif type_name == "Flansch (Vorschweiß)": return float(row_data[f'Flansch_b{pn_suffix}'])
        elif type_name == "T-Stück": return float(row_data['T_Stueck_H'])
        elif "Reduzierung" in type_name: return float(row_data['Red_Laenge_L'])
        return 0.0

# --- PHYSICS ---
class PhysicsEngine:
    DENSITY_STEEL = 7.85
    DENSITY_CEMENT = 2.40
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

# --- VISUALIZATION ---
class GeometryEngine:
    @staticmethod
    def solve_offset_3d(h, l, b):
        travel = math.sqrt(h**2 + l**2 + b**2); spread = math.sqrt(l**2 + b**2)
        angle = 90.0 if spread == 0 else math.degrees(math.atan(h / spread))
        return travel, angle

class Visualizer:
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

# --- DATABASE REPOSITORY ---
class DatabaseRepository:
    @staticmethod
    def init_tables():
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            # NUR ROHRBUCH, KEINE KALKULATION
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (id INTEGER PRIMARY KEY AUTOINCREMENT, iso TEXT, naht TEXT, datum TEXT, dimension TEXT, bauteil TEXT, laenge REAL, charge TEXT, schweisser TEXT)''')
            conn.commit()

    @staticmethod
    def add_rohrbuch(data):
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute('INSERT INTO rohrbuch (iso, naht, datum, dimension, bauteil, laenge, charge, schweisser) VALUES (?,?,?,?,?,?,?,?)', data)
            conn.commit()

    @staticmethod
    def get_all(table):
        with sqlite3.connect(DB_NAME) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    
    @staticmethod
    def delete(table, id):
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(f"DELETE FROM {table} WHERE id=?", (id,))
            conn.commit()

# --- EXPORT ---
def export_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer: df.to_excel(writer, index=False)
    return out.getvalue()

def export_pdf(df):
    if not PDF_AVAILABLE: return b""
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Rohrbuch Report", 0, 1, 'C'); pdf.ln(5)
    for _, r in df.iterrows():
        try:
            txt = f"{r.get('iso','')} | {r.get('naht','')} | {r.get('dimension','')}"
            pdf.cell(0, 8, txt.encode('latin-1','replace').decode('latin-1'), 1, 1)
        except: pass
    return pdf.output(dest='S').encode('latin-1')

# --- INITIALIZATION ---
DatabaseRepository.init_tables()

if 'fitting_list' not in st.session_state:
    st.session_state.fitting_list = []

if 'store' not in st.session_state:
    st.session_state.store = {
        'saw_mass': 1000.0, 'saw_gap': 4.0, 'saw_zme': False,
        'view_azim': 45, 'view_elev': 30
    }

def save_val(key): st.session_state.store[key] = st.session_state[f"_{key}"]
def get_val(key): return st.session_state.store.get(key)

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

st.title("PipeCraft V44.0")
st.caption(f"🔧 Engineering Suite: DN {selected_dn_global} | {selected_pn} | Radius: {standard_radius} mm")

# Nur 3 Tabs: Buch, Werkstatt, Rohrbuch
tab_buch, tab_werk, tab_proj = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch"])

# TAB 1: BUCH
with tab_buch:
    st.subheader("Technische Daten")
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='result-card-blue'><b>Außen-Ø:</b> {row['D_Aussen']} mm</div>", unsafe_allow_html=True)
    c1.markdown(f"<div class='result-card-blue'><b>Radius (3D):</b> {standard_radius} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>T-Stück (H):</b> {row['T_Stueck_H']} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='result-card-blue'><b>Reduzierung (L):</b> {row['Red_Laenge_L']} mm</div>", unsafe_allow_html=True)
    st.divider()
    schraube = row[f'Schraube_M{suffix}']; sw, nm = get_schrauben_info(schraube)
    mc1, mc2 = st.columns(2)
    mc1.markdown(f"<div class='result-card-blue'><b>Blattstärke:</b> {row[f'Flansch_b{suffix}']} mm</div>", unsafe_allow_html=True)
    mc2.markdown(f"<div class='result-card-blue'><b>Schraube:</b> {row[f'Lochzahl{suffix}']}x {schraube} (SW {sw})</div>", unsafe_allow_html=True)
    c_d1, c_d2 = st.columns(2)
    c_d1.markdown(f"<div class='detail-box'>Länge (Fest-Fest)<br><b>{row[f'L_Fest{suffix}']} mm</b></div>", unsafe_allow_html=True)
    c_d2.markdown(f"<div class='detail-box'>Länge (Fest-Los)<br><b>{row[f'L_Los{suffix}']} mm</b></div>", unsafe_allow_html=True)

# TAB 2: WERKSTATT
with tab_werk:
    tool_mode = st.radio("Werkzeug:", ["📏 Säge (Smart Cut)", "🔄 Bogen", "🔥 Stutzen", "📐 Etage"], horizontal=True, label_visibility="collapsed")
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
        else:
            f_dn = col_f2.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(selected_dn_global))

        f_angle = 45.0
        if "Zuschnitt" in f_type: f_angle = col_f3.number_input("Winkel", value=45.0, step=1.0)
        else: col_f3.write("-")
        f_count = col_f4.number_input("Anzahl", value=1, min_value=1)
        
        if st.button("Hinzufügen (+)", type="secondary"):
            deduct = FittingManager.get_deduction(f_type, f_dn, suffix, f_angle)
            name = f"{f_type} (DN {f_dn})"
            if "Zuschnitt" in f_type: name += f" [{f_angle}°]"
            if "Reduzierung" in f_type: name += f" (Länge via DN {f_dn})"
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
        if final < 0: st.error("Bauteile zu lang!")
        else:
            c_res1, c_res2 = st.columns(2)
            c_res1.markdown(f"<div class='result-card-green'>Sägelänge: {round(final, 1)} mm</div>", unsafe_allow_html=True)
            c_res2.info(f"Abzüge: {round(total_deduct, 1)} mm (Teile) + {round(total_gaps*spalt, 1)} mm (Spalte)")
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
        if dn1 > dn2: st.error("Stutzen zu groß!")
        else:
            rk = df_pipe[df_pipe['DN']==dn1].iloc[0]['D_Aussen']/2
            rg = df_pipe[df_pipe['DN']==dn2].iloc[0]['D_Aussen']/2
            c_tab, c_plot = st.columns([1, 2])
            table_data = []
            for a in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
                t = int(round(rg - math.sqrt(rg**2 - (rk * math.sin(math.radians(a)))**2), 0))
                u = int(round((rk * 2 * math.pi) * (a/360), 0))
                table_data.append({"Winkel": f"{a}°", "Tiefe": t, "Umfang": u})
            with c_tab: st.table(pd.DataFrame(table_data))
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

# TAB 3: ROHRBUCH
with tab_proj:
    st.subheader("Digitales Rohrbuch")
    with st.form("rb"):
        c1, c2, c3 = st.columns(3); iso = c1.text_input("ISO"); naht = c2.text_input("Naht"); datum = c3.date_input("Datum")
        c4, c5 = st.columns(2); dn_s = c4.selectbox("DN", df_pipe['DN']); len_s = c5.number_input("Länge", value=0)
        
        # UPGRADED BAUTEIL SELECTOR (DETAILED)
        c_bt = st.columns(1)[0]
        fitting_types_detailed = ["Rohr", "Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung (konz.)", "Muffe", "Nippel"]
        bauteil = c_bt.selectbox("Bauteil / Formteil", fitting_types_detailed)
        
        c6, c7 = st.columns(2); charge = c6.text_input("Charge"); schweisser = c7.text_input("Schweißer")
        if st.form_submit_button("Speichern"):
            DatabaseRepository.add_rohrbuch((iso, naht, datum.strftime("%d.%m.%Y"), f"DN {dn_s}", bauteil, len_s, charge, schweisser))
            st.success("Gespeichert!")
    
    df_rb = DatabaseRepository.get_all("rohrbuch")
    st.dataframe(df_rb, use_container_width=True)
    if not df_rb.empty:
        c_ex1, c_ex2 = st.columns(2)
        c_ex1.download_button("📥 Excel Download", export_excel(df_rb), "rohrbuch.xlsx")
        if PDF_AVAILABLE: c_ex2.download_button("📄 PDF Download", export_pdf(df_rb), "rohrbuch.pdf")
    with st.expander("Zeile löschen"):
        if not df_rb.empty:
            opts = {f"ID {r['id']}: {r['iso']} {r['naht']}": r['id'] for i, r in df_rb.iterrows()}
            if opts:
                sel = st.selectbox("Wähle Eintrag:", list(opts.keys()), key="rb_del_sel")
                if st.button("Löschen", key="rb_del_btn"): DatabaseRepository.delete("rohrbuch", opts[sel]); st.rerun()
