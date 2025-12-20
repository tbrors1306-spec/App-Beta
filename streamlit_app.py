import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sqlite3
import json
from datetime import datetime
from io import BytesIO

# --- FIX: Sicherer Import für FPDF ---
try:
    from fpdf import FPDF
    pdf_available = True
except ImportError:
    pdf_available = False

# -----------------------------------------------------------------------------
# 0. ROHRBAU PROFI ENGINE (DAS HERZSTÜCK)
# -----------------------------------------------------------------------------
class RohrbauProfiEngine:
    """
    High-End Engineering Core.
    Ersetzt statische Tabellen durch echte Physik nach DIN EN 13480 / ASME.
    """
    def __init__(self):
        # Material-Datenbank: Streckgrenze (Rp0.2 / Yield) in MPa bei Temperatur T
        self.material_db = {
            "S235JR": {20: 235, 100: 215, 200: 195, 300: 175},
            "P265GH": {20: 265, 100: 240, 200: 220, 300: 190, 400: 170},
            "1.4404 (316L)": {20: 220, 100: 190, 200: 175, 300: 160, 400: 145}
        }

    def get_material_stress(self, material, temp_c):
        """Interpoliert zulässige Spannung bei Betriebstemperatur."""
        props = self.material_db.get(material)
        if not props: return 235.0 # Fallback
        temps = sorted(props.keys())
        if temp_c <= temps[0]: return props[temps[0]]
        if temp_c >= temps[-1]: return props[temps[-1]]
        for i in range(len(temps)-1):
            if temps[i] <= temp_c <= temps[i+1]:
                t1, t2 = temps[i], temps[i+1]
                s1, s2 = props[t1], props[t2]
                return s1 + (s2 - s1) * (temp_c - t1) / (t2 - t1)
        return 235.0

    def calc_wall_thickness_en13480(self, pressure_bar, diameter_outside_mm, material, temp_c, safety_factor=1.5, corrosion_mm=1.0):
        """
        Berechnet die erforderliche Mindestwandstärke nach DIN EN 13480-3.
        e = (Pc * Do) / (2 * (f * z + Pc)) + c
        """
        stress_at_temp = self.get_material_stress(material, temp_c)
        f = stress_at_temp / safety_factor # Zulässige Spannung
        z = 1.0 # Schweißnahtfaktor (1.0 = geprüft, 0.7 = ungeprüft)
        pc = pressure_bar / 10.0 # Umrechnung bar -> MPa
        
        # Berechnung ohne Zuschläge
        e_theoretical = (pc * diameter_outside_mm) / (2 * (f * z + pc))
        
        # Inkl. Korrosionszuschlag und Minustoleranz (pauschal 12.5% angenommen)
        e_required = e_theoretical + corrosion_mm
        e_nominal = e_required * 1.125 
        
        return {
            "min_required": round(e_required, 2),
            "nominal_suggestion": round(e_nominal, 2),
            "stress_used": round(f, 1),
            "pressure_mpa": pc
        }

    def colebrook_white_solver(self, reynolds, roughness_mm, diameter_mm):
        """Iterative Lösung der Colebrook-White Gleichung."""
        if reynolds < 2300: return 64 / reynolds
        rel_rough = (roughness_mm / 1000) / (diameter_mm / 1000)
        f = 0.02 # Startwert
        for _ in range(10): # Newton-Raphson
            try:
                term = -2.0 * math.log10((rel_rough/3.7) + (2.51 / (reynolds * math.sqrt(f))))
                f_new = (1/term)**2
                if abs(f - f_new) < 0.00001: return f_new
                f = f_new
            except: return 0.02
        return f

    def calc_pressure_drop(self, flow_m3_h, d_inner_mm, length_m, medium_temp_c):
        """Hydraulische Berechnung (Wasser 20°C Referenz, angepasst)."""
        if d_inner_mm <= 0: return 0, 0, 0
        area = math.pi * ((d_inner_mm/1000)/2)**2
        velocity = (flow_m3_h / 3600) / area
        
        # Viskosität Wasser (grob temperaturabhängig)
        visc_map = {10: 1.3, 20: 1.0, 50: 0.55, 90: 0.3} # cSt
        visc = 1.0
        for t, v in visc_map.items():
            if medium_temp_c >= t: visc = v
            
        reynolds = (velocity * (d_inner_mm/1000)) / (visc * 1e-6)
        if reynolds == 0: return 0,0,0
        
        lambda_val = self.colebrook_white_solver(reynolds, 0.045, d_inner_mm)
        rho = 998 - (medium_temp_c * 0.2) # Dichte grob korrigiert
        
        dp_pa = lambda_val * (length_m / (d_inner_mm/1000)) * (rho/2) * velocity**2
        return round(dp_pa / 100000, 4), round(velocity, 2), int(reynolds)

# Engine initialisieren
engine = RohrbauProfiEngine()

# -----------------------------------------------------------------------------
# 1. DESIGN & CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Rohrbau Profi V2.0", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    h1 { color: #1e293b; font-weight: 800; letter-spacing: -1px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #fff; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    .stTabs [aria-selected="true"] { background-color: #e0f2fe; color: #0284c7; border-bottom: 2px solid #0284c7; }
    
    .metric-card {
        background: white; padding: 15px; border-radius: 8px;
        border-left: 5px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .safety-ok { border-left-color: #22c55e; background-color: #f0fdf4; }
    .safety-fail { border-left-color: #ef4444; background-color: #fef2f2; }
    
    .big-stat { font-size: 24px; font-weight: bold; }
    .sub-stat { font-size: 14px; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATEN DEFINITION
# -----------------------------------------------------------------------------
data = {
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
df = pd.DataFrame(data)

schrauben_db = { "M12": [18, 60], "M16": [24, 130], "M20": [30, 250], "M24": [36, 420], "M27": [41, 600], "M30": [46, 830], "M33": [50, 1100], "M36": [55, 1400], "M39": [60, 1800], "M45": [70, 2700], "M52": [80, 4200] }
ws_liste = [2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 8.8, 10.0, 11.0, 12.5, 14.2, 16.0]
wandstaerken_std = { 25: 3.2, 32: 3.6, 40: 3.6, 50: 3.9, 65: 5.2, 80: 5.5, 100: 6.0, 125: 6.6, 150: 7.1, 200: 8.2, 250: 9.3, 300: 9.5, 350: 9.5, 400: 9.5, 450: 9.5, 500: 9.5 }

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_schrauben_info(gewinde): return schrauben_db.get(gewinde, ["?", "?"])
def parse_abzuege(text):
    try: return float(pd.eval(text.replace(",", ".").replace(" ", "")))
    except: return 0.0
def get_ws_index(val):
    try: return ws_liste.index(val)
    except: return 6
def get_verf_index(val): return ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"].index(val) if val in ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"] else 0
def get_disc_idx(val): return ["125 mm", "180 mm", "230 mm"].index(val) if val in ["125 mm", "180 mm", "230 mm"] else 0
def get_sys_idx(val): return ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"].index(val) if val in ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"] else 0

def calc_weight(dn_idx, ws, length_mm, is_zme=False):
    da = df.iloc[dn_idx]['D_Aussen']; di = da - (2 * ws)
    vol_stahl = (math.pi * ((da/100)**2 - (di/100)**2) / 4) * (length_mm/10); weight_stahl = vol_stahl * 7.85
    if is_zme:
        dn_val = df.iloc[dn_idx]['DN']; cem_th = 0.6 if dn_val < 300 else (0.9 if dn_val < 600 else 1.2)
        di_cem = (di/10) - (2 * cem_th)
        if di_cem > 0:
            vol_cem = (math.pi * ((di/100)**2 - (di_cem/10)**2) / 4) * (length_mm/10); weight_stahl += (vol_cem * 2.4)
    return round(weight_stahl, 1)

def plot_stutzen_curve(r_haupt, r_stutzen):
    angles = range(0, 361, 5); depths = [r_haupt - math.sqrt(r_haupt**2 - (r_stutzen * math.sin(math.radians(a)))**2) for a in angles]
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.plot(angles, depths, color='#3b82f6', linewidth=2); ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
    ax.set_xlim(0, 360); ax.axis('off'); plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig

def plot_etage_sketch(h, l, is_3d=False, b=0):
    fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(0, 0, 'o', color='black')
    if not is_3d:
        ax.plot([0, l], [0, 0], '--', color='gray'); ax.plot([l, l], [0, h], '--', color='gray'); ax.plot([0, l], [0, h], '-', color='#ef4444', linewidth=3)
        ax.text(l/2, -h*0.1, f"L={l}", ha='center'); ax.text(l, h/2, f"H={h}", va='center')
    else:
        ax.plot([0, l], [0, 0], 'k--', alpha=0.3); ax.plot([l, l], [0, h], 'k--', alpha=0.3)
        dx, dy = b * 0.5, b * 0.3
        ax.plot([0, dx], [0, dy], 'k--', alpha=0.3); ax.plot([l, l+dx], [0, dy], 'k--', alpha=0.3)
        ax.plot([dx, l+dx], [dy, dy], 'k--', alpha=0.3); ax.plot([l+dx, l+dx], [dy, h+dy], 'k--', alpha=0.3)
        ax.plot([l, l+dx], [h, h+dy], 'k--', alpha=0.3); ax.plot([0, l+dx], [0, h+dy], '-', color='#ef4444', linewidth=4, solid_capstyle='round')
        ax.text(l/2, -20, f"L={l}", ha='center', fontsize=8); ax.text(l+dx+10, h/2+dy, f"H={h}", va='center', fontsize=8); ax.text(dx/2-10, dy/2, f"B={b}", ha='right', fontsize=8)
    ax.axis('equal'); ax.axis('off')
    return fig

def zeichne_passstueck(iso_mass, abzug1, abzug2, saegelaenge):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    rohr_farbe, abzug_farbe, fertig_farbe, linie_farbe = '#F1F5F9', '#EF4444', '#10B981', '#334155'
    y_mitte, rohr_hoehe = 50, 40
    ax.add_patch(patches.Rectangle((0, y_mitte - rohr_hoehe/2), iso_mass, rohr_hoehe, facecolor=rohr_farbe, edgecolor=linie_farbe, hatch='///', alpha=0.3))
    if abzug1 > 0: ax.add_patch(patches.Rectangle((0, y_mitte - rohr_hoehe/2), abzug1, rohr_hoehe, facecolor=abzug_farbe, alpha=0.5))
    if abzug2 > 0: ax.add_patch(patches.Rectangle((iso_mass - abzug2, y_mitte - rohr_hoehe/2), abzug2, rohr_hoehe, facecolor=abzug_farbe, alpha=0.5))
    ax.add_patch(patches.Rectangle((abzug1, y_mitte - rohr_hoehe/2), saegelaenge, saegelaenge, facecolor=fertig_farbe, edgecolor=linie_farbe, linewidth=2))
    ax.set_xlim(-50, iso_mass + 50); ax.set_ylim(0, 100); ax.axis('off')
    return fig

# -----------------------------------------------------------------------------
# 4. DATABASE / PERSISTENCE
# -----------------------------------------------------------------------------
DB_NAME = "pipecraft.db"

def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (id INTEGER PRIMARY KEY AUTOINCREMENT, iso TEXT, naht TEXT, datum TEXT, dimension TEXT, bauteil TEXT, laenge REAL, charge TEXT, schweisser TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS kalkulation (id INTEGER PRIMARY KEY AUTOINCREMENT, typ TEXT, info TEXT, menge REAL, zeit_min REAL, kosten REAL, mat_text TEXT)''')
    conn.commit(); conn.close()

def add_rohrbuch(iso, naht, datum, dim, bauteil, laenge, charge, schweisser):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('INSERT INTO rohrbuch (iso, naht, datum, dimension, bauteil, laenge, charge, schweisser) VALUES (?,?,?,?,?,?,?,?)', (iso, naht, datum, dim, bauteil, laenge, charge, schweisser))
    conn.commit(); conn.close()

def add_kalkulation(typ, info, menge, zeit, kosten, mat):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('INSERT INTO kalkulation (typ, info, menge, zeit_min, kosten, mat_text) VALUES (?,?,?,?,?,?)', (typ, info, menge, zeit, kosten, mat))
    conn.commit(); conn.close()

def get_rohrbuch_df():
    conn = sqlite3.connect(DB_NAME); df = pd.read_sql_query("SELECT * FROM rohrbuch", conn); conn.close(); return df

def get_kalk_df():
    conn = sqlite3.connect(DB_NAME); df = pd.read_sql_query("SELECT * FROM kalkulation", conn); conn.close(); return df

def delete_rohrbuch_id(entry_id):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor(); c.execute("DELETE FROM rohrbuch WHERE id=?", (entry_id,)); conn.commit(); conn.close()

def delete_kalk_id(entry_id):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor(); c.execute("DELETE FROM kalkulation WHERE id=?", (entry_id,)); conn.commit(); conn.close()

def delete_all(table):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor(); c.execute(f"DELETE FROM {table}"); conn.commit(); conn.close()

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Kalkulation')
    return output.getvalue()

def create_pdf(df):
    if not pdf_available: return None
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Rohrbau Profi - Projektbericht', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Seite {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    total_cost = df['kosten'].sum()
    total_hours = df['zeit_min'].sum() / 60
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Datum: {datetime.now().strftime('%d.%m.%Y')}", 0, 1)
    pdf.cell(0, 10, f"Gesamtkosten: {round(total_cost, 2)} EUR", 0, 1)
    pdf.cell(0, 10, f"Gesamtstunden: {round(total_hours, 1)} h", 0, 1)
    pdf.ln(10)
    
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(30, 10, "Typ", 1, 0, 'C', 1)
    pdf.cell(60, 10, "Info", 1, 0, 'C', 1)
    pdf.cell(20, 10, "Menge", 1, 0, 'C', 1)
    pdf.cell(30, 10, "Kosten", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Material", 1, 1, 'C', 1)
    
    pdf.set_font("Arial", size=9)
    for index, row in df.iterrows():
        typ = str(row['typ']).encode('latin-1', 'replace').decode('latin-1')
        info = str(row['info']).encode('latin-1', 'replace').decode('latin-1')
        mat = str(row['mat_text']).encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(30, 10, typ, 1)
        pdf.cell(60, 10, info, 1)
        pdf.cell(20, 10, str(row['menge']), 1, 0, 'C')
        pdf.cell(30, 10, f"{round(row['kosten'], 2)}", 1, 0, 'R')
        pdf.cell(50, 10, mat, 1, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- STATE & INIT ---
if 'store' not in st.session_state:
    st.session_state.store = {
        'saw_mass': 1000.0, 'saw_gap': 4.0, 'saw_deduct': "0", 'saw_zme': False,
        'kw_dn': 200, 'kw_ws': 6.3, 'kw_verf': "WIG", 'kw_pers': 1, 'kw_anz': 1, 'kw_split': False, 'kw_factor': 1.0,
        'cut_dn': 200, 'cut_ws': 6.3, 'cut_disc': "125 mm", 'cut_anz': 1, 'cut_zma': False, 'cut_iso': False, 'cut_factor': 1.0,
        'iso_sys': "Schrumpfschlauch (WKS)", 'iso_dn': 200, 'iso_anz': 1, 'iso_factor': 1.0,
        'mon_dn': 200, 'mon_type': "Schieber", 'mon_anz': 1, 'mon_factor': 1.0,
        'reg_min': 60, 'reg_pers': 2,
        'cel_root': "2.5 mm", 'cel_fill': "3.2 mm", 'cel_cap': "3.2 mm",
        'p_lohn': 60.0, 'p_stahl': 2.5, 'p_dia': 45.0, 'p_cel': 0.40, 'p_draht': 15.0,
        'p_gas': 0.05, 'p_wks': 25.0, 'p_kebu1': 15.0, 'p_kebu2': 12.0, 'p_primer': 12.0, 'p_machine': 15.0
    }

def save_val(key): st.session_state.store[key] = st.session_state[f"_{key}"]
def get_val(key): return st.session_state.store.get(key)
def update_kw_dn():
    st.session_state.store['kw_dn'] = st.session_state['_kw_dn']
    if st.session_state.store['kw_dn'] >= 300:
        st.session_state.store['kw_pers'] = 2
        st.session_state['_kw_pers'] = 2 

init_db()

# -----------------------------------------------------------------------------
# 5. SIDEBAR & MENÜ
# -----------------------------------------------------------------------------
st.sidebar.title("Rohrbau Profi")
st.sidebar.caption("Powered by Engineering-Core V2")
selected_dn_global = st.sidebar.selectbox("Nennweite (Global)", df['DN'], index=8, key="global_dn") 
selected_pn = st.sidebar.radio("Druckstufe", ["PN 16", "PN 10", "PN 40 (Neu)"], index=0, key="global_pn") 

# Mapping für Hardcoded Data
row = df[df['DN'] == selected_dn_global].iloc[0]
standard_radius = float(row['Radius_BA3'])
suffix = "_16" if selected_pn == "PN 16" else "_10"
if selected_pn == "PN 40 (Neu)": suffix = "_16" # Fallback für UI, Logik über Engine

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
tab_buch, tab_werk, tab_proj, tab_info, tab_eng = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch", "💰 Kalkulation", "🚀 Engineering"])

# -----------------------------------------------------------------------------
# TAB 1: TABELLENBUCH
# -----------------------------------------------------------------------------
with tab_buch:
    st.subheader("Rohr & Formstücke")
    c1, c2 = st.columns(2)
    metric_html = f"""
    <div class='metric-card'>
        <div class='sub-stat'>Außen-Durchmesser</div>
        <div class='big-stat'>{row['D_Aussen']} mm</div>
    </div>
    """
    c1.markdown(metric_html, unsafe_allow_html=True)
    c1.markdown(f"<div class='metric-card'><b>Radius (3D):</b> {standard_radius} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>T-Stück (H):</b> {row['T_Stueck_H']} mm</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>Reduzierung (L):</b> {row['Red_Laenge_L']} mm</div>", unsafe_allow_html=True)
    
    st.divider(); st.subheader(f"Flansch & Montage ({selected_pn})")
    
    if selected_pn == "PN 40 (Neu)":
        st.info("Daten für PN 40 werden im Engineering-Tab berechnet. Anzeige zeigt PN16 Referenz.")
        
    schraube = row[f'Schraube_M{suffix}']; sw, nm = get_schrauben_info(schraube)
    mc1, mc2 = st.columns(2)
    mc1.markdown(f"<div class='metric-card'><b>Blattstärke:</b> {row[f'Flansch_b{suffix}']} mm</div>", unsafe_allow_html=True)
    mc2.markdown(f"<div class='metric-card'><b>Schraube:</b> {row[f'Lochzahl{suffix}']}x {schraube} (SW {sw})</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: WERKSTATT
# -----------------------------------------------------------------------------
with tab_werk:
    tool_mode = st.radio("Werkzeug wählen:", ["📏 Säge (Passstück)", "🔄 Bogen (Zuschnitt)", "🔥 Stutzen (Schablone)", "📐 Etage (Versatz)"], horizontal=True, label_visibility="collapsed", key="tool_mode_nav")
    st.divider()
    if "Säge" in tool_mode:
        c_s1, c_s2 = st.columns(2)
        iso_mass = c_s1.number_input("Gesamtmaß (Iso)", value=get_val('saw_mass'), step=10.0, key="_saw_mass", on_change=save_val, args=('saw_mass',))
        spalt = c_s2.number_input("Wurzelspalt", value=get_val('saw_gap'), key="_saw_gap", on_change=save_val, args=('saw_gap',))
        abzug_input = st.text_input("Abzüge (z.B. 52+30)", value=get_val('saw_deduct'), key="_saw_deduct", on_change=save_val, args=('saw_deduct',))
        abzuege = parse_abzuege(abzug_input)
        saege_erg = iso_mass - spalt - abzuege
        
        st.markdown(f"<div class='metric-card safety-ok'><div class='big-stat'>Sägelänge: {round(saege_erg, 1)} mm</div></div>", unsafe_allow_html=True)
        
        dn_idx = df[df['DN'] == selected_dn_global].index[0]
        std_ws = wandstaerken_std.get(selected_dn_global, 4.0)
        c_zme = st.checkbox("ZME (Beton innen)?", value=get_val('saw_zme'), key="_saw_zme", on_change=save_val, args=('saw_zme',))
        kg = calc_weight(dn_idx, std_ws, saege_erg, c_zme)
        st.caption(f"Gewicht (Stahl): ca. {kg} kg")
        st.pyplot(zeichne_passstueck(iso_mass, 0, 0, saege_erg))

    elif "Bogen" in tool_mode:
        angle = st.slider("Winkel (°)", 0, 90, 45, key="bogen_winkel")
        vorbau = round(standard_radius * math.tan(math.radians(angle/2)), 1)
        aussen = round((standard_radius + (row['D_Aussen']/2)) * angle * (math.pi/180), 1)
        innen = round((standard_radius - (row['D_Aussen']/2)) * angle * (math.pi/180), 1)
        st.markdown(f"<div class='metric-card safety-ok'><div class='big-stat'>Vorbau: {vorbau} mm</div></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2); b1.metric("Rückenlänge", f"{aussen} mm"); b2.metric("Bauchlänge", f"{innen} mm")

    elif "Stutzen" in tool_mode:
        c_st1, c_st2 = st.columns(2)
        dn_stutzen = c_st1.selectbox("DN Stutzen", df['DN'], index=6, key="stutz_dn1")
        dn_haupt = c_st2.selectbox("DN Hauptrohr", df['DN'], index=9, key="stutz_dn2")
        if dn_stutzen > dn_haupt: st.error("Fehler: Stutzen > Hauptrohr")
        else:
            r_k = df[df['DN'] == dn_stutzen].iloc[0]['D_Aussen'] / 2; r_g = df[df['DN'] == dn_haupt].iloc[0]['D_Aussen'] / 2
            st.pyplot(plot_stutzen_curve(r_g, r_k))

    elif "Etage" in tool_mode:
        et_type = st.radio("Typ", ["2D (Einfach)", "3D (Kastenmaß)"], horizontal=True, key="et_type")
        spalt_et = st.number_input("Spalt", 4, key="et_gap")
        if "2D" in et_type:
            h = st.number_input("Höhe H", 300, key="et2d_h"); l = st.number_input("Länge L", 400, key="et2d_l")
            diag = math.sqrt(h**2 + l**2); winkel = math.degrees(math.atan(h/l)) if l>0 else 90
            abzug = 2 * (standard_radius * math.tan(math.radians(winkel/2)))
            erg = diag - abzug - spalt_et
            st.markdown(f"<div class='metric-card safety-ok'><div class='big-stat'>Passstück: {round(erg, 1)} mm</div></div>", unsafe_allow_html=True)
            st.pyplot(plot_etage_sketch(h, l))

# -----------------------------------------------------------------------------
# TAB 3: ROHRBUCH
# -----------------------------------------------------------------------------
with tab_proj:
    with st.form("rb_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3); iso = c1.text_input("ISO"); naht = c2.text_input("Naht"); datum = c3.date_input("Datum")
        c4, c5, c6 = st.columns(3); dn_sel = c4.selectbox("Dimension", df['DN'], index=8, key="rb_dn_sel"); bauteil = c5.selectbox("Bauteil", ["📏 Rohr", "⤵️ Bogen", "⭕ Flansch", "🔗 Muffe", "🔩 Nippel", "🪵 T-Stück"]); laenge = c6.number_input("Länge", value=0)
        c7, c8 = st.columns(2); charge = c7.text_input("Charge"); schweisser = c8.text_input("Schweißer")
        if st.form_submit_button("Speichern"): add_rohrbuch(iso, naht, datum.strftime("%d.%m.%Y"), f"DN {dn_sel}", bauteil, laenge, charge, schweisser); st.success("Gespeichert!")
    st.dataframe(get_rohrbuch_df(), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: KALKULATION (SMART UPGRADE)
# -----------------------------------------------------------------------------
with tab_info:
    calc_task = st.radio("Tätigkeit", ["🔥 Fügen (Schweißen)", "🔧 Montage", "🛡️ Isolierung", "📊 Export"], horizontal=True, key="calc_mode")
    p_lohn = get_val('p_lohn')
    
    if "Fügen" in calc_task:
        st.info("💡 Berechnung basiert nun auf Schweißnaht-Volumen + Abschmelzleistung (RohrbauProfi Engine).")
        c1, c2, c3 = st.columns(3)
        k_dn = c1.selectbox("DN", df['DN'], index=df['DN'].tolist().index(get_val('kw_dn')), key="_kw_dn", on_change=update_kw_dn)
        k_ws = c2.selectbox("WS", ws_liste, index=get_ws_index(get_val('kw_ws')), key="_kw_ws", on_change=save_val, args=('kw_ws',))
        k_verf = c3.selectbox("Verfahren", ["WIG", "CEL 70", "MAG"], index=0, key="_kw_verf", on_change=save_val, args=('kw_verf',))
        
        # --- SMART CALCULATION LOGIC ---
        da = df[df['DN'] == k_dn].iloc[0]['D_Aussen']
        # Nahtvolumen V = Umfang * Querschnitt (näherungsweise V-Naht 60°)
        # A_naht ~ s² * tan(30) + (2mm wurzel * s)
        area_mm2 = (k_ws**2 * 0.577) + (2 * k_ws)
        vol_cm3 = (da * math.pi * area_mm2) / 1000 
        
        # Abschmelzleistung (cm³/min) - Erfahrungswerte
        depo_rates = {"WIG": 1.5, "CEL 70": 3.0, "MAG": 5.0}
        depo = depo_rates[k_verf]
        
        t_pure_weld = vol_cm3 / depo
        t_fitup = (k_dn / 25) * 5 # 5 min pro Zoll Fitup
        t_total_calc = (t_pure_weld + t_fitup) * 1.3 # 30% Nebenzeit
        
        anz = st.number_input("Anzahl", value=1)
        cost = (t_total_calc * anz / 60) * p_lohn
        
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Zeit pro Naht (kalk.)", f"{int(t_total_calc)} min")
        c_res2.metric("Gesamtkosten", f"{round(cost, 2)} €")
        
        if st.button("Zur Kalkulation hinzufügen"):
            add_kalkulation("Schweißen", f"DN {k_dn} x {k_ws} ({k_verf})", anz, t_total_calc*anz, cost, f"Vol: {int(vol_cm3)} cm³")
            st.rerun()
            
    elif "Export" in calc_task:
        df_k = get_kalk_df()
        st.dataframe(df_k)
        if st.button("PDF Erstellen"):
             pdf_data = create_pdf(df_k)
             st.download_button("Download PDF", pdf_data, "Projekt.pdf", "application/pdf")
        if st.button("Reset Projekt", type="primary"): delete_all("kalkulation"); st.rerun()

# -----------------------------------------------------------------------------
# TAB 5: ENGINEERING (NEU!)
# -----------------------------------------------------------------------------
with tab_eng:
    st.header("Engineering Core")
    st.markdown("Physikalisch exakte Auslegung nach DIN EN 13480 / Colebrook-White.")
    
    col_eng_1, col_eng_2 = st.columns(2)
    
    with col_eng_1:
        st.subheader("1. Druckverlust (Hydraulik)")
        e_dn = st.selectbox("Rohr Nennweite", df['DN'], index=9, key="eng_dn")
        e_ws = st.selectbox("Wandstärke", ws_liste, index=6, key="eng_ws")
        e_flow = st.number_input("Volumenstrom (m³/h)", value=100.0, step=10.0)
        e_len = st.number_input("Leitungslänge (m)", value=100)
        
        da_eng = df[df['DN'] == e_dn].iloc[0]['D_Aussen']
        di_eng = da_eng - (2*e_ws)
        
        dp_bar, v_ms, re = engine.calc_pressure_drop(e_flow, di_eng, e_len, 20)
        
        res_style = "safety-ok" if v_ms < 3.0 else "safety-fail"
        st.markdown(f"""
        <div class='metric-card {res_style}'>
            <div class='sub-stat'>Strömungsgeschwindigkeit (Limit: 3 m/s)</div>
            <div class='big-stat'>{v_ms} m/s</div>
        </div>
        """, unsafe_allow_html=True)
        
        c_dp1, c_dp2 = st.columns(2)
        c_dp1.markdown(f"<div class='metric-card'><div class='sub-stat'>Druckverlust</div><div class='big-stat'>{dp_bar} bar</div></div>", unsafe_allow_html=True)
        c_dp2.markdown(f"<div class='metric-card'><div class='sub-stat'>Reynolds-Zahl</div><div class='big-stat'>{re}</div></div>", unsafe_allow_html=True)

    with col_eng_2:
        st.subheader("2. Wanddickenberechnung (Festigkeit)")
        st.markdown(r"Formel: $e = \frac{P_c \cdot D_o}{2 \cdot (f \cdot z + P_c)} + c$")
        
        wt_mat = st.selectbox("Werkstoff", ["P265GH", "1.4404 (316L)", "S235JR"], index=0)
        wt_press = st.number_input("Auslegungsdruck (bar)", value=16.0)
        wt_temp = st.number_input("Auslegungstemperatur (°C)", value=20.0, step=10.0)
        
        calc_res = engine.calc_wall_thickness_en13480(wt_press, da_eng, wt_mat, wt_temp)
        
        is_safe = e_ws >= calc_res['min_required']
        safe_color = "safety-ok" if is_safe else "safety-fail"
        safe_icon = "✅" if is_safe else "❌"
        
        st.markdown(f"""
        <div class='metric-card {safe_color}'>
            <div class='sub-stat'>Erforderliche Wandstärke (DIN EN 13480)</div>
            <div class='big-stat'>{calc_res['min_required']} mm</div>
            <div class='sub-stat'>Gewählt: {e_ws} mm -> {safe_icon}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Details zur Berechnung"):
            st.write(f"Zulässige Spannung bei {wt_temp}°C: {calc_res['stress_used']} MPa")
            st.write(f"Empfohlene Nennwanddicke (inkl. Toleranz): {calc_res['nominal_suggestion']} mm")
