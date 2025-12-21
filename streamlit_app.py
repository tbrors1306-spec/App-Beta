import streamlit as st
import pandas as pd
import math
import sqlite3
import logging
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import matplotlib.pyplot as plt

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
logger = logging.getLogger("PipeCraft_Pro_V5_2")

st.set_page_config(
    page_title="Rohrbau Profi 5.2",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e293b; font-family: 'Segoe UI', sans-serif; }
    
    /* Boxen Styles */
    .success-box {
        padding: 20px;
        background-color: #dcfce7;
        color: #166534;
        border-radius: 8px;
        border-left: 5px solid #22c55e;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-box {
        padding: 20px;
        background-color: #fee2e2;
        color: #991b1b;
        border-radius: 8px;
        border-left: 5px solid #ef4444;
        margin: 10px 0;
        text-align: center;
    }
    .info-box {
        padding: 15px;
        background-color: #eff6ff;
        color: #1e40af;
        border-radius: 6px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
    }
    
    /* Metriken hervorheben */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATEN-SCHICHT
# -----------------------------------------------------------------------------

@st.cache_data
def get_pipe_data() -> pd.DataFrame:
    """Lädt die statischen Rohdaten."""
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
    """Verwaltet Datenbankoperationen."""
    
    @staticmethod
    def init_db():
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        iso TEXT, naht TEXT, datum TEXT, 
                        dimension TEXT, bauteil TEXT, laenge REAL, 
                        charge TEXT, charge_apz TEXT, schweisser TEXT)''')
            
            # Migration APZ
            c.execute("PRAGMA table_info(rohrbuch)")
            cols = [info[1] for info in c.fetchall()]
            if 'charge_apz' not in cols:
                try: c.execute("ALTER TABLE rohrbuch ADD COLUMN charge_apz TEXT")
                except: pass
            conn.commit()

    @staticmethod
    def add_entry(data: dict):
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO rohrbuch 
                         (iso, naht, datum, dimension, bauteil, laenge, charge, charge_apz, schweisser) 
                         VALUES (:iso, :naht, :datum, :dimension, :bauteil, :laenge, :charge, :charge_apz, :schweisser)''', 
                         data)
            conn.commit()

    @staticmethod
    def get_all() -> pd.DataFrame:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query("SELECT * FROM rohrbuch ORDER BY id DESC", conn)
            if not df.empty: df['Löschen'] = False 
            else: 
                df = pd.DataFrame(columns=["id", "iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser", "Löschen"])
            return df

    @staticmethod
    def delete_entries(ids: List[int]):
        if not ids: return
        with sqlite3.connect(DB_NAME) as conn:
            placeholders = ', '.join('?' for _ in ids)
            conn.cursor().execute(f"DELETE FROM rohrbuch WHERE id IN ({placeholders})", ids)
            conn.commit()

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
    def total_deduction(self) -> float:
        return self.deduction_single * self.count

@dataclass
class SavedCut:
    id: int
    raw_length: float
    cut_length: float
    details: str
    timestamp: str

class PipeCalculator:
    """Zentrale Logik für Säge, Bogen und Stutzen."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_row(self, dn: int) -> pd.Series:
        row = self.df[self.df['DN'] == dn]
        return row.iloc[0] if not row.empty else self.df.iloc[0]

    # --- SÄGE (Z-Maße) ---
    def get_deduction(self, f_type: str, dn: int, pn: str, angle: float = 90.0) -> float:
        row = self.get_row(dn)
        suffix = "_16" if pn == "PN 16" else "_10"
        
        if "Bogen 90°" in f_type: return float(row['Radius_BA3'])
        if "Zuschnitt" in f_type: return float(row['Radius_BA3']) * math.tan(math.radians(angle / 2))
        if "Flansch" in f_type: return float(row[f'Flansch_b{suffix}'])
        if "T-Stück" in f_type: return float(row['T_Stueck_H'])
        if "Reduzierung" in f_type: return float(row['Red_Laenge_L'])
        return 0.0

    # --- BOGEN-RECHNER ---
    def calculate_bend_details(self, dn: int, angle: float) -> Dict[str, float]:
        """Berechnet alle relevanten Bogenmaße für die Geometrie-Ansicht."""
        row = self.get_row(dn)
        r = float(row['Radius_BA3'])
        da = float(row['D_Aussen'])
        rad = math.radians(angle)
        
        return {
            "vorbau": r * math.tan(rad / 2),
            "bogen_aussen": (r + da/2) * rad,
            "bogen_mitte": r * rad,
            "bogen_innen": (r - da/2) * rad
        }

    # --- STUTZEN (Sinus) ---
    def calculate_stutzen_coords(self, dn_haupt: int, dn_stutzen: int) -> pd.DataFrame:
        r_main = self.get_row(dn_haupt)['D_Aussen'] / 2
        r_stub = self.get_row(dn_stutzen)['D_Aussen'] / 2

        if r_stub > r_main: raise ValueError(f"Stutzen DN {dn_stutzen} ist größer als Hauptrohr!")

        table_data = []
        for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
            try:
                term = r_stub * math.sin(math.radians(angle))
                t_val = r_main - math.sqrt(r_main**2 - term**2)
            except: t_val = 0
            u_val = (r_stub * 2 * math.pi) * (angle / 360)
            table_data.append({"Winkel": f"{angle}°", "Tiefe (mm)": round(t_val, 1), "Umfang (mm)": round(u_val, 1)})
        return pd.DataFrame(table_data)

class HandbookCalculator:
    """
    NEU: Spezialisierte Logik für das technische Tabellenbuch (Smart Data).
    """
    
    # M-Gewinde: [SW, Nm_Trocken, Nm_Geschmiert(Molykote)]
    BOLT_DATA = {
        "M12": [19, 85, 55],    
        "M16": [24, 210, 135],
        "M20": [30, 410, 265],
        "M24": [36, 710, 460],
        "M27": [41, 1050, 680],
        "M30": [46, 1420, 920],
        "M33": [50, 1930, 1250],
        "M36": [55, 2480, 1600],
        "M39": [60, 3200, 2080],
        "M45": [70, 5000, 3250],
        "M52": [80, 7700, 5000]
    }

    @staticmethod
    def calculate_weight(od: float, wall: float, length: float) -> dict:
        """Berechnet Stahlgewicht und Wasserfüllung (Hydrotest)."""
        if wall <= 0: return {"steel": 0, "water": 0, "total": 0}
        
        # Stahl-Dichte ca. 7.85 kg/dm³
        id_mm = od - (2 * wall)
        vol_steel_m = (math.pi * (od**2 - id_mm**2) / 4) / 1_000_000 # m² Querschnitt
        weight_steel_kgm = vol_steel_m * 7850 # kg/m
        
        # Wasser
        vol_water_m = (math.pi * (id_mm**2) / 4) / 1_000_000 # m²
        weight_water_kgm = vol_water_m * 1000 # kg/m (Dichte 1.0)
        
        return {
            "kg_per_m_steel": weight_steel_kgm,
            "kg_per_m_water": weight_water_kgm,
            "total_steel": weight_steel_kgm * (length / 1000),
            "total_filled": (weight_steel_kgm + weight_water_kgm) * (length / 1000),
            "volume_l": vol_water_m * (length / 1000) * 1000 # in Liter
        }

    @staticmethod
    def get_bolt_length(flange_thk_1: float, flange_thk_2: float, bolt_dim: str, washers: int = 2, gasket: float = 2.0) -> int:
        """Berechnet die Schraubenlänge dynamisch."""
        try:
            d = int(bolt_dim.replace("M", ""))
            h_nut = d * 0.8
            h_washer = 4.0
            overhang = max(6.0, d * 0.4) 
            
            calc_len = flange_thk_1 + flange_thk_2 + gasket + (washers * h_washer) + h_nut + overhang
            
            # Aufrunden auf nächste 5mm
            remainder = calc_len % 5
            if remainder != 0:
                calc_len += (5 - remainder)
            
            return int(calc_len)
        except:
            return 0

class Visualizer:
    @staticmethod
    def plot_stutzen(dn_haupt: int, dn_stutzen: int, df_pipe: pd.DataFrame) -> plt.Figure:
        row_h = df_pipe[df_pipe['DN'] == dn_haupt].iloc[0]
        row_s = df_pipe[df_pipe['DN'] == dn_stutzen].iloc[0]
        r_main = row_h['D_Aussen'] / 2
        r_stub = row_s['D_Aussen'] / 2
        
        if r_stub > r_main: return None

        angles = range(0, 361, 5)
        depths = []
        for a in angles:
            try:
                term = r_stub * math.sin(math.radians(a))
                depths.append(r_main - math.sqrt(r_main**2 - term**2))
            except: depths.append(0)

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360)
        ax.set_ylabel("Tiefe (mm)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

class Exporter:
    @staticmethod
    def to_excel(df):
        output = BytesIO()
        export_df = df.drop(columns=['Löschen', 'id'], errors='ignore')
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Daten')
        return output.getvalue()

    @staticmethod
    def to_pdf(df):
        if not PDF_AVAILABLE: return b""
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Rohrbuch - {datetime.now().strftime('%d.%m.%Y')}", 0, 1, 'C')
        pdf.ln(5)
        
        cols = ["ISO", "Naht", "Datum", "DN", "Bauteil", "Charge", "APZ", "Schweißer"]
        widths = [30, 20, 25, 20, 40, 35, 35, 30]
        pdf.set_font("Arial", 'B', 8)
        for i, c in enumerate(cols): pdf.cell(widths[i], 8, c, 1)
        pdf.ln()
        
        pdf.set_font("Arial", size=8)
        export_df = df.drop(columns=['Löschen', 'id'], errors='ignore')
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

# -----------------------------------------------------------------------------
# 4. UI SEITEN (TABS)
# -----------------------------------------------------------------------------

def render_smart_saw(calc: PipeCalculator, df: pd.DataFrame, current_dn: int, pn: str):
    """Smarte Säge V4.1"""
    st.subheader("🪚 Smarte Säge 5.2")
    
    # State Healing
    if 'fitting_list' in st.session_state and st.session_state.fitting_list:
        try: _ = st.session_state.fitting_list[0].id
        except AttributeError: st.session_state.fitting_list = []

    if 'fitting_list' not in st.session_state: st.session_state.fitting_list = []
    if 'saved_cuts' not in st.session_state: st.session_state.saved_cuts = []
    if 'next_cut_id' not in st.session_state: st.session_state.next_cut_id = 1

    c_calc, c_list = st.columns([1.5, 1.5])

    with c_calc:
        with st.container(border=True):
            st.markdown("#### 1. Zuschnitt")
            raw_len = st.number_input("Schnittmaß aus Plan [mm]", min_value=0.0, step=10.0, format="%.1f")
            
            cg1, cg2, cg3 = st.columns(3)
            gap = cg1.number_input("Wurzelspalt (mm)", value=3.0, step=0.5)
            dicht_anz = cg2.number_input("Anz. Dicht.", 0, 5, 0)
            dicht_thk = cg3.number_input("Dicke (mm)", 0.0, 5.0, 2.0, disabled=(dicht_anz==0))
            
            st.divider()
            st.caption("Bauteil abziehen:")
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
                st.markdown("###### Liste:")
                for i, item in enumerate(st.session_state.fitting_list):
                    cr1, cr2, cr3 = st.columns([3, 1.5, 0.5])
                    cr1.text(f"{item.count}x {item.name}")
                    cr2.text(f"-{item.total_deduction:.1f}")
                    if cr3.button("x", key=f"d_{item.id}"):
                        st.session_state.fitting_list.pop(i)
                        st.rerun()
                if st.button("Alles löschen", type="secondary"):
                    st.session_state.fitting_list = []
                    st.rerun()

            sum_fit = sum(i.total_deduction for i in st.session_state.fitting_list)
            sum_gap = sum(i.count for i in st.session_state.fitting_list) * gap
            sum_gskt = dicht_anz * dicht_thk
            total = sum_fit + sum_gap + sum_gskt
            final = raw_len - total

            st.divider()
            if final < 0:
                st.markdown(f"<div class='error-box'>Negativmaß!<br>{final:.1f} mm</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='success-box'>Sägelänge<br><b>{final:.1f} mm</b></div>", unsafe_allow_html=True)
                st.caption(f"Details: Teile -{sum_fit:.1f} | Spalte -{sum_gap:.1f} | Dicht. -{sum_gskt:.1f}")
                
                if st.button("💾 Speichern", type="primary", use_container_width=True):
                    if raw_len > 0:
                        st.session_state.saved_cuts.append(SavedCut(st.session_state.next_cut_id, raw_len, final, f"{len(st.session_state.fitting_list)} Teile", datetime.now().strftime("%H:%M")))
                        st.session_state.next_cut_id += 1
                        st.session_state.fitting_list = []
                        st.rerun()

    with c_list:
        st.markdown("#### 📋 Gespeichert")
        if st.session_state.saved_cuts:
            data = [asdict(c) for c in st.session_state.saved_cuts]
            df_s = pd.DataFrame(data).drop(columns=['id']).rename(columns={'raw_length': 'Rohr', 'cut_length': 'Schnitt'})
            st.dataframe(df_s, use_container_width=True, hide_index=True)
            if st.button("Liste leeren"):
                st.session_state.saved_cuts = []
                st.rerun()

def render_geometry_tools(calc: PipeCalculator, df: pd.DataFrame):
    """Geometrie Tools inkl. BOGEN MITTE."""
    st.subheader("📐 Geometrie & Schablonen")
    
    t_stutz, t_bogen = st.tabs(["🔥 Stutzen-Schablone", "🔄 Bogen-Rechner"])
    
    with t_stutz:
        c1, c2 = st.columns(2)
        dn_stub = c1.selectbox("DN Stutzen", df['DN'], index=5, key="gs_dn1")
        dn_main = c2.selectbox("DN Hauptrohr", df['DN'], index=8, key="gs_dn2")
        
        if c1.button("Kurve berechnen"):
            try:
                df_coords = calc.calculate_stutzen_coords(dn_main, dn_stub)
                fig = Visualizer.plot_stutzen(dn_main, dn_stub, df)
                c_res1, c_res2 = st.columns([1, 2])
                with c_res1: st.table(df_coords)
                with c_res2: st.pyplot(fig)
            except ValueError as e:
                st.error(str(e))

    with t_bogen:
        st.markdown("#### Bogen Zuschnitt & Kontrolle")
        cb1, cb2 = st.columns(2)
        angle = cb1.slider("Winkel (°)", 0, 90, 45, key="gb_ang")
        dn_b = cb2.selectbox("DN Bogen", df['DN'], index=6, key="gb_dn")
        
        details = calc.calculate_bend_details(dn_b, angle)
        
        st.divider()
        c_vorbau, c_laengen = st.columns([1, 2])
        
        with c_vorbau:
            st.markdown("**Einbaumaß**")
            st.markdown(f"<div class='info-box'>Vorbau (Z-Maß)<br><b>{details['vorbau']:.1f} mm</b></div>", unsafe_allow_html=True)
            
        with c_laengen:
            st.markdown("**Bogenlängen (Material)**")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Außen (Rücken)", f"{details['bogen_aussen']:.1f} mm")
            cm2.metric("Mitte (Neutral)", f"{details['bogen_mitte']:.1f} mm") 
            cm3.metric("Innen (Bauch)", f"{details['bogen_innen']:.1f} mm")

def render_logbook(df_pipe: pd.DataFrame):
    """Rohrbuch V2.1"""
    st.subheader("📝 Digitales Rohrbuch")
    
    with st.expander("Eintrag hinzufügen", expanded=True):
        with st.form("lb_new"):
            c1, c2, c3 = st.columns(3)
            iso = c1.text_input("ISO")
            naht = c2.text_input("Naht")
            dat = c3.date_input("Datum")
            c4, c5, c6 = st.columns(3)
            bt = c4.selectbox("Bauteil", ["Rohrstoß", "Bogen", "Flansch", "T-Stück", "Stutzen", "Muffe"])
            dn = c5.selectbox("Dimension", df_pipe['DN'], index=8)
            ch = c6.text_input("Charge")
            c7, c8 = st.columns(2)
            apz = c7.text_input("APZ / Zeugnis")
            sch = c8.text_input("Schweißer")
            
            if st.form_submit_button("Speichern 💾", type="primary"):
                DatabaseRepository.add_entry({
                    "iso": iso, "naht": naht, "datum": dat.strftime("%d.%m.%Y"),
                    "dimension": f"DN {dn}", "bauteil": bt, "laenge": 0,
                    "charge": ch, "charge_apz": apz, "schweisser": sch
                })
                st.success("Gespeichert")
                st.rerun()

    st.divider()
    df = DatabaseRepository.get_all()
    if not df.empty:
        ce1, ce2, _ = st.columns([1,1,3])
        ce1.download_button("📥 Excel", Exporter.to_excel(df), "rohrbuch.xlsx")
        if PDF_AVAILABLE: ce2.download_button("📄 PDF", Exporter.to_pdf(df), "rohrbuch.pdf")
            
        edited = st.data_editor(
            df, hide_index=True, use_container_width=True,
            column_config={"Löschen": st.column_config.CheckboxColumn(default=False)},
            disabled=["id", "iso", "naht", "datum", "dimension", "bauteil", "charge", "charge_apz", "schweisser"]
        )
        to_del = edited[edited['Löschen'] == True]
        if not to_del.empty:
            if st.button(f"🗑️ {len(to_del)} löschen", type="primary"):
                DatabaseRepository.delete_entries(to_del['id'].tolist())
                st.rerun()

def render_tab_handbook(calc: PipeCalculator, dn: int, pn: str):
    """
    NEU V5.0: Smart Data Hub für den Großleitungsbau.
    """
    row = calc.get_row(dn)
    suffix = "_16" if pn == "PN 16" else "_10"
    
    st.subheader(f"📚 Smart Data: DN {dn} / {pn}")

    # Basisdaten
    od = float(row['D_Aussen'])
    flange_b = float(row[f'Flansch_b{suffix}'])
    lk = float(row[f'LK_k{suffix}'])
    bolt = row[f'Schraube_M{suffix}']
    n_holes = int(row[f'Lochzahl{suffix}'])
    
    # MODUL 1: ROHR & LASTEN
    with st.expander("🏗️ Rohrgewichte & Hydrotest (Kran/Gerüst)", expanded=True):
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

    # MODUL 2: FLANSCH & DICHTUNG
    c_geo1, c_geo2 = st.columns(2)
    with c_geo1:
        st.markdown("#### 📐 Flansch Geometrie")
        st.write(f"**Blattstärke:** {flange_b} mm")
        st.write(f"**Lochkreis:** {lk} mm")
        st.write(f"**Lochzahl:** {n_holes} x {bolt}")
        st.progress(lk / (od + 100), text=f"Lochkreis Verhältnis")

    with c_geo2:
        st.markdown("#### 🔘 Dichtung (Check)")
        d_innen = od - (2*wt_input) 
        d_aussen = lk - (int(bolt.replace("M","")) * 1.5)
        st.info(f"* Innen-Ø: ~{d_innen:.0f} mm\n* Außen-Ø: ~{d_aussen:.0f} mm\n* Dicke: 2.0 mm (Std)")

    st.divider()

    # MODUL 3: MONTAGEMANAGER
    st.markdown("#### 🔧 Montage & Drehmomente (8.8)")
    cb_col1, cb_col2 = st.columns([1, 2])
    
    with cb_col1:
        st.caption("Konfiguration")
        conn_type = st.radio("Typ", ["Fest-Fest (V-V)", "Fest-Los (V-L)", "Fest-Blind (V-B)"], index=0)
        use_washers = st.checkbox("Unterlegscheiben (2x)", value=True)
        gasket_thk = st.number_input("Dichtung (mm)", value=2.0, step=0.5)
        
    with cb_col2:
        bolt_info = HandbookCalculator.BOLT_DATA.get(bolt, [0, 0, 0])
        sw, nm_dry, nm_lube = bolt_info
        
        t1 = flange_b
        t2 = flange_b 
        if "Los" in conn_type: t2 = flange_b + 5 
        elif "Blind" in conn_type: t2 = flange_b + (dn * 0.02)
            
        n_washers = 2 if use_washers else 0
        calc_len = HandbookCalculator.get_bolt_length(t1, t2, bolt, n_washers, gasket_thk)
        
        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.markdown(f"<div class='success-box' style='padding:10px'>🔩 Bolzen<br><b>{bolt} x {calc_len}</b><br><span style='font-size:0.8em'>{n_holes} Stk.</span></div>", unsafe_allow_html=True)
        res_c2.markdown(f"<div class='info-box' style='text-align:center'>🔧 Schlüssel<br><b>SW {sw} mm</b><br><span style='font-size:0.8em'>Nuss/Ring</span></div>", unsafe_allow_html=True)
        
        is_lubed = st.toggle("Geschmiert (MoS2)?", value=True)
        torque = nm_lube if is_lubed else nm_dry
        style_col = "#dcfce7" if is_lubed else "#fee2e2"
        res_c3.markdown(f"<div style='background-color:{style_col}; padding:10px; border-radius:8px; text-align:center; border:1px solid #ccc'>💪 Moment<br><b>{torque} Nm</b><br><span style='font-size:0.8em'>{'Geschmiert' if is_lubed else 'Trocken'}</span></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------

def main():
    DatabaseRepository.init_db()
    df_pipe = get_pipe_data()
    calc = PipeCalculator(df_pipe)

    with st.sidebar:
        st.title("⚙️ Setup")
        dn = st.selectbox("Nennweite", df_pipe['DN'], index=8)
        pn = st.radio("Druckstufe", ["PN 16", "PN 10"], horizontal=True)

    t1, t2, t3, t4 = st.tabs(["🪚 Smarte Säge", "📐 Geometrie", "📝 Rohrbuch", "📚 Smart Data"])
    
    with t1: render_smart_saw(calc, df_pipe, dn, pn)
    with t2: render_geometry_tools(calc, df_pipe)
    with t3: render_logbook(df_pipe)
    with t4: render_tab_handbook(calc, dn, pn)

if __name__ == "__main__":
    main()
