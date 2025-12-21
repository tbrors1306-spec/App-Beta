"""
PipeCraft Enterprise Edition (V28.0)
------------------------------------
Module: Main Application
Description: Industrial Piping Calculator with 3D Visualization and Cost Estimation.
Architecture: Service-Oriented (Logic separated from UI).
Author: Senior Lead Software Engineer
"""

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Wichtig für 3D-Plots in Matplotlib
from mpl_toolkits.mplot3d import Axes3D 
import sqlite3
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import List, Tuple, Any, Optional, Union, Dict

# -----------------------------------------------------------------------------
# 0. SYSTEM CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------

# Logging konfigurieren für Traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipeCraft")

# Optionale Abhängigkeiten sicher importieren
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("Bibliothek 'fpdf' fehlt. PDF-Export wurde deaktiviert.")

# Streamlit Page Config
st.set_page_config(
    page_title="PipeCraft V28.0 Enterprise",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für professionelles UI-Design
st.markdown("""
<style>
    /* Basis-Layout */
    .stApp { 
        background-color: #f8f9fa; 
        color: #0f172a; 
    }
    h1, h2, h3 { 
        font-family: 'Helvetica Neue', sans-serif; 
        color: #1e293b !important; 
        font-weight: 800; 
        letter-spacing: -0.5px; 
    }
    
    /* Info-Karten (Blau) */
    .result-card-blue { 
        background-color: #eff6ff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #3b82f6; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        margin-bottom: 10px; 
        color: #1e3a8a; 
        font-size: 1rem; 
        line-height: 1.5;
    }
    
    /* Ergebnis-Karten (Grün) */
    .result-card-green { 
        background: linear-gradient(to right, #f0fdf4, #ffffff); 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 8px solid #22c55e; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); 
        margin-bottom: 15px; 
        text-align: center; 
        font-size: 1.6rem; 
        font-weight: 800; 
        color: #14532d; 
    }
    
    /* Detail-Boxen (Grau) */
    .detail-box { 
        background-color: #f1f5f9; 
        border: 1px solid #cbd5e1; 
        padding: 12px; 
        border-radius: 8px; 
        text-align: center; 
        font-size: 0.9rem; 
        color: #334155; 
        height: 100%; 
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
    }
    
    /* Warnung/Gewicht (Rot) */
    .weight-box { 
        background-color: #fff1f2; 
        border: 1px solid #fecdd3; 
        color: #be123c; 
        padding: 12px; 
        border-radius: 8px; 
        text-align: center; 
        font-weight: bold; 
        font-size: 1.1rem; 
        margin-top: 10px; 
    }
    
    /* Formular-Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input { 
        border-radius: 8px; 
        border: 1px solid #cbd5e1; 
    }
    div.stButton > button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 600; 
        border: 1px solid #cbd5e1; 
        transition: all 0.2s; 
    }
    div.stButton > button:hover { 
        border-color: #3b82f6; 
        color: #3b82f6; 
        background-color: #eff6ff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. DATA REPOSITORY (STATIC DATA)
# -----------------------------------------------------------------------------

# Rohdaten für DataFrame
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

# Initialisierung des Pandas DataFrames (Zentraler Datenspeicher)
try:
    df_pipe = pd.DataFrame(RAW_DATA)
except ValueError as e:
    st.error(f"KRITISCHER FEHLER: Datenbank-Inkonsistenz. Listenlängen stimmen nicht überein. Detail: {e}")
    st.stop()

# Zusatz-Datenbanken (Dictionaries)
SCHRAUBEN_DB = { 
    "M12": [18, 60], "M16": [24, 130], "M20": [30, 250], "M24": [36, 420], 
    "M27": [41, 600], "M30": [46, 830], "M33": [50, 1100], "M36": [55, 1400], 
    "M39": [60, 1800], "M45": [70, 2700], "M52": [80, 4200] 
}

WS_LISTE = [2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 8.8, 10.0, 11.0, 12.5, 14.2, 16.0]

# Standard-Wandstärken Mapping für automatische Gewichtsberechnung (Key: DN, Value: WS)
WS_STD_MAP = {
    25: 3.2, 32: 3.6, 40: 3.6, 50: 3.9, 65: 5.2, 80: 5.5, 100: 6.0, 
    125: 6.6, 150: 7.1, 200: 8.2, 250: 9.3, 300: 9.5, 350: 9.5, 400: 9.5, 
    450: 9.5, 500: 9.5
}

DB_NAME = "pipecraft.db"

# -----------------------------------------------------------------------------
# 2. LOGIC LAYER (HELPER & ENGINES)
# -----------------------------------------------------------------------------

def get_schrauben_info(gewinde: str) -> List[Union[int, str]]:
    """Gibt [Schlüsselweite, Drehmoment] für ein Gewinde zurück."""
    return SCHRAUBEN_DB.get(gewinde, ["?", "?"])

def parse_abzuege(text: str) -> float:
    """Berechnet mathematische Strings sicher (z.B. '50+30')."""
    try:
        if not text:
            return 0.0
        clean_text = text.replace(",", ".").replace(" ", "")
        # Whitelist-Prüfung für Sicherheit
        if not all(c in "0123456789.+-*/()" for c in clean_text):
            return 0.0
        return float(pd.eval(clean_text))
    except Exception:
        return 0.0

# UI-Helper für Index-Rückgewinnung (Verhindert Abstürze bei State-Reload)
def get_ws_index(val: float) -> int:
    try: 
        return WS_LISTE.index(val)
    except ValueError: 
        return 6 # Default auf gängige Wandstärke

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

class PhysicsEngine:
    """Berechnet physikalische Eigenschaften (Gewicht, Volumen)."""
    
    DENSITY_STEEL = 7.85 # kg/dm^3
    DENSITY_CEMENT = 2.40 # kg/dm^3
    
    @staticmethod
    def calculate_pipe_weight(dn_idx: int, ws: float, length_mm: float, is_zme: bool = False) -> float:
        """
        Berechnet das Rohrgewicht inkl. optionaler Zementauskleidung.
        
        Args:
            dn_idx: Index der DN in der Datenbank
            ws: Wandstärke in mm
            length_mm: Länge in mm
            is_zme: Boolean für Zementmörtelauskleidung
        """
        try:
            da_mm = df_pipe.iloc[dn_idx]['D_Aussen']
            
            # Einheitenumrechnung für Dichte-Formel (mm -> dm)
            # 1 dm = 100 mm -> radius in dm = r_mm / 100
            # 1 dm = 100 mm -> length in dm = l_mm / 100  <-- KORREKTUR: l_dm = l_mm / 100 wäre falsch. 1m = 10dm = 1000mm. Also /100.
            # Aber wir rechnen Volumen V = Pi * r^2 * h.
            # r in dm, h in dm -> V in dm³.
            
            length_dm = length_mm / 100.0
            ra_dm = (da_mm / 2) / 100.0
            ri_stahl_dm = ra_dm - (ws / 100.0)
            
            # Stahl-Masse (Hohlzylinder)
            vol_stahl = math.pi * (ra_dm**2 - ri_stahl_dm**2) * length_dm
            weight = vol_stahl * PhysicsEngine.DENSITY_STEEL
            
            # ZME-Masse (Zusatz)
            if is_zme:
                dn_val = df_pipe.iloc[dn_idx]['DN']
                # DIN 2614 Schätzung für Schichtdicke
                cem_th_mm = 6.0 if dn_val < 300 else (9.0 if dn_val < 600 else 12.0)
                ri_cem_dm = ri_stahl_dm - (cem_th_mm / 100.0)
                
                if ri_cem_dm > 0:
                    vol_cem = math.pi * (ri_stahl_dm**2 - ri_cem_dm**2) * length_dm
                    weight += (vol_cem * PhysicsEngine.DENSITY_CEMENT)
                    
            return round(weight, 1)
        except Exception as e:
            logger.error(f"Weight Calc Error: {e}")
            return 0.0

class GeometryEngine:
    """Berechnet Raumgeometrie, Isometrie und Zuschnitte."""
    
    @staticmethod
    def solve_offset_3d(h: float, l: float, b: float) -> Tuple[float, float]:
        """
        Löst das Raumdreieck für eine Etage.
        
        Args:
            h: Höhe (Vertical Offset)
            l: Länge (Horizontal Offset)
            b: Breite (Depth Offset / Sprung)
            
        Returns:
            Tuple: (Travel-Länge, Raumwinkel Alpha zur Horizontalen)
        """
        # Travel (Diagonale im Raum - Vektorlänge)
        travel = math.sqrt(h**2 + l**2 + b**2)
        
        # Spread (Diagonale am Boden - Projektion)
        spread = math.sqrt(l**2 + b**2)
        
        # Schutz vor Division durch Null
        if spread == 0:
            angle = 90.0
        else:
            angle = math.degrees(math.atan(h / spread))
            
        return travel, angle

class CostEngine:
    """Berechnet Zeit und Kosten für Arbeitsschritte."""
    
    @staticmethod
    def estimate_welding_time(dn: int, ws: float, process: str, layers: int = 1) -> float:
        """
        Schätzt Schweißzeit in Minuten basierend auf 'Inch-Diameter' Regel.
        """
        zoll = dn / 25.0
        
        # Basiszeit pro Zoll in Minuten (Erfahrungswerte)
        if "WIG" in process:
            base_min = 12.0
        elif "CEL" in process:
            base_min = 4.5
        else:
            base_min = 8.0 # MAG/E-Hand Standard
            
        # Wandstärken-Korrektur (ab Standard 6mm)
        ws_factor = ws / 6.0 if ws > 6.0 else 1.0
        
        return (zoll * base_min * ws_factor)

# -----------------------------------------------------------------------------
# 3. VISUALIZATION LAYER (PLOTTING)
# -----------------------------------------------------------------------------

class Visualizer:
    """Zuständig für alle Grafikausgaben (2D & 3D)."""
    
    @staticmethod
    def plot_stutzen_curve(r_haupt: float, r_stutzen: float) -> plt.Figure:
        """Erstellt den Plot für die Stutzen-Abwicklung (Mantelkurve)."""
        angles = range(0, 361, 5)
        try:
            # Formel für Durchdringungskurve Zylinder-Zylinder
            depths = [r_haupt - math.sqrt(r_haupt**2 - (r_stutzen * math.sin(math.radians(a)))**2) for a in angles]
        except ValueError:
            return plt.figure() # Leerer Plot bei mathematischem Fehler (Stutzen > Haupt)

        fig, ax = plt.subplots(figsize=(8, 1.2)) # Flaches Format für UI
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360)
        ax.axis('off') # Kein Rahmen, cleanes Design
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

    @staticmethod
    def plot_true_3d_pipe(length: float, width: float, height: float, azim: int, elev: int) -> plt.Figure:
        """
        Rendert eine echte 3D-Szene der Rohrleitung mittels mplot3d.
        """
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        # Vektor-Koordinaten (Start -> Ende)
        xs = [0, length]
        ys = [0, width]
        zs = [0, height]
        
        # Das Rohr (Fette rote Linie)
        ax.plot(xs, ys, zs, color='#ef4444', linewidth=5, solid_capstyle='round', label='Rohrachse')
        
        # Bounding Box (Hilfslinien für räumliches Verständnis)
        # 1. Bodenprojektion (Länge)
        ax.plot([0, length], [0, 0], [0, 0], 'k--', alpha=0.2, lw=1) 
        # 2. Verbindung zur Breite
        ax.plot([length, length], [0, width], [0, 0], 'k--', alpha=0.2, lw=1) 
        # 3. Schatten Rückwand
        ax.plot([0, length], [width, width], [0, 0], 'k--', alpha=0.1, lw=1) 
        
        # 4. Vertikale Projektion (Drop Line vom Zielpunkt)
        ax.plot([length, length], [width, width], [0, height], 'k--', alpha=0.3, lw=1) 
        
        # Start- und Endpunkt markieren
        ax.scatter([0], [0], [0], color='black', s=50, label="Start")
        ax.scatter([length], [width], [height], color='#10b981', s=50, label="Ende")
        
        # Achsen Beschriftung
        ax.set_xlabel('Länge (X)')
        ax.set_ylabel('Breite (Y)')
        ax.set_zlabel('Höhe (Z)')
        
        # Aspect Ratio Hack für matplotlib 3D (damit es nicht verzerrt ist)
        # Wir ermitteln das Maximum aller Dimensionen und setzen alle Limits gleich.
        max_dim = max(abs(length), abs(width), abs(height))
        if max_dim == 0: max_dim = 100 # Fallback
        
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        
        # Kamera View setzen
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_etage_2d(h: float, l: float) -> plt.Figure:
        """Klassische 2D-Ansicht für einfache Etagen."""
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(0, 0, 'o', color='black') # Start
        
        # Hilfslinien
        ax.plot([0, l], [0, 0], '--', color='gray')
        ax.plot([l, l], [0, h], '--', color='gray')
        
        # Vektor
        ax.plot([0, l], [0, h], '-', color='#ef4444', linewidth=3)
        
        # Text
        ax.text(l/2, -10, f"L={l}", ha='center')
        ax.text(l+10, h/2, f"H={h}", va='center')
        
        ax.axis('equal')
        ax.axis('off')
        return fig

# -----------------------------------------------------------------------------
# 4. DATABASE SERVICE (PERSISTENCE)
# -----------------------------------------------------------------------------

class DatabaseRepository:
    """Verwaltet alle Datenbank-Interaktionen sicher."""
    
    @staticmethod
    def init_tables():
        """Erstellt Tabellen falls nicht vorhanden."""
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        iso TEXT, naht TEXT, datum TEXT, 
                        dimension TEXT, bauteil TEXT, 
                        laenge REAL, charge TEXT, schweisser TEXT)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS kalkulation (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        typ TEXT, info TEXT, 
                        menge REAL, zeit_min REAL, 
                        kosten REAL, mat_text TEXT)''')
            conn.commit()

    @staticmethod
    def add_rohrbuch_entry(data: Tuple):
        """Fügt Eintrag ins Rohrbuch hinzu."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(
                'INSERT INTO rohrbuch (iso, naht, datum, dimension, bauteil, laenge, charge, schweisser) VALUES (?,?,?,?,?,?,?,?)', 
                data
            )
            conn.commit()

    @staticmethod
    def add_kalkulation_entry(data: Tuple):
        """Fügt Eintrag zur Kalkulation hinzu."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(
                'INSERT INTO kalkulation (typ, info, menge, zeit_min, kosten, mat_text) VALUES (?,?,?,?,?,?)', 
                data
            )
            conn.commit()

    @staticmethod
    def get_all(table_name: str) -> pd.DataFrame:
        """Liest ganze Tabelle als DataFrame."""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            logger.error(f"DB Read Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def delete_by_id(table_name: str, entry_id: int):
        """Löscht einen Eintrag."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(f"DELETE FROM {table_name} WHERE id=?", (entry_id,))
            conn.commit()

    @staticmethod
    def clear_table(table_name: str):
        """Löscht alle Einträge."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(f"DELETE FROM {table_name}")
            conn.commit()

# --- EXPORT SERVICES ---

def export_to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Kalkulation')
    return output.getvalue()

def export_to_pdf(df: pd.DataFrame) -> bytes:
    if not PDF_AVAILABLE:
        return b""
        
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'PipeCraft - Projektbericht', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Seite {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Metadaten
    total_cost = df['kosten'].sum()
    total_hours = df['zeit_min'].sum() / 60
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Datum: {datetime.now().strftime('%d.%m.%Y')}", 0, 1)
    pdf.cell(0, 10, f"Gesamtkosten: {round(total_cost, 2)} EUR", 0, 1)
    pdf.cell(0, 10, f"Gesamtstunden: {round(total_hours, 1)} h", 0, 1)
    pdf.ln(10)
    
    # Tabelle Header
    pdf.set_fill_color(220, 230, 255)
    pdf.set_font("Arial", 'B', 10)
    
    headers = [("Typ", 30), ("Info", 60), ("Menge", 20), ("Kosten", 30), ("Material", 50)]
    for text, width in headers:
        pdf.cell(width, 10, text, 1, 0, 'C', 1)
    pdf.ln()
    
    # Tabelle Body
    pdf.set_font("Arial", size=9)
    for _, row in df.iterrows():
        try:
            # Safe String Conversion
            vals = [
                str(row['typ']), 
                str(row['info']), 
                str(row['menge']), 
                f"{round(row['kosten'], 2)}", 
                str(row['mat_text'])
            ]
            
            # Decoding für PDF (latin-1 limitation workaround)
            vals = [v.encode('latin-1', 'replace').decode('latin-1') for v in vals]
            
            pdf.cell(30, 10, vals[0], 1)
            pdf.cell(60, 10, vals[1], 1)
            pdf.cell(20, 10, vals[2], 1, 0, 'C')
            pdf.cell(30, 10, vals[3], 1, 0, 'R')
            pdf.cell(50, 10, vals[4], 1, 1)
        except Exception:
            continue
            
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 5. APPLICATION STATE & INITIALIZATION
# -----------------------------------------------------------------------------

# Datenbank starten
DatabaseRepository.init_tables()

# Session State Default-Werte
DEFAULT_STATE = {
    'saw_mass': 1000.0, 'saw_gap': 4.0, 'saw_deduct': "0", 'saw_zme': False,
    'kw_dn': 200, 'kw_ws': 6.3, 'kw_verf': "WIG", 'kw_pers': 1, 'kw_anz': 1, 'kw_split': False, 'kw_factor': 1.0,
    'cut_dn': 200, 'cut_ws': 6.3, 'cut_disc': "125 mm", 'cut_anz': 1, 'cut_zma': False, 'cut_iso': False, 'cut_factor': 1.0,
    'iso_sys': "Schrumpfschlauch (WKS)", 'iso_dn': 200, 'iso_anz': 1, 'iso_factor': 1.0,
    'mon_dn': 200, 'mon_type': "Schieber", 'mon_anz': 1, 'mon_factor': 1.0,
    'reg_min': 60, 'reg_pers': 2, 
    'bogen_winkel': 45,
    'view_azim': 45, 'view_elev': 30, # 3D View Defaults
    
    # Globale Preiseinstellungen
    'p_lohn': 60.0, 'p_stahl': 2.5, 'p_dia': 45.0, 'p_cel': 0.40, 'p_draht': 15.0,
    'p_gas': 0.05, 'p_wks': 25.0, 'p_kebu1': 15.0, 'p_kebu2': 12.0, 'p_primer': 12.0, 
    'p_machine': 15.0
}

# State initialisieren wenn leer
if 'store' not in st.session_state:
    st.session_state.store = DEFAULT_STATE.copy()

# Callbacks
def save_val(key: str) -> None:
    """Speichert den aktuellen Widget-Wert in den persistenten Store."""
    if f"_{key}" in st.session_state:
        st.session_state.store[key] = st.session_state[f"_{key}"]

def get_val(key: str) -> Any:
    """Holt Wert aus Store."""
    return st.session_state.store.get(key, DEFAULT_STATE.get(key))

def update_kw_dn() -> None:
    """Spezial-Logik: Wenn DN >= 300, setze Personen auf 2."""
    st.session_state.store['kw_dn'] = st.session_state['_kw_dn']
    if st.session_state.store['kw_dn'] >= 300:
        st.session_state.store['kw_pers'] = 2
        st.session_state['_kw_pers'] = 2 # Update des Widgets erzwingen

# -----------------------------------------------------------------------------
# 6. USER INTERFACE (MAIN)
# -----------------------------------------------------------------------------

# Sidebar Menü
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=50) 
st.sidebar.markdown("### Menü")
selected_dn_global = st.sidebar.selectbox("Nennweite (Global)", df_pipe['DN'], index=8, key="global_dn") 
selected_pn = st.sidebar.radio("Druckstufe", ["PN 16", "PN 10"], index=0, key="global_pn") 

# Aktiver Kontext (Datenzeile)
row = df_pipe[df_pipe['DN'] == selected_dn_global].iloc[0]
standard_radius = float(row['Radius_BA3'])
suffix = "_16" if selected_pn == "PN 16" else "_10"

st.title("PipeCraft V28.0 (Enterprise)")
st.caption(f"🔧 Aktive Konfiguration: DN {selected_dn_global} | {selected_pn} | Radius: {standard_radius} mm")

# Haupt-Tabs
tab_buch, tab_werk, tab_proj, tab_info = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch", "💰 Kalkulation"])

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
    
    # 2.1 SÄGE / PASSSTÜCK
    if "Säge" in tool_mode:
        st.subheader("Passstück Berechnung")
        c_s1, c_s2 = st.columns(2)
        
        iso_mass = c_s1.number_input("Gesamtmaß (Iso)", value=get_val('saw_mass'), step=10.0, key="_saw_mass", on_change=save_val, args=('saw_mass',))
        spalt = c_s2.number_input("Wurzelspalt", value=get_val('saw_gap'), key="_saw_gap", on_change=save_val, args=('saw_gap',))
        abzug_input = st.text_input("Abzüge (z.B. 52+30)", value=get_val('saw_deduct'), key="_saw_deduct", on_change=save_val, args=('saw_deduct',))
        
        abzuege = parse_abzuege(abzug_input)
        saege_erg = iso_mass - spalt - abzuege
        
        # Ergebnis-Ausgabe
        st.markdown(f"<div class='result-card-green'>Sägelänge: {round(saege_erg, 1)} mm</div>", unsafe_allow_html=True)
        
        # Gewichtsanzeige
        dn_idx = df_pipe[df_pipe['DN'] == selected_dn_global].index[0]
        std_ws = WS_STD_MAP.get(selected_dn_global, 4.0)
        c_zme = st.checkbox("ZME (Beton innen)?", value=get_val('saw_zme'), key="_saw_zme", on_change=save_val, args=('saw_zme',))
        
        kg = PhysicsEngine.calculate_pipe_weight(dn_idx, std_ws, saege_erg, c_zme)
        st.markdown(f"<div class='weight-box'>⚖️ Gewicht: ca. {kg} kg</div>", unsafe_allow_html=True)
        
        # Info Box (Keine Grafik)
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

    # 2.2 BOGEN ZUSCHNITT
    elif "Bogen" in tool_mode:
        st.subheader("Bogen Zuschnitt")
        angle = st.slider("Winkel (°)", 0, 90, 45, key="bogen_winkel")
        
        vorbau = round(standard_radius * math.tan(math.radians(angle/2)), 1)
        aussen = round((standard_radius + (row['D_Aussen']/2)) * angle * (math.pi/180), 1)
        innen = round((standard_radius - (row['D_Aussen']/2)) * angle * (math.pi/180), 1)
        
        st.markdown(f"<div class='result-card-green'>Vorbau: {vorbau} mm</div>", unsafe_allow_html=True)
        
        b1, b2 = st.columns(2)
        b1.metric("Rücken (Außen)", f"{aussen} mm")
        b2.metric("Bauch (Innen)", f"{innen} mm")

    # 2.3 STUTZEN SCHABLONE
    elif "Stutzen" in tool_mode:
        st.subheader("Stutzen Schablone")
        c_st1, c_st2 = st.columns(2)
        dn_stutzen = c_st1.selectbox("DN Stutzen", df_pipe['DN'], index=6, key="stutz_dn1")
        dn_haupt = c_st2.selectbox("DN Hauptrohr", df_pipe['DN'], index=9, key="stutz_dn2")
        
        if dn_stutzen > dn_haupt:
            st.error("Fehler: Stutzen kann nicht größer als Hauptrohr sein.")
        else:
            r_k = df_pipe[df_pipe['DN'] == dn_stutzen].iloc[0]['D_Aussen'] / 2
            r_g = df_pipe[df_pipe['DN'] == dn_haupt].iloc[0]['D_Aussen'] / 2
            
            c_tab, c_plot = st.columns([1, 2])
            
            # Daten berechnen
            angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]
            table_data = []
            for a in angles:
                t = int(round(r_g - math.sqrt(r_g**2 - (r_k * math.sin(math.radians(a)))**2), 0))
                u = int(round((r_k * 2 * math.pi) * (a/360), 0))
                table_data.append([f"{a}°", t, u])
            
            with c_tab:
                st.dataframe(pd.DataFrame(table_data, columns=["Winkel", "Tiefe", "Umfang"]), hide_index=True)
            with c_plot:
                st.pyplot(Visualizer.plot_stutzen_curve(r_g, r_k))

    # 2.4 ETAGEN (3D ENGINE)
    elif "Etage" in tool_mode:
        st.subheader("3D Etagen Berechnung & Visualisierung")
        et_type = st.radio("Typ", ["2D (Einfach)", "3D (Kastenmaß)", "3D (Fix-Winkel)"], horizontal=True, key="et_type")
        spalt_et = st.number_input("Spalt", 4, key="et_gap")
        
        c_calc, c_vis = st.columns([1, 1.5]) # Plot bekommt mehr Platz
        weight_l = 0.0
        
        # 3D Kamera Steuerung
        with c_vis:
            st.caption("📷 Kamera drehen (3D View)")
            v1, v2 = st.columns(2)
            azim = v1.slider("Horizontal", 0, 360, get_val('view_azim'), key="_view_azim", on_change=save_val, args=('view_azim',))
            elev = v2.slider("Vertikal", 0, 90, get_val('view_elev'), key="_view_elev", on_change=save_val, args=('view_elev',))

        # Berechnungslogik & Rendering
        if "2D" in et_type:
            with c_calc:
                h = st.number_input("Höhe H", 300, key="et2d_h")
                l = st.number_input("Länge L", 400, key="et2d_l")
                
                travel, angle = GeometryEngine.solve_offset_3d(h, l, 0)
                abzug = 2 * (standard_radius * math.tan(math.radians(angle/2)))
                erg = travel - abzug - spalt_et
                
                st.markdown(f"<div class='result-card-green'>Säge: {round(erg, 1)} mm</div>", unsafe_allow_html=True)
                weight_l = erg
            with c_vis:
                # 2D wird als flache 3D-Isometrie gerendert für Konsistenz
                st.pyplot(Visualizer.plot_true_3d_pipe(l, 0, h, azim, elev))
        
        elif "Kastenmaß" in et_type:
            with c_calc:
                b = st.number_input("Breite (Sprung)", 200, key="et3d_b")
                h = st.number_input("Höhe", 300, key="et3d_h")
                l = st.number_input("Länge", 400, key="et3d_l")
                
                travel, angle = GeometryEngine.solve_offset_3d(h, l, b)
                abzug = 2 * (standard_radius * math.tan(math.radians(angle/2)))
                erg = travel - abzug - spalt_et
                
                st.markdown(f"<div class='result-card-green'>Säge: {round(erg, 1)} mm</div>", unsafe_allow_html=True)
                weight_l = erg
            with c_vis:
                st.pyplot(Visualizer.plot_true_3d_pipe(l, b, h, azim, elev))
                
        elif "Fix-Winkel" in et_type:
            with c_calc:
                b = st.number_input("Breite", 200, key="etfix_b")
                h = st.number_input("Höhe", 300, key="etfix_h")
                fix_w = st.selectbox("Winkel", [15, 30, 45, 60, 90], index=2, key="etfix_w")
                
                # Rückwärtsrechnung: Benötigte Länge für fixen Winkel
                s_real = math.sqrt(b**2 + h**2)
                l_req = s_real / math.tan(math.radians(fix_w))
                
                travel = math.sqrt(l_req**2 + s_real**2)
                abzug = 2 * (standard_radius * math.tan(math.radians(fix_w/2)))
                erg = travel - abzug - spalt_et
                
                st.info(f"Benötigte Länge L: {round(l_req, 1)} mm")
                st.markdown(f"<div class='result-card-green'>Säge: {round(erg, 1)} mm</div>", unsafe_allow_html=True)
                weight_l = erg
            with c_vis:
                st.pyplot(Visualizer.plot_true_3d_pipe(l_req, b, h, azim, elev))
        
        # Gewichtsanzeige für das berechnete Passstück
        if weight_l > 0:
            dn_idx = df_pipe[df_pipe['DN'] == selected_dn_global].index[0]
            std_ws = WS_STD_MAP.get(selected_dn_global, 4.0)
            c_zme_et = st.checkbox("ZME?", key="et_zme")
            kg = PhysicsEngine.calculate_pipe_weight(dn_idx, std_ws, weight_l, c_zme_et)
            st.markdown(f"<div class='weight-box'>⚖️ Gewicht: ca. {kg} kg</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: ROHRBUCH
# -----------------------------------------------------------------------------
with tab_proj:
    st.subheader("Digitales Rohrbuch")
    with st.form("rb_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        iso = c1.text_input("ISO")
        naht = c2.text_input("Naht")
        datum = c3.date_input("Datum")
        
        c4, c5, c6 = st.columns(3)
        dn_sel = c4.selectbox("Dimension", df_pipe['DN'], index=8, key="rb_dn_sel")
        bauteil = c5.selectbox("Bauteil", ["📏 Rohr", "⤵️ Bogen", "⭕ Flansch", "🔗 Muffe", "🔩 Nippel", "🪵 T-Stück", "🔻 Reduzierung"])
        laenge = c6.number_input("Länge", value=0)
        
        c7, c8 = st.columns(2)
        charge = c7.text_input("Charge")
        schweisser = c8.text_input("Schweißer")
        
        if st.form_submit_button("Speichern"):
            DatabaseRepository.add_rohrbuch_entry((iso, naht, datum.strftime("%d.%m.%Y"), f"DN {dn_sel}", bauteil, laenge, charge, schweisser))
            st.success("Gespeichert!")
    
    # Anzeige und Löschen
    df_rb = DatabaseRepository.get_all("rohrbuch")
    st.dataframe(df_rb, use_container_width=True)
    
    with st.expander("Zeile löschen"):
        if not df_rb.empty:
            opts = {f"ID {r['id']}: {r['iso']} {r['naht']}": r['id'] for i, r in df_rb.iterrows()}
            if opts:
                sel = st.selectbox("Wähle Eintrag:", list(opts.keys()), key="rb_del_sel")
                if st.button("Löschen", key="rb_del_btn"):
                    DatabaseRepository.delete_by_id("rohrbuch", opts[sel])
                    st.rerun()

# -----------------------------------------------------------------------------
# TAB 4: KALKULATION
# -----------------------------------------------------------------------------
with tab_info:
    # EINSTELLUNGEN
    with st.expander("💶 Preis-Datenbank & Config"):
        c_io1, c_io2 = st.columns(2)
        # Export/Import Config
        try:
            json_data = json.dumps(st.session_state.store)
            c_io1.download_button("💾 Einstellungen speichern", data=json_data, file_name="pipecraft_config.json", mime="application/json")
        except: pass
        
        uploaded_file = c_io2.file_uploader("📂 Einstellungen laden", type=["json"])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                st.session_state.store.update(data)
                st.success("Konfiguration geladen!")
                st.rerun()
            except: st.error("Fehler beim Laden der Datei")
        
        st.divider()
        # Preise
        c_p1, c_p2, c_p3 = st.columns(3)
        st.session_state.store['p_lohn'] = c_p1.number_input("Lohn (€/h)", value=get_val('p_lohn'), key="_p_lohn", on_change=save_val, args=('p_lohn',))
        st.session_state.store['p_stahl'] = c_p2.number_input("Stahl-Scheibe (€)", value=get_val('p_stahl'), key="_p_stahl", on_change=save_val, args=('p_stahl',))
        st.session_state.store['p_dia'] = c_p3.number_input("Diamant-Scheibe (€)", value=get_val('p_dia'), key="_p_dia", on_change=save_val, args=('p_dia',))
        
        c_p4, c_p5, c_p6 = st.columns(3)
        st.session_state.store['p_cel'] = c_p4.number_input("Elektrode CEL (€)", value=get_val('p_cel'), key="_p_cel", on_change=save_val, args=('p_cel',))
        st.session_state.store['p_draht'] = c_p5.number_input("Draht (€/kg)", value=get_val('p_draht'), key="_p_draht", on_change=save_val, args=('p_draht',))
        st.session_state.store['p_gas'] = c_p6.number_input("Gas (€/L)", value=get_val('p_gas'), key="_p_gas", on_change=save_val, args=('p_gas',))
        
        c_p7, c_p8, c_p9 = st.columns(3)
        st.session_state.store['p_wks'] = c_p7.number_input("WKS (€)", value=get_val('p_wks'), key="_p_wks", on_change=save_val, args=('p_wks',))
        st.session_state.store['p_kebu1'] = c_p8.number_input("Kebu 1.2 (€)", value=get_val('p_kebu1'), key="_p_kebu1", on_change=save_val, args=('p_kebu1',))
        st.session_state.store['p_primer'] = c_p9.number_input("Primer (€/L)", value=get_val('p_primer'), key="_p_primer", on_change=save_val, args=('p_primer',))
        st.session_state.store['p_machine'] = c_p9.number_input("Geräte-Pauschale (€/h)", value=get_val('p_machine'), key="_p_machine", on_change=save_val, args=('p_machine',))

    # MODUS WAHL
    kalk_sub_mode = st.radio("Ansicht:", ["Eingabe & Rechner", "📊 Projekt Status / Export"], horizontal=True, label_visibility="collapsed")
    st.divider()

    if kalk_sub_mode == "Eingabe & Rechner":
        calc_task = st.radio("Tätigkeit", ["🔥 Fügen (Schweißen)", "✂️ Trennen (Vorbereitung)", "🔧 Montage (Armaturen)", "🛡️ Isolierung", "🚗 Regie"], horizontal=True, key="calc_mode")
        st.markdown("---")
        
        # Lokale Preise für einfache Nutzung
        p_lohn = get_val('p_lohn'); p_cel = get_val('p_cel'); p_draht = get_val('p_draht')
        p_gas = get_val('p_gas'); p_wks = get_val('p_wks'); p_kebu_in = get_val('p_kebu1')
        p_primer = get_val('p_primer'); p_stahl_disc = get_val('p_stahl'); p_dia_disc = get_val('p_dia')
        p_machine = get_val('p_machine')

        # 4.1 FÜGEN
        if "Fügen" in calc_task:
            c1, c2, c3 = st.columns(3)
            # DN mit Auto-Update
            k_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('kw_dn')), key="_kw_dn", on_change=update_kw_dn)
            k_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('kw_ws')), key="_kw_ws", on_change=save_val, args=('kw_ws',))
            k_verf = c3.selectbox("Verfahren", ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"], index=get_verf_index(get_val('kw_verf')), key="_kw_verf", on_change=save_val, args=('kw_verf',))
            
            c4, c5 = st.columns(2)
            # Hinweis bei Großrohr
            if get_val('kw_dn') >= 300:
                st.info("ℹ️ Großrohr (≥ DN 300): Team-Größe automatisch auf 2 gesetzt.")
            
            pers_count = c4.number_input("Anzahl Mitarbeiter", value=get_val('kw_pers'), min_value=1, key="_kw_pers", on_change=save_val, args=('kw_pers',))
            anz = c5.number_input("Anzahl Nähte", value=get_val('kw_anz'), min_value=1, key="_kw_anz", on_change=save_val, args=('kw_anz',))
            
            factor = st.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('kw_factor'), 0.1, key="_kw_factor", on_change=save_val, args=('kw_factor',))
            split_entry = st.checkbox("Als 2 Positionen speichern? (Vorb. + Fügen)", value=get_val('kw_split'), key="_kw_split", on_change=save_val, args=('kw_split',))
            
            # Kalkulation (CostEngine Nutzung)
            welding_minutes = CostEngine.estimate_welding_time(k_dn, k_ws, k_verf)
            fitting_minutes = (k_dn / 25.0) * 2.5 # 2.5 min pro Zoll Fittingzeit
            
            duration_per_seam = ((welding_minutes + fitting_minutes) / pers_count) * factor
            crew_hourly_rate = (pers_count * p_lohn) + (pers_count * p_machine)
            total_labor_cost = (duration_per_seam / 60 * crew_hourly_rate) * anz
            
            # Materialkosten
            da = df_pipe[df_pipe['DN'] == k_dn].iloc[0]['D_Aussen']
            kg = (da * math.pi * k_ws**2 * 0.7 / 1000 * 7.85 / 1000) * 1.5
            
            if "CEL" in k_verf:
                mat_cost = ((5.0 * kg) * 0.40) * anz
                mat_text = "CEL Elektroden"
            else:
                mat_cost = (kg * p_draht + (duration_per_seam/60 * 15 * p_gas)) * anz
                mat_text = f"{round(kg,1)} kg Draht"
            
            total_cost = total_labor_cost + mat_cost
            total_time = duration_per_seam * anz
            
            m1, m2 = st.columns(2)
            m1.metric("Zeit Total", f"{int(total_time)} min")
            m2.metric("Kosten Total", f"{round(total_cost, 2)} €")
            
            if st.button("Hinzufügen", key="add_komplett"):
                if split_entry:
                    t_half = total_time / 2
                    c_half_lab = (t_half / 60) * crew_hourly_rate
                    DatabaseRepository.add_kalkulation_entry(("Vorbereitung", f"DN {k_dn} Fitting", anz, t_half, c_half_lab, "-"))
                    DatabaseRepository.add_kalkulation_entry(("Fügen", f"DN {k_dn} Welding", anz, t_half, c_half_lab + mat_cost, mat_text))
                else:
                    DatabaseRepository.add_kalkulation_entry(("Fügen", f"DN {k_dn} {k_verf}", anz, total_time, total_cost, mat_text))
                st.success("Gespeichert!")
                st.rerun()

        # 4.2 TRENNEN
        elif "Trennen" in calc_task:
            c1, c2, c3, c4 = st.columns(4)
            c_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('cut_dn')), key="_cut_dn", on_change=save_val, args=('cut_dn',))
            c_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('cut_ws')), key="_cut_ws", on_change=save_val, args=('cut_ws',))
            disc = c3.selectbox("Scheibe", ["125 mm", "180 mm", "230 mm"], index=get_disc_idx(get_val('cut_disc')), key="_cut_disc", on_change=save_val, args=('cut_disc',))
            anz = c4.number_input("Anzahl", value=get_val('cut_anz'), min_value=1, key="_cut_anz", on_change=save_val, args=('cut_anz',))
            
            c5, c6 = st.columns(2)
            zma = c5.checkbox("Beton (ZMA)?", value=get_val('cut_zma'), key="_cut_zma", on_change=save_val, args=('cut_zma',))
            iso = c6.checkbox("Mantel entfernen?", value=get_val('cut_iso'), key="_cut_iso", on_change=save_val, args=('cut_iso',))
            factor = st.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('cut_factor'), 0.1, key="_cut_factor", on_change=save_val, args=('cut_factor',))
            
            zoll = c_dn / 25.0
            cap = 14000 if "230" in disc else (7000 if "180" in disc else 3500)
            
            # Faktoren
            zma_fac_d = 2.0 if zma else 1.0
            zma_fac_t = 3.0 if zma else 1.0
            iso_fac = 1.3 if iso else 1.0
            
            t_total = zoll * 0.5 * zma_fac_t * iso_fac * factor * anz
            area = (math.pi * df_pipe[df_pipe['DN']==c_dn].iloc[0]['D_Aussen']) * c_ws
            n_disc = math.ceil((area * zma_fac_d * anz) / cap)
            
            cost = ((t_total/60 * p_lohn) + (n_disc * (p_dia_disc if zma else p_stahl_disc)))
            
            m1, m2 = st.columns(2)
            m1.metric("Zeit", f"{int(t_total)} min")
            m2.metric("Kosten", f"{round(cost, 2)} €")
            
            if st.button("Hinzufügen", key="cut_add"):
                DatabaseRepository.add_kalkulation_entry(("Vorbereitung", f"DN {c_dn} ({disc})", anz, t_total, cost, f"{n_disc}x Scheiben"))
                st.rerun()

        # 4.3 MONTAGE
        elif "Montage" in calc_task:
            c1, c2, c3 = st.columns(3)
            m_type = c1.selectbox("Bauteil", ["Schieber", "Klappe", "Hydrant", "Formstück"], key="mon_type")
            m_dn = c2.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('mon_dn')), key="_mon_dn", on_change=save_val, args=('mon_dn',))
            m_anz = c3.number_input("Anzahl", value=get_val('mon_anz'), min_value=1, key="_mon_anz", on_change=save_val, args=('mon_anz',))
            
            factor = st.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('mon_factor'), 0.1, key="_mon_factor", on_change=save_val, args=('mon_factor',))
            
            row_mon = df_pipe[df_pipe['DN'] == m_dn].iloc[0]
            bolts = row_mon[f'Lochzahl{suffix}']
            
            time_per_piece = (bolts * 2.5) + 20 
            total_time = time_per_piece * m_anz * factor
            total_cost = (total_time / 60) * (p_lohn + p_machine)
            
            m1, m2 = st.columns(2)
            m1.metric("Zeit Total", f"{int(total_time)} min")
            m2.metric("Kosten Total", f"{round(total_cost, 2)} €")
            
            if st.button("Hinzufügen", key="mon_add"):
                DatabaseRepository.add_kalkulation_entry(("Montage", f"DN {m_dn} {m_type}", m_anz, total_time, total_cost, f"{bolts*2}x Schrauben"))
                st.rerun()

        # 4.4 ISOLIERUNG
        elif "Isolierung" in calc_task:
            sys_opts = ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"]
            sys = st.radio("System", sys_opts, horizontal=True, index=get_sys_idx(get_val('iso_sys')), key="_iso_sys", on_change=save_val, args=('iso_sys',))
            c1, c2, c3 = st.columns(3)
            i_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('iso_dn')), key="_iso_dn", on_change=save_val, args=('iso_dn',))
            i_anz = c2.number_input("Anzahl", value=get_val('iso_anz'), min_value=1, key="_iso_anz", on_change=save_val, args=('iso_anz',))
            factor = c3.slider("⏱️ Zeit-Faktor", 0.5, 2.0, get_val('iso_factor'), 0.1, key="_iso_factor", on_change=save_val, args=('iso_factor',))
            
            time = (20 + (i_dn * 0.07)) * factor * i_anz
            da = df_pipe[df_pipe['DN'] == i_dn].iloc[0]['D_Aussen']
            flaeche = (da * math.pi / 1000) * 0.5 * i_anz
            
            c_mat = 0; txt = ""
            if "WKS" in sys:
                c_mat = p_wks * i_anz; txt = f"{i_anz}x WKS"
            elif "B50" in sys:
                c_mat = (flaeche * 4.0 * 15.0) + (flaeche * 0.2 * 12.0); txt = "B50+Folie"
            else:
                c_mat = flaeche * 4.0 * 15.0; txt = "B80 Band"
            
            cost = ((time/60 * p_lohn) + c_mat)
            m1, m2 = st.columns(2)
            m1.metric("Zeit", f"{int(time)} min"); m2.metric("Kosten", f"{round(cost, 2)} €")
            
            if st.button("Hinzufügen", key="iso_add"):
                DatabaseRepository.add_kalkulation_entry(("Iso", f"DN {i_dn} {sys}", i_anz, time, cost, txt))
                st.rerun()

        # 4.5 REGIE
        elif "Regie" in calc_task:
            c1, c2 = st.columns(2)
            t = c1.number_input("Minuten", value=get_val('reg_min'), step=15, key="_reg_min", on_change=save_val, args=('reg_min',))
            p = c2.number_input("Personen", value=get_val('reg_pers'), min_value=1, key="_reg_pers", on_change=save_val, args=('reg_pers',))
            cost = (t/60 * p_lohn) * p
            st.metric("Kosten", f"{round(cost, 2)} €")
            if st.button("Hinzufügen", key="reg_add"):
                DatabaseRepository.add_kalkulation_entry(("Regie", f"{p} Pers.", 1, t, cost, "-"))
                st.rerun()

        # LIVE STATUS
        st.markdown("### 📊 Projekt Status (Live)")
        df_k = DatabaseRepository.get_all("kalkulation")
        if not df_k.empty:
            sc1, sc2 = st.columns(2)
            sc1.metric("Gesamt-Kosten", f"{round(df_k['kosten'].sum(), 2)} €")
            sc2.metric("Gesamt-Stunden", f"{round(df_k['zeit_min'].sum()/60, 1)} h")
            st.dataframe(df_k, use_container_width=True)
            
            c_del, c_rst = st.columns(2)
            with c_del.expander("Zeile löschen"):
                opts = {f"ID {r['id']}: {r['typ']}": r['id'] for i, r in df_k.iterrows()}
                if opts:
                    sel = st.selectbox("Wähle:", list(opts.keys()), key="kalk_del_sel_inline")
                    if st.button("Löschen", key="kalk_del_btn_inline"):
                        DatabaseRepository.delete_by_id("kalkulation", opts[sel])
                        st.rerun()
            if c_rst.button("Alles Löschen", type="primary", key="kalk_reset_inline"):
                DatabaseRepository.clear_table("kalkulation")
                st.rerun()
        else:
            st.info("Projekt ist leer.")

    # REPORTING MODE
    elif kalk_sub_mode == "📊 Projekt Status / Export":
        st.header("Projekt Übersicht & Export")
        df_k = DatabaseRepository.get_all("kalkulation")
        if not df_k.empty:
            sc1, sc2 = st.columns(2)
            sc1.metric("Gesamt-Kosten", f"{round(df_k['kosten'].sum(), 2)} €")
            sc2.metric("Gesamt-Stunden", f"{round(df_k['zeit_min'].sum()/60, 1)} h")
            st.dataframe(df_k, use_container_width=True)
            
            c_del, c_rst = st.columns(2)
            with c_del.expander("Zeile löschen"):
                opts = {f"ID {r['id']}: {r['typ']}": r['id'] for i, r in df_k.iterrows()}
                if opts:
                    sel = st.selectbox("Wähle:", list(opts.keys()), key="kalk_del_sel")
                    if st.button("Löschen", key="kalk_del_btn"):
                        DatabaseRepository.delete_by_id("kalkulation", opts[sel])
                        st.rerun()
            if c_rst.button("Alles Löschen", type="primary", key="kalk_reset"):
                DatabaseRepository.clear_table("kalkulation")
                st.rerun()
            
            st.markdown("---")
            c_xls, c_pdf = st.columns(2)
            
            # Excel Export
            xlsx_data = export_to_excel(df_k)
            c_xls.download_button(label="📥 Excel Exportieren", data=xlsx_data, file_name=f"PipeCraft_{datetime.now().date()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            # PDF Export
            if PDF_AVAILABLE:
                pdf_data = export_to_pdf(df_k)
                c_pdf.download_button(label="📄 PDF Exportieren", data=pdf_data, file_name=f"PipeCraft_{datetime.now().date()}.pdf", mime="application/pdf")
            else:
                c_pdf.warning("PDF-Export benötigt 'fpdf' (siehe requirements.txt)")
        else:
            st.info("Keine Daten vorhanden.")
