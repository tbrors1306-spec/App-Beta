"""
PipeCraft V46.1 (Clean Code Master Edition - Hotfix)
----------------------------------------------------
Fix: Missing 'datetime' import added.

FOKUS:
1. Technische Berechnungen (Sägelisten, Bogenabwicklung, Stutzen).
2. Technische Dokumentation (Digitales Rohrbuch).

ENTFERNT (auf Kundenwunsch):
- Kommerzielle Kalkulation (Preise/Zeiten).
- Lagerverwaltung / Bestandsführung.
- Gewichtsberechnungen.
- Etagen-Berechnungen (3D).

Author: Senior Lead Software Engineer
Date: 2023-12-21
"""

# -----------------------------------------------------------------------------
# 1. IMPORTS & SYSTEM CONFIGURATION
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import sqlite3
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any
# WICHTIG: Dieser Import hat gefehlt
from datetime import datetime

# Logging konfigurieren für Debugging-Zwecke
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipeCraft")

# Versuch, die PDF-Bibliothek zu laden (Optional)
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("FPDF Bibliothek nicht gefunden. PDF-Export ist deaktiviert.")

# Streamlit Seiten-Konfiguration (Muss der erste Streamlit-Befehl sein)
st.set_page_config(
    page_title="PipeCraft V46.1",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für ein professionelles Design
st.markdown("""
<style>
    /* Grundlegendes App-Design */
    .stApp { 
        background-color: #f8f9fa; 
        color: #0f172a; 
    }
    
    /* Typografie */
    h1, h2, h3 { 
        font-family: 'Segoe UI', sans-serif; 
        color: #1e293b !important; 
        font-weight: 700; 
    }
    
    /* Info-Karten (Blau) */
    .result-card-blue { 
        background-color: #eff6ff; 
        padding: 20px; 
        border-radius: 8px; 
        border-left: 5px solid #3b82f6; 
        margin-bottom: 10px; 
        color: #1e3a8a; 
        font-size: 1rem;
    }
    
    /* Ergebnis-Karten (Grün) */
    .result-card-green { 
        background: #f0fdf4; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 8px solid #22c55e; 
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
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        font-size: 0.95rem; 
        color: #334155; 
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Eingabefelder verschönern */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input { 
        border-radius: 4px; 
        border: 1px solid #cbd5e1; 
    }
    
    /* Buttons einheitlich gestalten */
    div.stButton > button { 
        width: 100%; 
        border-radius: 4px; 
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
# 2. DATA LAYER (NORMDATEN)
# -----------------------------------------------------------------------------

# Statische Datenbank für Rohrleitungs-Komponenten (DIN/EN/ASME Mix für Praxisnähe)
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

# Initialisierung des Pandas DataFrames mit Integritätscheck
try:
    df_pipe = pd.DataFrame(RAW_DATA)
except ValueError as e:
    st.error(f"FATALER FEHLER: Inkonsistente Daten-Arrays. {e}")
    st.stop()

# Schrauben-Details: [Schlüsselweite, Anzugsdrehmoment (ca.)]
SCHRAUBEN_DB = { 
    "M12": [18, 60], 
    "M16": [24, 130], 
    "M20": [30, 250], 
    "M24": [36, 420], 
    "M27": [41, 600], 
    "M30": [46, 830], 
    "M33": [50, 1100], 
    "M36": [55, 1400], 
    "M39": [60, 1800], 
    "M45": [70, 2700], 
    "M52": [80, 4200] 
}

DB_NAME = "pipecraft_v46.db"

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & SERVICES
# -----------------------------------------------------------------------------

def get_row_by_dn(dn: int) -> pd.Series:
    """
    Holt sicher eine Zeile aus der Datenbank für eine spezifische Nennweite.
    Wenn DN nicht gefunden wird, wird die erste Zeile als Fallback zurückgegeben.
    """
    try:
        return df_pipe[df_pipe['DN'] == dn].iloc[0]
    except IndexError:
        return df_pipe.iloc[0]

def get_schrauben_info(gewinde: str) -> List[Any]:
    """
    Gibt Details zur Schraube zurück (Schlüsselweite, Drehmoment).
    """
    return SCHRAUBEN_DB.get(gewinde, ["?", "?"])

# --- CLASS: SMART CUT LOGIC ---

@dataclass
class SelectedFitting:
    """
    Datenmodell für ein ausgewähltes Bauteil in der Sägeliste.
    """
    type_name: str
    count: int
    deduction_single: float
    dn_spec: int # Die Dimension, die für dieses Bauteil relevant ist

class FittingManager:
    """
    Zentrale Klasse für die Ermittlung von Abzugsmaßen (Z-Maßen).
    """
    
    @staticmethod
    def get_deduction(type_name: str, dn_target: int, pn_suffix: str = "_16", custom_angle: float = 45.0) -> float:
        """
        Ermittelt das Abzugsmaß.
        
        WICHTIG: 'dn_target' muss der Durchmesser sein, der die Baulänge bestimmt.
        Bei einer Reduzierung ist dies immer der GROSSE Durchmesser (DN Groß).
        """
        row_data = get_row_by_dn(dn_target)
        
        if type_name == "Bogen 90° (BA3)":
            return float(row_data['Radius_BA3'])
        
        elif type_name == "Bogen (Zuschnitt)":
            # Berechnung des Stichmaßes: Radius * tan(Winkel / 2)
            radius = float(row_data['Radius_BA3'])
            angle_rad = math.radians(custom_angle / 2)
            return radius * math.tan(angle_rad)
            
        elif type_name == "Flansch (Vorschweiß)":
            # Abzug ist die Blattstärke
            return float(row_data[f'Flansch_b{pn_suffix}'])
            
        elif type_name == "T-Stück":
            # Standard Bauhöhe H
            return float(row_data['T_Stueck_H'])
            
        elif "Reduzierung" in type_name:
            # Die Länge einer konzentrischen Reduzierung richtet sich nach dem großen DN
            return float(row_data['Red_Laenge_L'])
            
        return 0.0

# --- CLASS: VISUALIZATION SERVICE ---

class Visualizer:
    """
    Zuständig für die Erstellung von technischen Grafiken mit Matplotlib.
    """
    
    @staticmethod
    def plot_stutzen_curve(r_haupt: float, r_stutzen: float) -> plt.Figure:
        """
        Erstellt die Abwicklungskurve (Sinuskurve) für einen Stutzen.
        """
        # Wir berechnen Punkte in 5-Grad-Schritten für eine glatte Kurve
        angles = range(0, 361, 5)
        
        try:
            # Formel für die Verschneidung zweier Zylinder
            depths = [r_haupt - math.sqrt(r_haupt**2 - (r_stutzen * math.sin(math.radians(a)))**2) for a in angles]
        except ValueError:
            # Falls Stutzen > Hauptrohr (mathematisch nicht möglich ohne Durchdringung)
            return plt.figure()

        fig, ax = plt.subplots(figsize=(8, 1.5))
        
        # Plotten der Kurve
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        
        # Fläche füllen für bessere Optik
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        
        # Achsen formatieren
        ax.set_xlim(0, 360)
        ax.set_ylabel("Tiefe (mm)")
        ax.set_xlabel("Umfangswinkel (°)")
        
        # Layout straffen
        plt.tight_layout()
        return fig

# --- CLASS: DATABASE REPOSITORY ---

class DatabaseRepository:
    """
    Kapselt alle Datenbankzugriffe (SQLite).
    Verwendet Context Manager für sicheres Schließen der Verbindungen.
    """
    
    @staticmethod
    def init_tables():
        """Initialisiert die Datenbanktabelle für das Rohrbuch."""
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            # Nur noch die Rohrbuch-Tabelle, da Kalkulation/Lager entfernt wurden
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        iso TEXT, 
                        naht TEXT, 
                        datum TEXT, 
                        dimension TEXT, 
                        bauteil TEXT, 
                        laenge REAL, 
                        charge TEXT, 
                        schweisser TEXT)''')
            conn.commit()

    @staticmethod
    def add_rohrbuch_entry(data: Tuple):
        """Fügt einen neuen Eintrag ins Rohrbuch hinzu."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(
                'INSERT INTO rohrbuch (iso, naht, datum, dimension, bauteil, laenge, charge, schweisser) VALUES (?,?,?,?,?,?,?,?)', 
                data
            )
            conn.commit()

    @staticmethod
    def get_all_rohrbuch() -> pd.DataFrame:
        """Liest das gesamte Rohrbuch aus."""
        with sqlite3.connect(DB_NAME) as conn:
            return pd.read_sql_query("SELECT * FROM rohrbuch", conn)
    
    @staticmethod
    def delete_entry(entry_id: int):
        """Löscht einen Eintrag anhand der ID."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute("DELETE FROM rohrbuch WHERE id=?", (entry_id,))
            conn.commit()

# --- HELPER: EXPORT ---

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Erstellt eine Excel-Datei im Speicher."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Rohrbuch")
    return output.getvalue()

def export_to_pdf(df: pd.DataFrame) -> bytes:
    """Erstellt eine einfache PDF-Liste im Speicher."""
    if not PDF_AVAILABLE:
        return b""
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Titel
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Digitales Rohrbuch - Export", 0, 1, 'C')
    pdf.ln(5)
    
    # Datenzeilen
    pdf.set_font("Arial", size=10)
    for index, row in df.iterrows():
        try:
            # Zeileninhalt zusammenbauen
            line_text = f"ISO: {row.get('iso', '')} | Naht: {row.get('naht', '')} | {row.get('dimension', '')} | {row.get('bauteil', '')}"
            # Latin-1 Encoding erzwingen für FPDF Kompatibilität
            safe_text = line_text.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 8, safe_text, 1, 1)
        except Exception:
            continue
            
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 4. INITIALIZATION & SESSION STATE
# -----------------------------------------------------------------------------

# Datenbank initialisieren
DatabaseRepository.init_tables()

# Session State für Smart Cut Liste initialisieren
if 'fitting_list' not in st.session_state:
    st.session_state.fitting_list = []

# Session State für persistente Werte (Inputs)
if 'store' not in st.session_state:
    st.session_state.store = {
        'saw_mass': 1000.0, 
        'saw_gap': 4.0,
        'saw_dn_large': 100, # Default DN
        'bogen_winkel': 45
    }

# Hilfsfunktionen für State-Handling
def save_val(key):
    """Speichert den Wert eines Widgets in den permanenten Store."""
    st.session_state.store[key] = st.session_state[f"_{key}"]

def get_val(key):
    """Holt einen Wert aus dem Store."""
    return st.session_state.store.get(key)

# -----------------------------------------------------------------------------
# 5. UI IMPLEMENTATION (MAIN APP)
# -----------------------------------------------------------------------------

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=50) 
st.sidebar.markdown("### Einstellungen")

# Globale Dimension-Auswahl
selected_dn_global = st.sidebar.selectbox(
    "Nennweite (Global)", 
    df_pipe['DN'], 
    index=8, # Default auf DN 150 (Index 8)
    key="global_dn"
) 

selected_pn = st.sidebar.radio(
    "Druckstufe", 
    ["PN 16", "PN 10"], 
    index=0, 
    key="global_pn"
) 

# Kontext-Daten laden
row_data_global = get_row_by_dn(selected_dn_global)
standard_radius_global = float(row_data_global['Radius_BA3'])
suffix = "_16" if selected_pn == "PN 16" else "_10"

# --- HAUPTBEREICH ---
st.title("PipeCraft V46.1")
st.caption(f"🔧 Aktive Konfiguration: DN {selected_dn_global} | {selected_pn} | Standard-Radius: {standard_radius_global} mm")

# Tabs definieren
tab_buch, tab_werk, tab_proj = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch"])

# -----------------------------------------------------------------------------
# TAB 1: TABELLENBUCH (READ-ONLY DATEN)
# -----------------------------------------------------------------------------
with tab_buch:
    st.subheader("Technische Daten")
    
    # Zeile 1: Rohrdimensionen
    c_info1, c_info2 = st.columns(2)
    c_info1.markdown(f"<div class='result-card-blue'><b>Außen-Durchmesser:</b> {row_data_global['D_Aussen']} mm</div>", unsafe_allow_html=True)
    c_info1.markdown(f"<div class='result-card-blue'><b>Bauart 3 Radius (90°):</b> {standard_radius_global} mm</div>", unsafe_allow_html=True)
    
    c_info2.markdown(f"<div class='result-card-blue'><b>T-Stück Bauhöhe (H):</b> {row_data_global['T_Stueck_H']} mm</div>", unsafe_allow_html=True)
    c_info2.markdown(f"<div class='result-card-blue'><b>Reduzierung Länge (L):</b> {row_data_global['Red_Laenge_L']} mm</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Zeile 2: Flanschdaten
    st.subheader(f"Flanschdaten ({selected_pn})")
    
    schraube_dim = row_data_global[f'Schraube_M{suffix}']
    sw_info, nm_info = get_schrauben_info(schraube_dim)
    
    col_f1, col_f2 = st.columns(2)
    col_f1.markdown(f"<div class='result-card-blue'><b>Blattstärke:</b> {row_data_global[f'Flansch_b{suffix}']} mm</div>", unsafe_allow_html=True)
    col_f2.markdown(f"<div class='result-card-blue'><b>Schrauben:</b> {row_data_global[f'Lochzahl{suffix}']}x {schraube_dim} (SW {sw_info})</div>", unsafe_allow_html=True)
    
    # Zeile 3: Einbaulängen (Details)
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.markdown(f"<div class='detail-box'>Länge (Fest-Fest)<br><b>{row_data_global[f'L_Fest{suffix}']} mm</b></div>", unsafe_allow_html=True)
    col_d2.markdown(f"<div class='detail-box'>Länge (Fest-Los)<br><b>{row_data_global[f'L_Los{suffix}']} mm</b></div>", unsafe_allow_html=True)
    col_d3.markdown(f"<div class='detail-box'>Drehmoment (ca.)<br><b>{nm_info} Nm</b></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: WERKSTATT (BERECHNUNGEN)
# -----------------------------------------------------------------------------
with tab_werk:
    # Navigation innerhalb der Werkstatt
    tool_mode = st.radio(
        "Werkzeug wählen:", 
        ["📏 Säge (Smart Cut)", "🔄 Bogen (Zuschnitt)", "🔥 Stutzen (Schablone)"], 
        horizontal=True, 
        label_visibility="collapsed", 
        key="tool_nav"
    )
    st.divider()
    
    # --- SUB-TOOL 1: SMART CUT (SÄGE) ---
    if "Säge" in tool_mode:
        st.subheader("Smart Cut System (Passstück)")
        
        # Eingabe: Basisdaten
        col_base1, col_base2 = st.columns(2)
        iso_mass = col_base1.number_input("Gesamtmaß aus Isometrie (mm)", value=get_val('saw_mass'), step=10.0, key="_saw_mass", on_change=save_val, args=('saw_mass',))
        spalt = col_base2.number_input("Wurzelspalt pro Naht (mm)", value=get_val('saw_gap'), key="_saw_gap", on_change=save_val, args=('saw_gap',))
        
        st.markdown("#### 🧩 Bauteile hinzufügen")
        
        # Eingabe: Formteile (Der Konfigurator)
        c_fit1, c_fit2, c_fit3, c_fit4 = st.columns([2, 1.5, 1, 1])
        
        # 1. Typ wählen
        fitting_type = c_fit1.selectbox("Bauteil Typ", ["Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung (konz.)"])
        
        # 2. Dimension wählen (Smart Logic für Reduzierung)
        fitting_dn_large = selected_dn_global # Standardwert
        fitting_dn_small = None
        
        if "Reduzierung" in fitting_type:
            # Bei Reduzierung zeigen wir ZWEI Felder an
            c_red_lg, c_red_sm = c_fit2.columns(2)
            fitting_dn_large = c_red_lg.selectbox("DN (Groß)", df_pipe['DN'], index=df_pipe['DN'].tolist().index(selected_dn_global), key="sel_dn_lg")
            fitting_dn_small = c_red_sm.selectbox("DN (Klein)", df_pipe['DN'], index=0, key="sel_dn_sm")
        else:
            # Sonst nur eins
            fitting_dn_large = c_fit2.selectbox("Dimension (DN)", df_pipe['DN'], index=df_pipe['DN'].tolist().index(selected_dn_global), key="sel_dn_norm")

        # 3. Winkel (nur bei Bogen Zuschnitt)
        fitting_angle = 45.0
        if "Zuschnitt" in fitting_type:
            fitting_angle = c_fit3.number_input("Winkel °", value=45.0, step=1.0)
        else:
            c_fit3.write("-")
            
        # 4. Anzahl
        fitting_count = c_fit4.number_input("Anzahl", value=1, min_value=1)
        
        # Button: Hinzufügen
        if st.button("Hinzufügen (+)", type="secondary"):
            # Berechnung des Abzugs (basiert auf DN Groß bei Reduzierung!)
            deduction_value = FittingManager.get_deduction(fitting_type, fitting_dn_large, suffix, fitting_angle)
            
            # Name für die Anzeige generieren
            display_name = f"{fitting_type} (DN {fitting_dn_large})"
            if "Zuschnitt" in fitting_type:
                display_name += f" [{fitting_angle}°]"
            if "Reduzierung" in fitting_type and fitting_dn_small:
                display_name = f"Reduzierung {fitting_dn_large} x {fitting_dn_small}"
            
            # Zur Liste hinzufügen
            st.session_state.fitting_list.append(SelectedFitting(display_name, fitting_count, deduction_value, fitting_dn_large))
            st.rerun()
            
        # --- Listen-Anzeige & Berechnung ---
        total_deduction_sum = 0.0
        total_gaps_count = 0
        
        if st.session_state.fitting_list:
            st.markdown("---")
            st.markdown("##### Gewählte Bauteile:")
            
            for i, item in enumerate(st.session_state.fitting_list):
                row_c1, row_c2, row_c3 = st.columns([4, 2, 1])
                
                # Subtotal für diese Zeile
                sub_deduction = item.deduction_single * item.count
                total_deduction_sum += sub_deduction
                total_gaps_count += item.count
                
                row_c1.write(f"**{item.count}x** {item.type_name}")
                row_c2.caption(f"Abzug: -{round(sub_deduction, 1)} mm")
                
                if row_c3.button("🗑️", key=f"del_item_{i}"):
                    st.session_state.fitting_list.pop(i)
                    st.rerun()
            
            if st.button("Liste leeren", type="primary"):
                st.session_state.fitting_list = []
                st.rerun()
        
        # Finale Berechnung
        total_gap_deduction = total_gaps_count * spalt
        final_cut_length = iso_mass - total_deduction_sum - total_gap_deduction
        
        st.markdown("---")
        
        if final_cut_length < 0:
            st.error(f"Fehler: Die Summe der Abzüge ({total_deduction_sum + total_gap_deduction} mm) ist größer als das Isomaß!")
        else:
            res_c1, res_c2 = st.columns(2)
            res_c1.markdown(f"<div class='result-card-green'>Sägelänge: {round(final_cut_length, 1)} mm</div>", unsafe_allow_html=True)
            
            detail_text = f"Formteile: -{round(total_deduction_sum, 1)} mm | Spalte ({total_gaps_count}): -{round(total_gap_deduction, 1)} mm"
            res_c2.info(detail_text)

    # --- SUB-TOOL 2: BOGEN (ZUSCHNITT) ---
    elif "Bogen" in tool_mode:
        st.subheader("Bogen Zuschnitt (Detail)")
        
        angle_slider = st.slider("Winkel (°)", 0, 90, 45, key="bogen_winkel")
        
        # Radien-Berechnung
        r_mid = standard_radius_global
        da_val = row_data_global['D_Aussen']
        
        r_aussen = r_mid + (da_val / 2) # Rücken
        r_innen = r_mid - (da_val / 2)  # Bauch
        
        # Bogenlängen (Kreisbogenformel: 2*pi*r * alpha/360)
        # Vereinfacht: r * rad(alpha)
        len_mid = r_mid * math.radians(angle_slider)
        len_out = r_aussen * math.radians(angle_slider)
        len_in = r_innen * math.radians(angle_slider)
        
        # Vorbau (Stichmaß)
        vorbau_val = r_mid * math.tan(math.radians(angle_slider / 2))
        
        st.markdown(f"<div class='result-card-green'>Vorbau (Stichmaß): {round(vorbau_val, 1)} mm</div>", unsafe_allow_html=True)
        
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Rücken (Außen)", f"{round(len_out, 1)} mm")
        col_b2.metric("Mitte (Neutral)", f"{round(len_mid, 1)} mm")
        col_b3.metric("Bauch (Innen)", f"{round(len_in, 1)} mm")

    # --- SUB-TOOL 3: STUTZEN (SCHABLONE) ---
    elif "Stutzen" in tool_mode:
        st.subheader("Stutzen Schablone")
        
        c_st1, c_st2 = st.columns(2)
        dn_stutzen = c_st1.selectbox("DN Stutzen (Abzweig)", df_pipe['DN'], index=6)
        dn_haupt = c_st2.selectbox("DN Hauptrohr", df_pipe['DN'], index=9)
        
        if dn_stutzen > dn_haupt:
            st.error("Fehler: Stutzen darf nicht größer als das Hauptrohr sein!")
        else:
            # Radien holen
            r_k = df_pipe[df_pipe['DN'] == dn_stutzen].iloc[0]['D_Aussen'] / 2
            r_g = df_pipe[df_pipe['DN'] == dn_haupt].iloc[0]['D_Aussen'] / 2
            
            c_table, c_plot = st.columns([1, 2])
            
            # Tabelle generieren (0 bis 180 Grad)
            table_rows = []
            for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
                # Tiefe berechnen (Verschneidungsformel)
                t_val = int(round(r_g - math.sqrt(r_g**2 - (r_k * math.sin(math.radians(angle)))**2), 0))
                # Umfang berechnen (Abwicklungslänge am Stutzen)
                u_val = int(round((r_k * 2 * math.pi) * (angle / 360), 0))
                
                table_rows.append({
                    "Winkel": f"{angle}°",
                    "Tiefe (mm)": t_val,
                    "Umfang (mm)": u_val
                })
            
            with c_table:
                st.table(pd.DataFrame(table_rows))
            
            with c_plot:
                st.pyplot(Visualizer.plot_stutzen_curve(r_g, r_k))

# -----------------------------------------------------------------------------
# TAB 3: ROHRBUCH (DOKUMENTATION)
# -----------------------------------------------------------------------------
with tab_proj:
    st.subheader("Digitales Rohrbuch")
    
    # Eingabeformular
    with st.form("rohrbuch_form", clear_on_submit=False):
        c_rb1, c_rb2, c_rb3 = st.columns(3)
        iso_in = c_rb1.text_input("ISO Nummer")
        naht_in = c_rb2.text_input("Naht Nummer")
        datum_in = c_rb3.date_input("Datum")
        
        c_rb4, c_rb5 = st.columns(2)
        dn_in = c_rb4.selectbox("Dimension", df_pipe['DN'], index=8)
        len_in = c_rb5.number_input("Länge (mm)", value=0)
        
        # Detaillierte Bauteil-Auswahl für Dokumentation
        fitting_types_docu = [
            "Rohr", 
            "Bogen 90° (BA3)", 
            "Bogen (Zuschnitt)", 
            "Flansch (Vorschweiß)", 
            "T-Stück", 
            "Reduzierung (konz.)", 
            "Muffe", 
            "Nippel"
        ]
        bauteil_in = st.selectbox("Bauteil / Formteil", fitting_types_docu)
        
        c_rb6, c_rb7 = st.columns(2)
        charge_in = c_rb6.text_input("Charge / Materialnummer")
        schweisser_in = c_rb7.text_input("Schweißer Kürzel")
        
        if st.form_submit_button("Naht Speichern"):
            DatabaseRepository.add_rohrbuch_entry(
                (iso_in, naht_in, datum_in.strftime("%d.%m.%Y"), f"DN {dn_in}", bauteil_in, len_in, charge_in, schweisser_in)
            )
            st.success("Eintrag gespeichert!")
            st.rerun()
            
    # Tabelle anzeigen
    df_rohrbuch = DatabaseRepository.get_all_rohrbuch()
    st.markdown("### Aktuelle Einträge")
    st.dataframe(df_rohrbuch, use_container_width=True)
    
    # Export Funktionen
    if not df_rohrbuch.empty:
        col_export1, col_export2 = st.columns(2)
        
        # Excel Export
        col_export1.download_button(
            label="📥 Als Excel herunterladen",
            data=export_to_excel(df_rohrbuch),
            file_name=f"Rohrbuch_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # PDF Export
        if PDF_AVAILABLE:
            col_export2.download_button(
                label="📄 Als PDF herunterladen",
                data=export_to_pdf(df_rohrbuch),
                file_name=f"Rohrbuch_{datetime.now().strftime('%Y-%m-%d')}.pdf",
                mime="application/pdf"
            )
            
    # Lösch-Funktion
    with st.expander("Einträge verwalten / löschen"):
        if not df_rohrbuch.empty:
            # Mapping erstellen: Anzeige-String -> ID
            delete_options = {f"ID {row['id']}: ISO {row['iso']} - Naht {row['naht']}": row['id'] for index, row in df_rohrbuch.iterrows()}
            
            selected_delete = st.selectbox("Eintrag zum Löschen wählen:", list(delete_options.keys()))
            
            if st.button("Ausgewählten Eintrag löschen"):
                id_to_delete = delete_options[selected_delete]
                DatabaseRepository.delete_entry(id_to_delete)
                st.rerun()
