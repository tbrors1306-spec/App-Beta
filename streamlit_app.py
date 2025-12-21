import streamlit as st
import pandas as pd
import math
import sqlite3
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime
import matplotlib.pyplot as plt

# Versuch, FPDF für PDF-Export zu importieren
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. KONFIGURATION & LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipeCraft_Pro")

st.set_page_config(
    page_title="Rohrbau Profi 8.0",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für professionelles UI
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e293b; font-family: 'Segoe UI', sans-serif; }
    
    /* Metrik-Karten Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Ergebnis Box (Grün) */
    .success-box {
        padding: 20px;
        background-color: #dcfce7;
        color: #166534;
        border-radius: 8px;
        border-left: 5px solid #22c55e;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    
    /* Info Box (Blau) */
    .info-box {
        padding: 15px;
        background-color: #eff6ff;
        color: #1e40af;
        border-radius: 6px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
    }
    
    /* Tabellen Header etwas schicker */
    thead tr th {
        background-color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATEN-SCHICHT (DATA LAYER)
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

# Schrauben Referenzdaten
SCHRAUBEN_DB = { 
    "M12": {"sw": 18, "nm": 60}, "M16": {"sw": 24, "nm": 130}, "M20": {"sw": 30, "nm": 250},
    "M24": {"sw": 36, "nm": 420}, "M27": {"sw": 41, "nm": 600}, "M30": {"sw": 46, "nm": 830},
    "M33": {"sw": 50, "nm": 1100}, "M36": {"sw": 55, "nm": 1400}, "M39": {"sw": 60, "nm": 1800},
    "M45": {"sw": 70, "nm": 2700}, "M52": {"sw": 80, "nm": 4200} 
}

DB_NAME = "rohrbau_profi.db"

class DatabaseRepository:
    """Verwaltet Datenbankoperationen (SQLite) mit Migrations-Logik."""
    
    @staticmethod
    def init_db():
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            # Basistabelle erstellen
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        iso TEXT, 
                        naht TEXT, 
                        datum TEXT, 
                        dimension TEXT, 
                        bauteil TEXT, 
                        laenge REAL, 
                        charge TEXT, 
                        charge_apz TEXT,
                        schweisser TEXT)''')
            
            # MIGRATION CHECK: Prüfen, ob die Spalte 'charge_apz' existiert (für Upgrades von V1.0)
            c.execute("PRAGMA table_info(rohrbuch)")
            columns = [info[1] for info in c.fetchall()]
            
            if 'charge_apz' not in columns:
                try:
                    c.execute("ALTER TABLE rohrbuch ADD COLUMN charge_apz TEXT")
                    logging.info("Datenbank migriert: Spalte 'charge_apz' hinzugefügt.")
                except Exception as e:
                    logging.error(f"Fehler bei Migration: {e}")
            
            conn.commit()

    @staticmethod
    def add_entry(data: dict):
        """Nutzt nun ein Dictionary für sicherere Zuordnung."""
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
            # Füge eine Spalte 'Löschen' für den Editor hinzu (Standard False)
            df = pd.read_sql_query("SELECT * FROM rohrbuch ORDER BY id DESC", conn)
            # Falls DataFrame leer ist, müssen wir sicherstellen, dass 'Löschen' existiert
            if not df.empty:
                df['Löschen'] = False 
            else:
                # Leeres DataFrame mit korrekten Spalten erzeugen
                cols = ["id", "iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser", "Löschen"]
                df = pd.DataFrame(columns=cols)
            return df

    @staticmethod
    def delete_entries(ids: List[int]):
        """Löscht mehrere Einträge auf einmal."""
        if not ids: return
        with sqlite3.connect(DB_NAME) as conn:
            # SQL IN Clause sicher bauen
            placeholders = ', '.join('?' for _ in ids)
            sql = f"DELETE FROM rohrbuch WHERE id IN ({placeholders})"
            conn.cursor().execute(sql, ids)
            conn.commit()

# -----------------------------------------------------------------------------
# 3. HELPER & LOGIK-SCHICHT
# -----------------------------------------------------------------------------

class Exporter:
    """Zuständig für Excel und PDF Downloads."""
    
    @staticmethod
    def to_excel(df: pd.DataFrame) -> bytes:
        output = BytesIO()
        # Spalte 'Löschen' und 'id' für den Export entfernen
        export_df = df.drop(columns=['Löschen', 'id'], errors='ignore')
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Rohrbuch')
        return output.getvalue()

    @staticmethod
    def to_pdf(df: pd.DataFrame) -> bytes:
        if not PDF_AVAILABLE:
            return b"Fehler: FPDF Bibliothek nicht installiert."
            
        try:
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            
            # Titel
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"Rohrbuch Export - {datetime.now().strftime('%d.%m.%Y')}", 0, 1, 'C')
            pdf.ln(5)
            
            # Header
            pdf.set_font("Arial", 'B', 8)
            cols = ["ISO", "Naht", "Datum", "DN", "Bauteil", "Charge", "APZ", "Schweißer"]
            widths = [35, 20, 25, 20, 40, 35, 35, 30]
            
            for i, col in enumerate(cols):
                pdf.cell(widths[i], 8, col, 1, 0, 'C')
            pdf.ln()
            
            # Data
            pdf.set_font("Arial", size=8)
            export_df = df.drop(columns=['Löschen', 'id'], errors='ignore')
            
            for _, row in export_df.iterrows():
                try:
                    # Daten holen und sicherstellen dass es Strings sind
                    v_iso = str(row.get('iso', ''))
                    v_naht = str(row.get('naht', ''))
                    v_dat = str(row.get('datum', ''))
                    v_dim = str(row.get('dimension', ''))
                    v_bau = str(row.get('bauteil', ''))[:25]
                    v_cha = str(row.get('charge', ''))
                    v_apz = str(row.get('charge_apz', ''))
                    v_schw = str(row.get('schweisser', ''))

                    # Zeile schreiben
                    pdf.cell(widths[0], 8, v_iso.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.cell(widths[1], 8, v_naht.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.cell(widths[2], 8, v_dat.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.cell(widths[3], 8, v_dim.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.cell(widths[4], 8, v_bau.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.cell(widths[5], 8, v_cha.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.cell(widths[6], 8, v_apz.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.cell(widths[7], 8, v_schw.encode('latin-1', 'replace').decode('latin-1'), 1)
                    pdf.ln()
                except Exception:
                    continue 
                    
            return pdf.output(dest='S').encode('latin-1', 'replace')
            
        except Exception as e:
            return str(e).encode()

@dataclass
class FittingItem:
    """Datenmodell für ein Bauteil in der aktuellen Kalkulation."""
    name: str
    count: int
    deduction_single: float
    dn: int
    
    @property
    def total_deduction(self) -> float:
        return self.deduction_single * self.count

@dataclass
class CutListEntry:
    """Datenmodell für einen gespeicherten Schnitt in der Liste."""
    id: int
    iso_ref: str        # Referenz zur Zeichnung
    spool_nr: str       # Nummer des Teilstücks
    raw_length: float   # Isomaß
    cut_length: float   # Fertiges Sägelänge
    details: str        # Info über Abzüge (z.B. "2x Bogen, 1x Dichtung")
    timestamp: str

class PipeCalculator:
    """Enthält die reine Berechnungslogik, getrennt von der UI."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_row_by_dn(self, dn: int) -> pd.Series:
        """Findet die Datenzeile für eine Nennweite."""
        row = self.df[self.df['DN'] == dn]
        if row.empty:
            return self.df.iloc[0] # Fallback
        return row.iloc[0]

    def get_deduction(self, fitting_type: str, dn_large: int, pn_suffix: str = "_16", angle: float = 90.0) -> float:
        row = self.get_row_by_dn(dn_large)
        if "Bogen 90°" in fitting_type:
            return float(row['Radius_BA3'])
        elif "Bogen (Zuschnitt)" in fitting_type:
            radius = float(row['Radius_BA3'])
            return radius * math.tan(math.radians(angle / 2))
        elif "Flansch" in fitting_type:
            col_name = f'Flansch_b{pn_suffix}'
            return float(row[col_name])
        elif "T-Stück" in fitting_type:
            return float(row['T_Stueck_H'])
        elif "Reduzierung" in fitting_type:
            return float(row['Red_Laenge_L'])
        return 0.0

    def calculate_stutzen_coords(self, dn_haupt: int, dn_stutzen: int) -> Tuple[pd.DataFrame, plt.Figure]:
        r_main = self.get_row_by_dn(dn_haupt)['D_Aussen'] / 2
        r_stub = self.get_row_by_dn(dn_stutzen)['D_Aussen'] / 2

        if r_stub > r_main:
            raise ValueError("Stutzenradius darf nicht größer als Hauptrohr sein.")

        angles = range(0, 361, 5)
        try:
            depths = [r_main - math.sqrt(r_main**2 - (r_stub * math.sin(math.radians(a)))**2) for a in angles]
        except ValueError:
             raise ValueError("Mathematischer Fehler bei der Kurvenberechnung.")

        table_data = []
        for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
            t_val = r_main - math.sqrt(r_main**2 - (r_stub * math.sin(math.radians(angle)))**2)
            u_val = (r_stub * 2 * math.pi) * (angle / 360)
            table_data.append({
                "Winkel": f"{angle}°",
                "Tiefe (mm)": round(t_val, 1),
                "Umfang (mm)": round(u_val, 1)
            })

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360)
        ax.set_ylabel("Tiefe (mm)")
        ax.set_xlabel("Winkel (°)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        return pd.DataFrame(table_data), fig

# -----------------------------------------------------------------------------
# 4. UI-KOMPONENTEN (VIEWS)
# -----------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> Tuple[int, str]:
    with st.sidebar:
        st.title("⚙️ Einstellungen")
        with st.container():
            st.markdown("### Globale Parameter")
            selected_dn = st.selectbox("Nennweite (DN)", df['DN'], index=8, key="global_dn")
            selected_pn = st.radio("Druckstufe", ["PN 16", "PN 10"], horizontal=True, key="global_pn")
        st.info("Diese Einstellungen beeinflussen alle Tabs.")
        st.divider()
        st.caption(f"Rohrbau Profi v8.0\n© 2025 PipeCraft Solutions")
        return selected_dn, selected_pn

def render_tab_handbook(calc: PipeCalculator, dn: int, pn: str):
    row = calc.get_row_by_dn(dn)
    suffix = "_16" if pn == "PN 16" else "_10"
    st.markdown(f"## Tabellenbuch: DN {dn} / {pn}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📐 Rohr & Bogen")
        st.metric("Außen-Ø", f"{row['D_Aussen']} mm")
        st.metric("Radius (BA3)", f"{row['Radius_BA3']} mm")
    with col2:
        st.markdown(f"### 🔩 Flansch ({pn})")
        st.metric("Blattstärke", f"{row[f'Flansch_b{suffix}']} mm")
        st.metric("Lochkreis", f"{row[f'LK_k{suffix}']} mm")
        schraube = row[f'Schraube_M{suffix}']
        info = SCHRAUBEN_DB.get(schraube, {'sw': '?', 'nm': '?'})
        st.info(f"**{row[f'Lochzahl{suffix}']}x {schraube}**\n\nSW: {info['sw']} mm | {info['nm']} Nm")
    with col3:
        st.markdown("### 🧩 Einbaulängen")
        st.write(f"**T-Stück (H):** {row['T_Stueck_H']} mm")
        st.write(f"**Reduzierung (L):** {row['Red_Laenge_L']} mm")
        st.write(f"**Schrauben (F-F):** {row[f'L_Fest{suffix}']} mm")
        st.write(f"**Schrauben (F-L):** {row[f'L_Los{suffix}']} mm")

def render_tab_workshop(calc: PipeCalculator, df: pd.DataFrame, current_dn: int, pn: str):
    st.markdown("## 📐 Werkstatt & Sägeliste V2.0")
    if 'cut_list_storage' not in st.session_state:
        st.session_state.cut_list_storage = []
    if 'next_id' not in st.session_state:
        st.session_state.next_id = 1

    tab_saw, tab_bend, tab_branch = st.tabs(["🪚 Smart Sägeliste", "🔄 Bogen Details", "🔥 Stutzen"])
    
    with tab_saw:
        col_calc, col_storage = st.columns([1.2, 1.8])
        
        with col_calc:
            st.markdown("### 1. Kalkulation (Spool)")
            with st.container(): 
                c_meta1, c_meta2 = st.columns(2)
                iso_ref = c_meta1.text_input("ISO / Projekt", key="meta_iso", placeholder="z.B. L-1004")
                spool_nr = c_meta2.text_input("Spool Nr.", key="meta_spool", placeholder="01")
            st.divider()
            iso_mass = st.number_input("Isomaß [mm]", min_value=0.0, step=10.0, format="%.1f", key="input_iso_mass")
            
            with st.expander("🛠️ Bauteile & Abzüge", expanded=True):
                st.caption("Dichtungen & Spalte")
                c_gap1, c_gap2, c_gap3 = st.columns(3)
                gap_weld = c_gap1.number_input("Wurzelspalt (mm)", value=3.0, step=0.5)
                gasket_count = c_gap2.number_input("Anz. Dicht.", value=0, min_value=0, max_value=2)
                gasket_thk = c_gap3.number_input("Dicke (mm)", value=2.0, min_value=0.0, disabled=(gasket_count==0))
                
                st.caption("Formteile hinzufügen")
                with st.form("add_fitting_v2", clear_on_submit=True):
                    f_type = st.selectbox("Typ", ["Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung (konz.)"])
                    c_f1, c_f2 = st.columns(2)
                    f_dn = c_f1.selectbox("DN", df['DN'], index=df['DN'].tolist().index(current_dn))
                    f_angle = 90.0
                    if "Zuschnitt" in f_type:
                        f_angle = c_f2.number_input("Winkel", value=45.0)
                    else:
                        c_f2.text("Standard")
                    f_count = st.number_input("Anzahl", 1, 10, 1)
                    if st.form_submit_button("Hinzufügen ➕"):
                        deduction = calc.get_deduction(f_type, f_dn, "_16" if pn == "PN 16" else "_10", f_angle)
                        item_name = f"{f_type} DN{f_dn}"
                        if "Zuschnitt" in f_type: item_name += f" ({f_angle}°)"
                        st.session_state.fitting_list.append(FittingItem(item_name, f_count, deduction, f_dn))
                        st.rerun()

            sum_fittings = sum([i.total_deduction for i in st.session_state.fitting_list])
            count_fittings = sum([i.count for i in st.session_state.fitting_list])
            sum_welds = count_fittings * gap_weld
            sum_gaskets = gasket_count * gasket_thk
            total_deduction = sum_fittings + sum_welds + sum_gaskets
            final_cut = iso_mass - total_deduction

            if final_cut < 0:
                st.error(f"Negativmaß! Abzüge ({total_deduction}) > Iso ({iso_mass})")
            else:
                st.markdown(f"<div class='success-box' style='text-align:center'>Sägelänge<br><span style='font-size:1.8em'>{final_cut:.1f} mm</span></div>", unsafe_allow_html=True)
                st.caption(f"Teile: -{sum_fittings:.1f} | Spalte: -{sum_welds:.1f} | Dicht.: -{sum_gaskets:.1f}")
                
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button("Reset Bauteile 🗑️"):
                    st.session_state.fitting_list = []
                    st.rerun()
                if btn_col2.button("💾 Speichern", type="primary"):
                    if iso_mass > 0:
                        new_entry = CutListEntry(
                            id=st.session_state.next_id,
                            iso_ref=iso_ref if iso_ref else "-",
                            spool_nr=spool_nr if spool_nr else f"#{st.session_state.next_id}",
                            raw_length=iso_mass,
                            cut_length=final_cut,
                            details=f"{count_fittings} Teile, {gasket_count} Dicht.",
                            timestamp=datetime.now().strftime("%H:%M")
                        )
                        st.session_state.cut_list_storage.append(new_entry)
                        st.session_state.next_id += 1
                        st.success("Gespeichert!")
                        st.rerun()
                    else:
                        st.warning("Isomaß fehlt.")

        with col_storage:
            st.markdown("### 📋 Gespeicherte Schnitte")
            if not st.session_state.cut_list_storage:
                st.info("Noch keine Schnitte gespeichert.")
            else:
                data = [{"ID": e.id, "ISO": e.iso_ref, "Spool": e.spool_nr, "Iso-Maß": e.raw_length, "Säge-Maß": e.cut_length, "Info": e.details} for e in st.session_state.cut_list_storage]
                df_cuts = pd.DataFrame(data)
                st.dataframe(df_cuts, use_container_width=True, hide_index=True, column_config={"Säge-Maß": st.column_config.NumberColumn("Säge-Maß (mm)", format="%.1f mm")})
                
                act_c1, act_c2 = st.columns(2)
                if act_c1.button("Letzten löschen"):
                    st.session_state.cut_list_storage.pop()
                    st.rerun()
                if act_c2.button("Liste leeren"):
                    st.session_state.cut_list_storage = []
                    st.session_state.next_id = 1
                    st.rerun()
                
                csv = df_cuts.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                st.download_button("📥 Liste Exportieren (CSV)", csv, f"Saegeliste_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    with tab_bend:
        c_b1, c_b2 = st.columns(2)
        angle = c_b1.slider("Winkel", 0, 90, 45)
        row = calc.get_row_by_dn(current_dn)
        radius = float(row['Radius_BA3'])
        da = float(row['D_Aussen'])
        vorbau = radius * math.tan(math.radians(angle / 2))
        len_mid = radius * math.radians(angle)
        len_out = (radius + da/2) * math.radians(angle)
        len_in = (radius - da/2) * math.radians(angle)
        c_b2.markdown(f"<div class='info-box'>Vorbau (Z-Maß): <b>{vorbau:.1f} mm</b></div>", unsafe_allow_html=True)
        met1, met2, met3 = st.columns(3)
        met1.metric("Bogenlänge Außen", f"{len_out:.1f} mm")
        met2.metric("Bogenlänge Mitte", f"{len_mid:.1f} mm")
        met3.metric("Bogenlänge Innen", f"{len_in:.1f} mm")

    with tab_branch:
        c_s1, c_s2 = st.columns(2)
        dn_stub = c_s1.selectbox("DN Stutzen", df['DN'], index=5)
        dn_main = c_s2.selectbox("DN Hauptrohr", df['DN'], index=8)
        if c_s1.button("Berechnen 📐"):
            try:
                df_res, fig = calc.calculate_stutzen_coords(dn_main, dn_stub)
                st.pyplot(fig)
                with st.expander("Schablonen-Daten"):
                    st.table(df_res)
            except ValueError as e:
                st.error(f"Fehler: {e}")

def render_tab_logbook(df_pipe: pd.DataFrame):
    st.markdown("## 📋 Digitales Rohrbuch V2.0")
    
    with st.expander("📝 Neuen Eintrag erfassen", expanded=True):
        with st.form("rohrbuch_entry_v2", clear_on_submit=False):
            st.caption("Naht-Daten")
            r1c1, r1c2, r1c3 = st.columns(3)
            iso = r1c1.text_input("ISO / Zeichnungs-Nr.", key="rb_iso")
            naht = r1c2.text_input("Naht-Nr.", key="rb_naht")
            datum = r1c3.date_input("Schweißdatum")
            
            st.caption("Material & Bauteil")
            r2c1, r2c2, r2c3 = st.columns(3)
            bauteil_liste = ["Rohr (Längsnaht)", "Rohrstoß", "Bogen 90°", "Bogen 45°", "T-Stück", "Reduzierung", "Flansch (Vorschweiß)", "Flansch (Blind)", "Muffe", "Nippel", "Olet/Stutzen"]
            bauteil = r2c1.selectbox("Bauteil / Komponente", bauteil_liste)
            dim = r2c2.selectbox("Dimension", df_pipe['DN'], index=8)
            laenge = r2c3.number_input("Länge/Menge (falls relevant)", value=0.0, step=10.0)
            
            st.caption("Qualitätssicherung")
            r3c1, r3c2, r3c3 = st.columns(3)
            charge = r3c1.text_input("Charge (Material)")
            apz = r3c2.text_input("Charge APZ / Zeugnis")
            schweisser = r3c3.text_input("Schweißer Stempel")
            
            if st.form_submit_button("Naht Speichern 💾", type="primary"):
                if iso and naht and schweisser:
                    data_dict = {
                        "iso": iso, "naht": naht, "datum": datum.strftime("%d.%m.%Y"),
                        "dimension": f"DN {dim}", "bauteil": bauteil, "laenge": laenge,
                        "charge": charge, "charge_apz": apz, "schweisser": schweisser
                    }
                    DatabaseRepository.add_entry(data_dict)
                    st.success(f"Naht {naht} (ISO {iso}) gespeichert.")
                    st.rerun()
                else:
                    st.warning("Bitte mindestens ISO, Naht-Nr. und Schweißer angeben.")

    st.divider()
    df_log = DatabaseRepository.get_all()
    
    if df_log.empty:
        st.info("Das Rohrbuch ist leer.")
    else:
        col_ex1, col_ex2, col_spacer = st.columns([1, 1, 4])
        file_suffix = datetime.now().strftime('%Y%m%d')
        col_ex1.download_button("📥 Excel Export", data=Exporter.to_excel(df_log), file_name=f"Rohrbuch_{file_suffix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        if PDF_AVAILABLE:
            col_ex2.download_button("📄 PDF Export", data=Exporter.to_pdf(df_log), file_name=f"Rohrbuch_{file_suffix}.pdf", mime="application/pdf")
        else:
            col_ex2.warning("PDF-Export inaktiv (FPDF fehlt)")

        st.markdown("### Aktuelle Einträge verwalten")
        st.caption("Nutze die Checkbox 'Löschen', um Einträge zu markieren.")
        
        edited_df = st.data_editor(
            df_log,
            column_config={
                "Löschen": st.column_config.CheckboxColumn("Auswahl", help="Zum Löschen markieren", default=False),
                "id": None, 
            },
            disabled=["iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser"],
            hide_index=True,
            use_container_width=True,
            key="editor_logbook"
        )
        
        rows_to_delete = edited_df[edited_df['Löschen'] == True]
        if not rows_to_delete.empty:
            count = len(rows_to_delete)
            st.warning(f"{count} Eintrag/Einträge zum Löschen markiert.")
            col_del1, col_del2 = st.columns([1, 4])
            if col_del1.button(f"🗑️ {count} löschen", type="primary"):
                ids_list = rows_to_delete['id'].tolist()
                DatabaseRepository.delete_entries(ids_list)
                st.success("Gelöscht!")
                st.rerun()

# -----------------------------------------------------------------------------
# 5. MAIN APP EXECUTION
# -----------------------------------------------------------------------------

def main():
    DatabaseRepository.init_db()
    df_pipe = get_pipe_data()
    calc = PipeCalculator(df_pipe)
    
    if 'fitting_list' not in st.session_state:
        st.session_state.fitting_list = []
    
    dn_sel, pn_sel = render_sidebar(df_pipe)

    tab1, tab2, tab3 = st.tabs(["📘 Tabellenbuch", "🛠️ Werkstatt & Zuschnitt", "📋 Rohrbuch"])
    
    with tab1:
        render_tab_handbook(calc, dn_sel, pn_sel)
    with tab2:
        render_tab_workshop(calc, df_pipe, dn_sel, pn_sel)
    with tab3:
        render_tab_logbook(df_pipe)

if __name__ == "__main__":
    main()
