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
    """
    Lädt die statischen Rohdaten. Caching verbessert die Performance.
    """
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
    """Verwaltet Datenbankoperationen (SQLite)."""
    
    @staticmethod
    def init_db():
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        iso TEXT, naht TEXT, datum TEXT, 
                        dimension TEXT, bauteil TEXT, laenge REAL, 
                        charge TEXT, schweisser TEXT)''')
            conn.commit()

    @staticmethod
    def add_entry(data: Tuple):
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO rohrbuch (iso, naht, datum, dimension, bauteil, laenge, charge, schweisser) VALUES (?,?,?,?,?,?,?,?)', data)
            conn.commit()

    @staticmethod
    def get_all() -> pd.DataFrame:
        with sqlite3.connect(DB_NAME) as conn:
            return pd.read_sql_query("SELECT * FROM rohrbuch ORDER BY id DESC", conn)

    @staticmethod
    def delete_entry(entry_id: int):
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute("DELETE FROM rohrbuch WHERE id=?", (entry_id,))
            conn.commit()

# -----------------------------------------------------------------------------
# 3. LOGIK-SCHICHT (CALCULATION LOGIC & MODELS)
# -----------------------------------------------------------------------------

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
        """
        Berechnet das Abzugsmaß (Z-Maß) für ein Bauteil.
        """
        row = self.get_row_by_dn(dn_large)
        
        if "Bogen 90°" in fitting_type:
            return float(row['Radius_BA3'])
        
        elif "Bogen (Zuschnitt)" in fitting_type:
            # Formel: Radius * tan(Winkel / 2)
            radius = float(row['Radius_BA3'])
            return radius * math.tan(math.radians(angle / 2))
        
        elif "Flansch" in fitting_type:
            col_name = f'Flansch_b{pn_suffix}'
            return float(row[col_name])
            
        elif "T-Stück" in fitting_type:
            return float(row['T_Stueck_H'])
            
        elif "Reduzierung" in fitting_type:
            # Bei Reduzierung gilt Länge L basierend auf DN groß
            return float(row['Red_Laenge_L'])
            
        return 0.0

    def calculate_stutzen_coords(self, dn_haupt: int, dn_stutzen: int) -> Tuple[pd.DataFrame, plt.Figure]:
        """Berechnet Koordinaten und Plot für Stutzen-Ausschneidung."""
        r_main = self.get_row_by_dn(dn_haupt)['D_Aussen'] / 2
        r_stub = self.get_row_by_dn(dn_stutzen)['D_Aussen'] / 2

        if r_stub > r_main:
            raise ValueError("Stutzenradius darf nicht größer als Hauptrohr sein.")

        angles = range(0, 361, 5) # Plotting resolution
        try:
            # Verschneidungsformel Zylinder/Zylinder
            depths = [r_main - math.sqrt(r_main**2 - (r_stub * math.sin(math.radians(a)))**2) for a in angles]
        except ValueError:
             raise ValueError("Mathematischer Fehler bei der Kurvenberechnung.")

        # DataFrame für Tabelle (grobere Schritte)
        table_data = []
        for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
            t_val = r_main - math.sqrt(r_main**2 - (r_stub * math.sin(math.radians(angle)))**2)
            u_val = (r_stub * 2 * math.pi) * (angle / 360)
            table_data.append({
                "Winkel": f"{angle}°",
                "Tiefe (mm)": round(t_val, 1),
                "Umfang (mm)": round(u_val, 1)
            })

        # Plot erstellen
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360)
        ax.set_ylabel("Ausschnitt-Tiefe (mm)")
        ax.set_xlabel("Umfangswinkel (°)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        return pd.DataFrame(table_data), fig

# -----------------------------------------------------------------------------
# 4. UI-KOMPONENTEN (VIEWS)
# -----------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> Tuple[int, str]:
    with st.sidebar:
        st.title("⚙️ Einstellungen")
        
        # Container für bessere Gruppierung
        with st.container():
            st.markdown("### Globale Parameter")
            selected_dn = st.selectbox(
                "Nennweite (DN)", 
                df['DN'], 
                index=8, # DN 150 Default
                key="global_dn"
            )
            selected_pn = st.radio(
                "Druckstufe", 
                ["PN 16", "PN 10"], 
                horizontal=True,
                key="global_pn"
            )
        
        st.info("Diese Einstellungen beeinflussen alle Tabs (Tabellenbuch, Sägeliste etc.).")
        st.divider()
        st.caption(f"Rohrbau Profi v8.0\n© 2025 PipeCraft Solutions")
        
        return selected_dn, selected_pn

def render_tab_handbook(calc: PipeCalculator, dn: int, pn: str):
    row = calc.get_row_by_dn(dn)
    suffix = "_16" if pn == "PN 16" else "_10"
    
    st.markdown(f"## Tabellenbuch: DN {dn} / {pn}")
    
    # 3 Spalten Layout für kompakte Info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📐 Rohr & Bogen")
        st.metric("Außen-Ø", f"{row['D_Aussen']} mm")
        st.metric("Radius (BA3)", f"{row['Radius_BA3']} mm", help="Standard Rohrbogen Bauart 3")
        st.metric("Wandstärke (Ref)", "Norm-abhängig")

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
        st.write(f"**Schraubenlänge (F-F):** {row[f'L_Fest{suffix}']} mm")
        st.write(f"**Schraubenlänge (F-L):** {row[f'L_Los{suffix}']} mm")

def render_tab_workshop(calc: PipeCalculator, df: pd.DataFrame, current_dn: int, pn: str):
    """
    Rendert den Werkstatt-Tab mit der neuen V2.0 Sägeliste.
    """
    st.markdown("## 📐 Werkstatt & Sägeliste V2.0")
    
    # Initialisiere Session State für die persistente Liste, falls noch nicht vorhanden
    if 'cut_list_storage' not in st.session_state:
        st.session_state.cut_list_storage = []
    if 'next_id' not in st.session_state:
        st.session_state.next_id = 1

    tab_saw, tab_bend, tab_branch = st.tabs(["🪚 Smart Sägeliste", "🔄 Bogen Details", "🔥 Stutzen"])
    
    # --- SUB-TAB: SMART SÄGELISTE (NEU V2.0) ---
    with tab_saw:
        # Split-View: Links Kalkulation, Rechts Liste
        col_calc, col_storage = st.columns([1.2, 1.8])
        
        with col_calc:
            st.markdown("### 1. Kalkulation (Aktuelles Spool)")
            
            # A) Metadaten (Kontext)
            with st.container(): 
                c_meta1, c_meta2 = st.columns(2)
                iso_ref = c_meta1.text_input("ISO / Projekt", key="meta_iso", placeholder="z.B. L-1004")
                spool_nr = c_meta2.text_input("Spool Nr.", key="meta_spool", placeholder="01")

            st.divider()

            # B) Maßeingabe
            iso_mass = st.number_input("Gesamtmaß (Isometrie) [mm]", min_value=0.0, step=10.0, format="%.1f", key="input_iso_mass")
            
            # C) Abzüge definieren (Dichtungen & Formteile)
            with st.expander("🛠️ Bauteile & Abzüge", expanded=True):
                # 1. Dichtungen & Spalte (NEU)
                st.caption("Dichtungen & Spalte")
                c_gap1, c_gap2, c_gap3 = st.columns(3)
                gap_weld = c_gap1.number_input("Wurzelspalt (mm)", value=3.0, step=0.5)
                gasket_count = c_gap2.number_input("Anz. Dichtungen", value=0, min_value=0, max_value=2, help="0=Keine, 1=Einseitig, 2=Beidseitig")
                gasket_thk = c_gap3.number_input("Dicke (mm)", value=2.0, min_value=0.0, step=0.5, disabled=(gasket_count==0))
                
                # 2. Formteile hinzufügen
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
                    if st.form_submit_button("Bauteil hinzufügen ➕"):
                        deduction = calc.get_deduction(f_type, f_dn, "_16" if pn == "PN 16" else "_10", f_angle)
                        item_name = f"{f_type} DN{f_dn}"
                        if "Zuschnitt" in f_type: item_name += f" ({f_angle}°)"
                        st.session_state.fitting_list.append(FittingItem(item_name, f_count, deduction, f_dn))
                        st.rerun()

            # D) Live-Berechnung des aktuellen Stücks
            # Summe Formteile
            sum_fittings = sum([i.total_deduction for i in st.session_state.fitting_list])
            count_fittings = sum([i.count for i in st.session_state.fitting_list])
            
            # Summe Spalte (Annahme: 1 Naht pro Bauteil + 1)
            # Hier vereinfacht: User gibt Spalt vor, wir multiplizieren mit Anzahl Bauteile. 
            # In der Praxis oft: (Anzahl Bauteile) * Spalt.
            sum_welds = count_fittings * gap_weld
            
            # Summe Dichtungen
            sum_gaskets = gasket_count * gasket_thk
            
            total_deduction = sum_fittings + sum_welds + sum_gaskets
            final_cut = iso_mass - total_deduction

            # Anzeige Ergebnis
            if final_cut < 0:
                st.error(f"Negativmaß! Abzüge ({total_deduction}) > Iso ({iso_mass})")
            else:
                st.markdown(f"<div class='success-box' style='text-align:center'>Sägelänge<br><span style='font-size:1.8em'>{final_cut:.1f} mm</span></div>", unsafe_allow_html=True)
                
                # Detail-Info string bauen
                details_str = f"Teile: -{sum_fittings:.1f} | Spalte: -{sum_welds:.1f} | Dichtungen: -{sum_gaskets:.1f}"
                st.caption(details_str)
                
                # E) Buttons: Reset & Speichern
                btn_col1, btn_col2 = st.columns(2)
                
                if btn_col1.button("Reset Bauteile 🗑️"):
                    st.session_state.fitting_list = []
                    st.rerun()
                
                # Zur Liste hinzufügen
                if btn_col2.button("💾 In Liste speichern", type="primary"):
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
                        st.warning("Bitte Isomaß eingeben.")

        with col_storage:
            st.markdown("### 📋 Gespeicherte Schnitte")
            
            if not st.session_state.cut_list_storage:
                st.info("Noch keine Schnitte gespeichert.")
            else:
                # Liste als DataFrame aufbereiten
                data = [{
                    "ID": e.id,
                    "ISO": e.iso_ref,
                    "Spool": e.spool_nr,
                    "Iso-Maß": e.raw_length,
                    "Säge-Maß": e.cut_length,
                    "Info": e.details
                } for e in st.session_state.cut_list_storage]
                
                df_cuts = pd.DataFrame(data)
                
                # Tabelle anzeigen
                st.dataframe(
                    df_cuts, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Säge-Maß": st.column_config.NumberColumn(
                            "Säge-Maß (mm)",
                            format="%.1f mm",
                        )
                    }
                )
                
                # Aktionen für die Liste
                act_c1, act_c2 = st.columns(2)
                
                if act_c1.button("Letzten Eintrag löschen"):
                    st.session_state.cut_list_storage.pop()
                    st.rerun()
                    
                if act_c2.button("Liste komplett leeren"):
                    st.session_state.cut_list_storage = []
                    st.session_state.next_id = 1
                    st.rerun()

                # CSV Download
                csv = df_cuts.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                st.download_button(
                    label="📥 Liste Exportieren (CSV)",
                    data=csv,
                    file_name=f"Saegeliste_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    type="primary"
                )

    # --- SUB-TAB: BOGEN ---
    with tab_bend:
        c_b1, c_b2 = st.columns(2)
        angle = c_b1.slider("Winkel", 0, 90, 45)
        
        row = calc.get_row_by_dn(current_dn)
        radius = float(row['Radius_BA3'])
        da = float(row['D_Aussen'])
        
        # Berechnung
        vorbau = radius * math.tan(math.radians(angle / 2))
        len_mid = radius * math.radians(angle)
        len_out = (radius + da/2) * math.radians(angle)
        len_in = (radius - da/2) * math.radians(angle)
        
        c_b2.markdown(f"<div class='info-box'>Vorbau (Z-Maß): <b>{vorbau:.1f} mm</b></div>", unsafe_allow_html=True)
        
        met1, met2, met3 = st.columns(3)
        met1.metric("Bogenlänge Außen", f"{len_out:.1f} mm")
        met2.metric("Bogenlänge Mitte", f"{len_mid:.1f} mm")
        met3.metric("Bogenlänge Innen", f"{len_in:.1f} mm")

    # --- SUB-TAB: STUTZEN ---
    with tab_branch:
        c_s1, c_s2 = st.columns(2)
        dn_stub = c_s1.selectbox("DN Stutzen", df['DN'], index=5)
        dn_main = c_s2.selectbox("DN Hauptrohr", df['DN'], index=8)
        
        if c_s1.button("Berechnen 📐"):
            try:
                df_res, fig = calc.calculate_stutzen_coords(dn_main, dn_stub)
                st.pyplot(fig)
                with st.expander("Schablonen-Daten anzeigen"):
                    st.table(df_res)
            except ValueError as e:
                st.error(f"Fehler: {e}")

def render_tab_logbook(df_pipe: pd.DataFrame):
    st.markdown("## Digitales Rohrbuch")
    
    with st.expander("📝 Neuen Eintrag erfassen", expanded=True):
        with st.form("rohrbuch_entry"):
            c1, c2, c3 = st.columns(3)
            iso = c1.text_input("ISO-Nr.")
            naht = c2.text_input("Naht-Nr.")
            datum = c3.date_input("Datum")
            
            c4, c5, c6 = st.columns(3)
            dim = c4.selectbox("Dimension", df_pipe['DN'], key="rb_dn")
            bauteil = c5.selectbox("Bauteil", ["Naht", "Bogen", "Flansch", "Passstück"])
            laenge = c6.number_input("Länge (mm)", value=0.0)
            
            c7, c8 = st.columns(2)
            charge = c7.text_input("Charge")
            schweisser = c8.text_input("Schweißer ID")
            
            if st.form_submit_button("Speichern 💾"):
                DatabaseRepository.add_entry((iso, naht, datum.strftime("%Y-%m-%d"), f"DN {dim}", bauteil, laenge, charge, schweisser))
                st.success("Gespeichert!")
                st.rerun()

    st.markdown("### Historie")
    df_log = DatabaseRepository.get_all()
    st.dataframe(df_log, use_container_width=True, hide_index=True)
    
    if not df_log.empty:
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "rohrbuch.csv", "text/csv")

# -----------------------------------------------------------------------------
# 5. MAIN APP EXECUTION
# -----------------------------------------------------------------------------

def main():
    # Init
    DatabaseRepository.init_db()
    df_pipe = get_pipe_data()
    calc = PipeCalculator(df_pipe)
    
    # Session State initialisieren
    if 'fitting_list' not in st.session_state:
        st.session_state.fitting_list = []
    
    # Sidebar
    dn_sel, pn_sel = render_sidebar(df_pipe)

    # Main Content
    tab1, tab2, tab3 = st.tabs(["📘 Tabellenbuch", "🛠️ Werkstatt & Zuschnitt", "📋 Rohrbuch"])
    
    with tab1:
        render_tab_handbook(calc, dn_sel, pn_sel)
        
    with tab2:
        render_tab_workshop(calc, df_pipe, dn_sel, pn_sel)
        
    with tab3:
        render_tab_logbook(df_pipe)

if __name__ == "__main__":
    main()
