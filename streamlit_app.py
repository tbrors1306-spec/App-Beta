import streamlit as st
import pandas as pd
import math
import sqlite3
import logging
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import List, Tuple
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
logger = logging.getLogger("PipeCraft_Pro_V4_1")

st.set_page_config(
    page_title="Rohrbau Profi 4.1",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modernes CSS für bessere Lesbarkeit
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e293b; font-family: 'Segoe UI', sans-serif; }
    
    /* Ergebnis Box (Grün) */
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
    
    /* Fehler Box (Rot) */
    .error-box {
        padding: 20px;
        background-color: #fee2e2;
        color: #991b1b;
        border-radius: 8px;
        border-left: 5px solid #ef4444;
        margin: 10px 0;
        text-align: center;
    }

    /* Container Styling */
    .calculation-area {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
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

SCHRAUBEN_DB = { 
    "M12": {"sw": 18, "nm": 60}, "M16": {"sw": 24, "nm": 130}, "M20": {"sw": 30, "nm": 250},
    "M24": {"sw": 36, "nm": 420}, "M27": {"sw": 41, "nm": 600}, "M30": {"sw": 46, "nm": 830},
    "M33": {"sw": 50, "nm": 1100}, "M36": {"sw": 55, "nm": 1400}, "M39": {"sw": 60, "nm": 1800},
    "M45": {"sw": 70, "nm": 2700}, "M52": {"sw": 80, "nm": 4200} 
}

DB_NAME = "rohrbau_profi.db"

class DatabaseRepository:
    """Verwaltet Datenbankoperationen."""
    
    @staticmethod
    def init_db():
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            # Rohrbuch Tabelle (Doku)
            c.execute('''CREATE TABLE IF NOT EXISTS rohrbuch (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        iso TEXT, naht TEXT, datum TEXT, 
                        dimension TEXT, bauteil TEXT, laenge REAL, 
                        charge TEXT, charge_apz TEXT, schweisser TEXT)''')
            
            # Migration APZ Spalte
            c.execute("PRAGMA table_info(rohrbuch)")
            cols = [info[1] for info in c.fetchall()]
            if 'charge_apz' not in cols:
                try:
                    c.execute("ALTER TABLE rohrbuch ADD COLUMN charge_apz TEXT")
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
    """Ein Bauteil in der AKTUELLEN Berechnung."""
    id: str # Unique ID für Löschung im UI
    name: str
    count: int
    deduction_single: float
    dn: int
    
    @property
    def total_deduction(self) -> float:
        return self.deduction_single * self.count

@dataclass
class SavedCut:
    """Ein fertig berechneter und gespeicherter Schnitt."""
    id: int
    raw_length: float
    cut_length: float
    details: str
    timestamp: str

class PipeCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_row(self, dn: int) -> pd.Series:
        row = self.df[self.df['DN'] == dn]
        return row.iloc[0] if not row.empty else self.df.iloc[0]

    def get_deduction(self, f_type: str, dn: int, pn: str, angle: float = 90.0) -> float:
        row = self.get_row(dn)
        suffix = "_16" if pn == "PN 16" else "_10"
        
        if "Bogen 90°" in f_type: return float(row['Radius_BA3'])
        if "Zuschnitt" in f_type: return float(row['Radius_BA3']) * math.tan(math.radians(angle / 2))
        if "Flansch" in f_type: return float(row[f'Flansch_b{suffix}'])
        if "T-Stück" in f_type: return float(row['T_Stueck_H'])
        if "Reduzierung" in f_type: return float(row['Red_Laenge_L'])
        return 0.0

class Exporter:
    @staticmethod
    def to_excel(df):
        output = BytesIO()
        export_df = df.drop(columns=['Löschen', 'id'], errors='ignore')
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Rohrbuch')
        return output.getvalue()

    @staticmethod
    def to_pdf(df):
        if not PDF_AVAILABLE: return b""
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Rohrbuch - {datetime.now().strftime('%d.%m.%Y')}", 0, 1, 'C')
        pdf.ln(5)
        pdf.set_font("Arial", size=8)
        
        cols = ["ISO", "Naht", "Datum", "DN", "Bauteil", "Charge", "APZ", "Schweißer"]
        for c in cols: pdf.cell(30, 8, c, 1)
        pdf.ln()
        
        export_df = df.drop(columns=['Löschen', 'id'], errors='ignore')
        for _, row in export_df.iterrows():
            line = [str(row.get(c.lower(), ''))[:18] for c in cols] # Mapping vereinfacht
            # Spezifisches Mapping für keys die nicht exakt matchen (iso vs ISO)
            line[0] = str(row.get('iso',''))
            line[1] = str(row.get('naht',''))
            line[6] = str(row.get('charge_apz',''))
            
            for item in line:
                try: pdf.cell(30, 8, item.encode('latin-1','replace').decode('latin-1'), 1)
                except: pdf.cell(30, 8, "?", 1)
            pdf.ln()
        return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 4. UI SEITEN
# -----------------------------------------------------------------------------

def render_tab_workshop(calc: PipeCalculator, df: pd.DataFrame, current_dn: int, pn: str):
    st.subheader("🪚 Smart Säge V4.1")
    
    # --- FEHLERBEHEBUNG FÜR SESSION STATE ---
    # Falls alte Objekte ohne 'id' im Speicher liegen, Liste bereinigen um Absturz zu verhindern.
    if 'fitting_list' in st.session_state and st.session_state.fitting_list:
        try:
            # Test: Hat das erste Objekt eine ID?
            _ = st.session_state.fitting_list[0].id
        except AttributeError:
            st.session_state.fitting_list = []
            st.warning("Versions-Update: Alte Berechnungsdaten wurden zurückgesetzt.")
            st.rerun()

    # State Init
    if 'fitting_list' not in st.session_state: st.session_state.fitting_list = []
    if 'saved_cuts' not in st.session_state: st.session_state.saved_cuts = []
    if 'next_cut_id' not in st.session_state: st.session_state.next_cut_id = 1

    # LAYOUT: Links Berechnung, Rechts Gespeicherte Liste
    col_calc, col_saved = st.columns([1.5, 1.5])

    # --- LINKE SPALTE: BERECHNUNG ---
    with col_calc:
        with st.container(border=True):
            st.markdown("#### 1. Zuschnitt berechnen")
            
            # A) Basisdaten
            raw_len = st.number_input("Rohrlänge (Schnittmaß aus Plan) [mm]", min_value=0.0, step=10.0, format="%.1f")
            
            col_g1, col_g2, col_g3 = st.columns(3)
            gap = col_g1.number_input("Wurzelspalt (mm)", value=3.0, step=0.5)
            dicht_anz = col_g2.number_input("Anz. Dichtungen", 0, 5, 0)
            dicht_thk = col_g3.number_input("Dicke (mm)", 0.0, 5.0, 2.0, disabled=(dicht_anz==0))
            
            st.divider()
            
            # B) Bauteile Hinzufügen
            st.caption("Bauteil abziehen:")
            c_add1, c_add2, c_add3, c_add4 = st.columns([2, 1.5, 1, 1])
            f_type = c_add1.selectbox("Typ", ["Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung"], label_visibility="collapsed")
            f_dn = c_add2.selectbox("DN", df['DN'], index=df['DN'].tolist().index(current_dn), label_visibility="collapsed")
            f_cnt = c_add3.number_input("Anz.", 1, 10, 1, label_visibility="collapsed")
            
            # Zuschnittswinkel nur anzeigen wenn nötig
            f_ang = 90.0
            if "Zuschnitt" in f_type:
                f_ang = st.slider("Winkel (°)", 0, 90, 45)

            if c_add4.button("➕", type="primary", help="Zur aktuellen Berechnung hinzufügen"):
                deduction = calc.get_deduction(f_type, f_dn, pn, f_ang)
                name = f"{f_type} DN{f_dn}" + (f" ({f_ang}°)" if "Zuschnitt" in f_type else "")
                # Unique ID generieren basierend auf Zeit, damit man einzelne Items löschen kann
                unique_id = f"{len(st.session_state.fitting_list)}_{datetime.now().timestamp()}"
                st.session_state.fitting_list.append(FittingItem(unique_id, name, f_cnt, deduction, f_dn))
                st.rerun()

            # C) LIVE LISTE (Warenkorb)
            if st.session_state.fitting_list:
                st.markdown("###### Aktuelle Abzüge:")
                
                for idx, item in enumerate(st.session_state.fitting_list):
                    c_row1, c_row2, c_row3 = st.columns([3, 1.5, 0.5])
                    c_row1.text(f"{item.count}x {item.name}")
                    c_row2.text(f"-{item.total_deduction:.1f} mm")
                    # Hier war der Fehler: Zugriff auf item.id, wenn item noch vom alten Typ war
                    if c_row3.button("🗑️", key=f"del_{item.id}"):
                        st.session_state.fitting_list.pop(idx)
                        st.rerun()
                
                if st.button("Alle Abzüge leeren", type="secondary"):
                    st.session_state.fitting_list = []
                    st.rerun()

            # D) ERGEBNIS RECHNUNG
            sum_fittings = sum(i.total_deduction for i in st.session_state.fitting_list)
            # Spalte: Wir nehmen an 1 Spalt pro Bauteil + 1 (oder einfach Manuell) -> Hier vereinfacht:
            # Anzahl Bauteile = Anzahl Spalte (Näherung). Wenn Fittingliste leer, aber Gap > 0, ziehen wir nix ab
            count_fittings = sum(i.count for i in st.session_state.fitting_list)
            sum_gaps = count_fittings * gap 
            sum_dicht = dicht_anz * dicht_thk
            
            total_deduct = sum_fittings + sum_gaps + sum_dicht
            final_cut = raw_len - total_deduct
            
            st.divider()
            
            if final_cut < 0:
                st.markdown(f"<div class='error-box'>ACHTUNG: Negativmaß!<br>Abzug ({total_deduct:.1f}) > Rohr ({raw_len:.1f})</div>", unsafe_allow_html=True)
                st.warning("Bitte Liste der Abzüge prüfen und korrigieren.")
            else:
                st.markdown(f"<div class='success-box'>Sägelänge<br><span style='font-size:2.5em; font-weight:bold'>{final_cut:.1f} mm</span></div>", unsafe_allow_html=True)
                st.caption(f"Details: Bauteile -{sum_fittings:.1f} | Spalte -{sum_gaps:.1f} | Dichtungen -{sum_dicht:.1f}")
                
                # E) Speichern
                if st.button("💾 Schnitt in Liste speichern", type="primary", use_container_width=True):
                    if raw_len > 0:
                        detail_txt = f"{count_fittings} Teile, {dicht_anz} Dicht."
                        new_cut = SavedCut(
                            st.session_state.next_cut_id, raw_len, final_cut, detail_txt, datetime.now().strftime("%H:%M")
                        )
                        st.session_state.saved_cuts.append(new_cut)
                        st.session_state.next_cut_id += 1
                        # Reset Calculation Area
                        st.session_state.fitting_list = [] 
                        st.rerun()

    # --- RECHTE SPALTE: GESPEICHERTE LISTE ---
    with col_saved:
        st.markdown("#### 📋 Fertige Schnittliste")
        if not st.session_state.saved_cuts:
            st.info("Noch keine Schnitte gespeichert.")
        else:
            # Dataframe für Anzeige
            data = [asdict(c) for c in st.session_state.saved_cuts]
            df_show = pd.DataFrame(data).drop(columns=['id'])
            df_show.rename(columns={'raw_length': 'Rohr', 'cut_length': 'Schnitt', 'details': 'Info', 'timestamp': 'Zeit'}, inplace=True)
            
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            
            col_act1, col_act2 = st.columns(2)
            if col_act1.button("Letzten löschen"):
                st.session_state.saved_cuts.pop()
                st.rerun()
            if col_act2.button("Liste leeren"):
                st.session_state.saved_cuts = []
                st.rerun()
                
            # Export CSV
            csv = df_show.to_csv(sep=";", decimal=",", index=False).encode('utf-8')
            st.download_button("📥 Liste als CSV", csv, "saegeliste.csv", "text/csv", use_container_width=True)

def render_tab_handbook(calc, dn, pn):
    row = calc.get_row(dn)
    suffix = "_16" if pn == "PN 16" else "_10"
    st.subheader(f"Technische Daten DN {dn} ({pn})")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Außen-Ø", f"{row['D_Aussen']} mm")
    c1.metric("Radius (BA3)", f"{row['Radius_BA3']} mm")
    
    c2.metric("Blattstärke", f"{row[f'Flansch_b{suffix}']} mm")
    c2.metric("Lochkreis", f"{row[f'LK_k{suffix}']} mm")
    
    c3.info(f"Schrauben: {row[f'Lochzahl{suffix}']}x {row[f'Schraube_M{suffix}']}")
    c3.text(f"Länge (F-F): {row[f'L_Fest{suffix}']} mm")

def render_tab_logbook(df_pipe):
    st.subheader("Digitales Rohrbuch")
    
    with st.expander("Eintrag hinzufügen", expanded=True):
        with st.form("new_entry", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            iso = c1.text_input("ISO")
            naht = c2.text_input("Naht")
            dat = c3.date_input("Datum")
            
            c4, c5, c6 = st.columns(3)
            bt = c4.selectbox("Bauteil", ["Naht", "Bogen", "Flansch", "Passstück"])
            dn = c5.selectbox("DN", df_pipe['DN'], index=8)
            ch = c6.text_input("Charge")
            
            c7, c8 = st.columns(2)
            apz = c7.text_input("APZ")
            sch = c8.text_input("Schweißer")
            
            if st.form_submit_button("Speichern", type="primary"):
                DatabaseRepository.add_entry({
                    "iso": iso, "naht": naht, "datum": dat.strftime("%d.%m.%Y"),
                    "dimension": f"DN {dn}", "bauteil": bt, "laenge": 0,
                    "charge": ch, "charge_apz": apz, "schweisser": sch
                })
                st.success("Gespeichert")
                st.rerun()

    df = DatabaseRepository.get_all()
    if not df.empty:
        edited = st.data_editor(df, key="editor", hide_index=True, use_container_width=True, 
                                column_config={"Löschen": st.column_config.CheckboxColumn(default=False)})
        
        to_del = edited[edited['Löschen'] == True]
        if not to_del.empty:
            if st.button(f"{len(to_del)} Einträge löschen"):
                DatabaseRepository.delete_entries(to_del['id'].tolist())
                st.rerun()
                
        # Exports
        c_ex1, c_ex2 = st.columns(2)
        c_ex1.download_button("Excel Export", Exporter.to_excel(df), "rohrbuch.xlsx")
        if PDF_AVAILABLE:
            c_ex2.download_button("PDF Export", Exporter.to_pdf(df), "rohrbuch.pdf")

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------

def main():
    DatabaseRepository.init_db()
    df_pipe = get_pipe_data()
    calc = PipeCalculator(df_pipe)

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Einstellungen")
        sel_dn = st.selectbox("Nennweite", df_pipe['DN'], index=8)
        sel_pn = st.radio("Druckstufe", ["PN 16", "PN 10"], horizontal=True)

    # Tabs
    t1, t2, t3, t4 = st.tabs(["🪚 Werkstatt", "📘 Daten", "📝 Rohrbuch", "🔄 Tools"])
    
    with t1: render_tab_workshop(calc, df_pipe, sel_dn, sel_pn)
    with t2: render_tab_handbook(calc, sel_dn, sel_pn)
    with t3: render_tab_logbook(df_pipe)
    with t4: 
        st.info("Bogen & Stutzen Rechner hier einfügen...")

if __name__ == "__main__":
    main()
