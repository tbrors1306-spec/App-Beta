"""
PipeCraft V47.0 (Reference Book Master Edition)
-----------------------------------------------
Fokus: Revolutioniertes Tabellenbuch-Modul.

Features Tabellenbuch:
1.  Independent Lookup: Nachschlagen von Maßen ohne Änderung der Projekt-Config.
2.  TechDraw Engine: Generiert technische Zeichnungen (Flansch-Lochbilder) live.
3.  Material-Engine: Gewichtsberechnung für Stahl, V2A, V4A.
4.  Smart Compare: Intelligente Anzeige von PN-Stufen.

Author: Senior Lead Software Engineer
"""

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sqlite3
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Union

# -----------------------------------------------------------------------------
# 0. SYSTEM SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="PipeCraft V47.0", page_icon="📘", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #0f172a; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 700; color: #1e293b; }
    
    /* Karten-Design für Tabellenbuch */
    .tech-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .tech-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .tech-value { font-size: 1.25rem; font-weight: 600; color: #0f172a; }
    
    .stSelectbox div[data-baseweb="select"] { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. DATA LAYER (NORMEN)
# -----------------------------------------------------------------------------
RAW_DATA = {
    'DN':           [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500],
    'D_Aussen':     [33.7, 42.4, 48.3, 60.3, 76.1, 88.9, 114.3, 139.7, 168.3, 219.1, 273.0, 323.9, 355.6, 406.4, 457.0, 508.0],
    'Radius_BA3':   [38, 48, 57, 76, 95, 114, 152, 190, 229, 305, 381, 457, 533, 610, 686, 762],
    # Flansch PN 16
    'Flansch_b_16': [16, 16, 16, 18, 18, 20, 20, 22, 22, 24, 26, 28, 30, 32, 32, 34], # Dicken angepasst (Praxisnähe)
    'LK_k_16':      [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 355, 410, 470, 525, 585, 650],
    'Lochzahl_16':  [4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 16, 16, 20, 20],
    'Schraube_M_16':["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M24", "M24", "M24", "M27", "M27", "M30"],
    # Flansch PN 10 (Abweichungen meist erst ab DN 200 relevant)
    'Flansch_b_10': [16, 16, 16, 18, 18, 20, 20, 22, 22, 24, 26, 26, 26, 26, 28, 28],
    'LK_k_10':      [85, 100, 110, 125, 145, 160, 180, 210, 240, 295, 350, 400, 460, 515, 565, 620],
    'Lochzahl_10':  [4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 16, 16, 20, 20],
    'Schraube_M_10':["M12", "M16", "M16", "M16", "M16", "M16", "M16", "M16", "M20", "M20", "M20", "M20", "M20", "M24", "M24", "M24"]
}

df_pipe = pd.DataFrame(RAW_DATA)

MATERIALS = {
    "Stahl (S235/P235)": 7.85,
    "Edelstahl (1.4301/V2A)": 7.90,
    "Edelstahl (1.4571/V4A)": 7.98,
    "Aluminium": 2.70
}

# -----------------------------------------------------------------------------
# 2. LOGIC & VISUALIZATION ENGINE
# -----------------------------------------------------------------------------

class TechDrawEngine:
    """Generiert technische Zeichnungen on-the-fly."""
    
    @staticmethod
    def draw_flange_face(dn: int, da: float, lk: float, holes: int, bolt_size: str):
        """Zeichnet eine Flansch-Draufsicht mit Lochbild."""
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Radien
        r_outer = (lk + 40) / 2 # Annäherung Blattaußenmaß
        r_lk = lk / 2
        r_inner = (da - 6.3) / 2 # Annäherung ID (Standard WS)
        
        # 1. Hauptkörper
        circle_outer = plt.Circle((0, 0), r_outer, color='#cbd5e1', fill=True, alpha=0.3)
        circle_inner = plt.Circle((0, 0), r_inner, color='white', fill=True)
        ax.add_artist(circle_outer)
        ax.add_artist(circle_inner)
        
        # 2. Lochkreis (Strichlinie)
        circle_lk = plt.Circle((0, 0), r_lk, color='#64748b', fill=False, linestyle='--')
        ax.add_artist(circle_lk)
        
        # 3. Schraubenlöcher
        # Bolt size string parsing (M16 -> 16mm -> Hole ~18mm)
        try:
            bolt_dia = int(bolt_size.replace('M', '')) + 2
        except: bolt_dia = 18
            
        for i in range(holes):
            angle = (2 * math.pi / holes) * i
            x = r_lk * math.cos(angle)
            y = r_lk * math.sin(angle)
            hole = plt.Circle((x, y), bolt_dia/2, color='white', ec='#334155', linewidth=1)
            ax.add_artist(hole)
            
        # Crosshair center
        ax.plot([-r_outer*1.1, r_outer*1.1], [0, 0], 'k-.', linewidth=0.5, alpha=0.5)
        ax.plot([0, 0], [-r_outer*1.1, r_outer*1.1], 'k-.', linewidth=0.5, alpha=0.5)
        
        # Limits & Cleanup
        limit = r_outer * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig

# -----------------------------------------------------------------------------
# 3. DATABASE REPOSITORY (Reduced for Stability)
# -----------------------------------------------------------------------------
# (Hier nur minimalistisch, da Fokus auf Tabellenbuch)
DB_NAME = "pipecraft_v47.db"
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute('CREATE TABLE IF NOT EXISTS dummy (id INTEGER)')
init_db()

# -----------------------------------------------------------------------------
# 4. UI IMPLEMENTATION
# -----------------------------------------------------------------------------

# --- SIDEBAR (Global Context - aber Tabellenbuch ist jetzt unabhängig!) ---
st.sidebar.markdown("### ⚙️ Projekt-Kontext")
st.sidebar.info("Das Tabellenbuch (rechts) funktioniert jetzt unabhängig von diesen Einstellungen.")
_ = st.sidebar.selectbox("Projekt DN", df_pipe['DN'], index=8) 

# --- MAIN APP ---
st.title("PipeCraft V47.0")

# Wir konzentrieren uns nur auf den ersten Tab, wie gewünscht
tab_buch, tab_werk, tab_proj = st.tabs(["📘 Tabellenbuch (Advanced)", "📐 Werkstatt", "📝 Rohrbuch"])

# ==============================================================================
# DAS NEUE TABELLENBUCH
# ==============================================================================
with tab_buch:
    st.markdown("### 🔍 Interaktiver Norm-Katalog")
    
    # 1. CONTROL PANEL (Unabhängig vom Global State)
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    
    with col_ctrl1:
        lookup_dn = st.selectbox("Nennweite (DN)", df_pipe['DN'], index=6) # Default DN 100
    
    with col_ctrl2:
        lookup_pn = st.radio("Druckstufe", ["PN 10", "PN 16", "Vergleich"], index=1)
    
    with col_ctrl3:
        lookup_mat = st.selectbox("Material (Dichte)", list(MATERIALS.keys()), index=0)
        density = MATERIALS[lookup_mat]

    # Daten holen
    row = df_pipe[df_pipe['DN'] == lookup_dn].iloc[0]
    
    st.divider()

    # 2. DISPLAY PANEL
    c_left, c_right = st.columns([1, 1.5])
    
    # --- LINKE SPALTE: DATEN & RECHNER ---
    with c_left:
        st.markdown("#### 📏 Rohrdimensionen")
        
        # Flexible Wandstärke Eingabe
        std_ws = 3.6 if lookup_dn <= 40 else (4.0 if lookup_dn <= 100 else (6.3 if lookup_dn <= 200 else 8.8))
        custom_ws = st.number_input("Wandstärke (mm)", value=std_ws, step=0.1, format="%.1f")
        
        # Berechnete Werte
        di = row['D_Aussen'] - (2 * custom_ws)
        # Gewicht pro Meter: Fläche * Dichte
        # Fläche A = pi * (Ra^2 - Ri^2)
        # Gewicht = A (dm²) * 10 (dm) * Dichte
        ra_dm = (row['D_Aussen']/2) / 100
        ri_dm = (di/2) / 100
        area_dm2 = math.pi * (ra_dm**2 - ri_dm**2)
        weight_m = area_dm2 * 10 * density
        
        # Visualisierung Cards
        st.markdown(f"""
        <div class="tech-card">
            <div class="tech-label">Außendurchmesser (Da)</div>
            <div class="tech-value">{row['D_Aussen']} mm</div>
        </div>
        <div class="tech-card">
            <div class="tech-label">Innendurchmesser (Di)</div>
            <div class="tech-value">{round(di, 1)} mm</div>
        </div>
        <div class="tech-card" style="border-left: 4px solid #f59e0b;">
            <div class="tech-label">Metergewicht ({lookup_mat.split()[0]})</div>
            <div class="tech-value">{round(weight_m, 2)} kg/m</div>
        </div>
        """, unsafe_allow_html=True)

    # --- RECHTE SPALTE: VISUALISIERUNG & FLANSCH ---
    with c_right:
        st.markdown("#### ⚙️ Flansch & Lochbild")
        
        # Daten vorbereiten je nach PN Modus
        pns_to_show = []
        if lookup_pn == "Vergleich": pns_to_show = ["10", "16"]
        elif lookup_pn == "PN 10": pns_to_show = ["10"]
        else: pns_to_show = ["16"]
        
        cols_pn = st.columns(len(pns_to_show))
        
        for idx, pn_suffix in enumerate(pns_to_show):
            with cols_pn[idx]:
                key_suffix = f"_{pn_suffix}" # z.B. _10 oder _16
                
                # Daten aus Row holen
                lk = row[f'LK_k{key_suffix}']
                holes = row[f'Lochzahl{key_suffix}']
                bolt = row[f'Schraube_M{key_suffix}']
                thick = row[f'Flansch_b{key_suffix}']
                
                st.markdown(f"**PN {pn_suffix}**")
                
                # Matplotlib Drawing
                fig = TechDrawEngine.draw_flange_face(lookup_dn, row['D_Aussen'], lk, holes, bolt)
                st.pyplot(fig, use_container_width=True)
                
                # Detail Tabelle
                st.markdown(f"""
                | Parameter | Wert |
                | :--- | :--- |
                | **Lochkreis (k)** | **{lk} mm** |
                | Schrauben | {holes}x {bolt} |
                | Blattstärke (b) | {thick} mm |
                """)

# ==============================================================================
# PLACEHOLDERS (Andere Tabs nur rudimentär, um Fokus zu wahren)
# ==============================================================================
with tab_werk:
    st.info("Werkstatt-Module (Säge/Stutzen/Bogen) sind in V46.1 voll implementiert. Hier fokussieren wir auf das Tabellenbuch.")

with tab_proj:
    st.info("Rohrbuch-Module sind in V46.1 voll implementiert.")
