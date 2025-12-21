import streamlit as st
import pandas as pd
import math
import json
from datetime import datetime
from io import BytesIO

# --- OPTIONAL IMPORTS (SOFT FAIL) ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. CORE ARCHITECTURE: DATA & PHYSICS (NO UI HERE!)
# -----------------------------------------------------------------------------

class PipeDatabase:
    """
    Single Source of Truth.
    Verwendet ein Dictionary für O(1) Zugriffszeit und Datenintegrität.
    """
    # Struktur: DN: [Außen-Ø, Wand_Std, Radius_BA3, Flansch_K, Schrauben_Anz, Gewinde]
    _DB = {
        25:   {'da': 33.7,  's': 2.6, 'r': 38,   'k': 85,  'n': 4,  'm': 'M12'},
        32:   {'da': 42.4,  's': 2.6, 'r': 48,   'k': 100, 'n': 4,  'm': 'M16'},
        40:   {'da': 48.3,  's': 2.6, 'r': 57,   'k': 110, 'n': 4,  'm': 'M16'},
        50:   {'da': 60.3,  's': 2.9, 'r': 76,   'k': 125, 'n': 4,  'm': 'M16'},
        65:   {'da': 76.1,  's': 2.9, 'r': 95,   'k': 145, 'n': 8,  'm': 'M16'}, # Norm-Achtung: Manchmal 4 Loch
        80:   {'da': 88.9,  's': 3.2, 'r': 114,  'k': 160, 'n': 8,  'm': 'M16'},
        100:  {'da': 114.3, 's': 3.6, 'r': 152,  'k': 180, 'n': 8,  'm': 'M16'},
        125:  {'da': 139.7, 's': 4.0, 'r': 190,  'k': 210, 'n': 8,  'm': 'M16'},
        150:  {'da': 168.3, 's': 4.5, 'r': 229,  'k': 240, 'n': 8,  'm': 'M20'},
        200:  {'da': 219.1, 's': 6.3, 'r': 305,  'k': 295, 'n': 12, 'm': 'M20'},
        250:  {'da': 273.0, 's': 6.3, 'r': 381,  'k': 355, 'n': 12, 'm': 'M24'},
        300:  {'da': 323.9, 's': 7.1, 'r': 457,  'k': 410, 'n': 12, 'm': 'M24'},
        400:  {'da': 406.4, 's': 8.8, 'r': 610,  'k': 525, 'n': 16, 'm': 'M27'},
        500:  {'da': 508.0, 's': 10.0, 'r': 762, 'k': 650, 'n': 20, 'm': 'M30'}
    }

    @classmethod
    def get(cls, dn):
        return cls._DB.get(dn, {'da': 0, 's': 0, 'r': 0, 'k': 0, 'n': 0, 'm': '?'})

    @classmethod
    def get_all_dn(cls):
        return sorted(cls._DB.keys())

class EngineeringCore:
    """
    Reine Berechnungslogik.
    """
    @staticmethod
    def calc_weight(da_mm, s_mm, length_mm, density=7.85):
        """Berechnet Rohrgewicht in kg."""
        if length_mm <= 0: return 0.0
        da_dm = da_mm / 100
        di_dm = (da_mm - 2*s_mm) / 100
        l_dm = length_mm / 100
        vol_dm3 = (math.pi * (da_dm**2 - di_dm**2) / 4) * l_dm
        return round(vol_dm3 * density, 2)

    @staticmethod
    def calc_etage_2d(h, l, radius):
        """
        Berechnet 2D-Etage.
        Returns: (Diagonale_Mitte, Winkel, Schnittlänge)
        """
        hyp = math.sqrt(h**2 + l**2)
        if l == 0: return h, 90.0, h - (2*radius)
        angle = math.degrees(math.atan(h/l))
        # Tangentenlänge (Abzug pro Bogen) = R * tan(alpha/2)
        abzug = 2 * (radius * math.tan(math.radians(angle/2)))
        cut_len = hyp - abzug
        return round(hyp, 1), round(angle, 1), round(cut_len, 1)

    @staticmethod
    def calc_weld_seam(dn, count, method="WIG"):
        """
        Kalkuliert Schweißzeit basierend auf Volumen.
        """
        props = PipeDatabase.get(dn)
        if props['da'] == 0: return 0, 0
        
        # Geometrie V-Naht (ca. s² * faktor + Wurzel)
        s = props['s']
        area_mm2 = (s**2 * 0.6) + (1.5 * s)
        vol_cm3_per_seam = (math.pi * props['da'] * area_mm2) / 1000
        
        # Zeitfaktoren (Minuten pro cm³ Nahtvolumen inkl. Nebenzeit)
        factors = {"WIG": 2.5, "MAG": 1.2, "E-Hand": 1.8}
        time_factor = factors.get(method, 2.0)
        
        # Rüstzeit pro Naht (abhängig von DN)
        fitup_time = (dn / 25) * 4 # 4 min pro Zoll
        
        total_time_min = ((vol_cm3_per_seam * time_factor) + fitup_time) * count
        return round(total_time_min, 0), round(vol_cm3_per_seam * count, 1)

    @staticmethod
    def parse_math_string(text_input):
        """Sicherer Parser für '50+30-2' Eingaben."""
        if not text_input: return 0.0
        allowed = "0123456789.+- "
        if not all(c in allowed for c in text_input):
            return 0.0
        try:
            # Eval ist hier sicher, da input gefiltert
            return float(eval(text_input))
        except:
            return 0.0

# -----------------------------------------------------------------------------
# 2. UI HELPERS & VISUALIZATION
# -----------------------------------------------------------------------------

def plot_etage_dynamic(l, h, cut_len, angle, dn):
    if not PLOT_AVAILABLE: return None
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    # Simple schematic
    ax.plot([0, l], [0, 0], 'k--', alpha=0.3) # Boden
    ax.plot([l, l], [0, h], 'k--', alpha=0.3) # Höhe
    ax.plot([0, l], [0, h], color='#2563eb', linewidth=4, solid_capstyle='round') # Rohr
    
    # Annotations
    ax.text(l/2, -h*0.15 if h>0 else h*0.15, f"L={l}", ha='center', fontsize=9)
    ax.text(l*1.05, h/2, f"H={h}", va='center', fontsize=9)
    
    # Result Box
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="#2563eb", lw=1)
    ax.text(l/2, h/2, f"Säge: {cut_len} mm\n({angle}°)", ha='center', va='center', bbox=bbox_props, fontweight='bold', color='#1e3a8a')
    
    ax.set_title(f"Isometrie Vorschau (DN {dn})", loc='left', fontsize=10)
    ax.axis('off')
    ax.axis('equal')
    return fig

# -----------------------------------------------------------------------------
# 3. REPORTING & EXPORT
# -----------------------------------------------------------------------------

def generate_pdf_report(project_name, items):
    if not PDF_AVAILABLE: return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Projektbericht: {project_name}", 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Erstellt am: {datetime.now().strftime('%d.%m.%Y')}", 0, 1)
    pdf.ln(5)
    
    # Table Header
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(20, 10, "Pos", 1, 0, 'C', 1)
    pdf.cell(30, 10, "Kategorie", 1, 0, 'L', 1)
    pdf.cell(80, 10, "Beschreibung", 1, 0, 'L', 1)
    pdf.cell(30, 10, "Menge/Maß", 1, 0, 'C', 1)
    pdf.cell(30, 10, "Kosten (€)", 1, 1, 'R', 1)
    
    total = 0
    for i, item in enumerate(items):
        cost = item.get('cost', 0)
        total += cost
        pdf.cell(20, 10, str(i+1), 1)
        pdf.cell(30, 10, str(item.get('cat', '-')), 1)
        pdf.cell(80, 10, str(item.get('desc', '-')), 1)
        pdf.cell(30, 10, str(item.get('qty', '-')), 1, 0, 'C')
        pdf.cell(30, 10, f"{cost:.2f}", 1, 1, 'R')
        
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Gesamtsumme: {total:.2f} EUR", 0, 1, 'R')
    
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

st.set_page_config(page_title="PipeCraft V2.0 Enterprise", page_icon="🏭", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; }
    .metric-box {
        background: white; padding: 15px; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6;
    }
    .metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: white; border-radius: 4px; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #e0f2fe; color: #0369a1; border-bottom: 2px solid #0369a1; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization (No SQLite!)
if 'project_items' not in st.session_state: st.session_state.project_items = []
if 'settings' not in st.session_state: 
    st.session_state.settings = {'lohn': 65.0, 'mat_factor': 1.1}

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    active_dn = st.selectbox("Globale Nennweite", PipeDatabase.get_all_dn(), index=6)
    
    # Live Data Display
    props = PipeDatabase.get(active_dn)
    st.markdown("### ℹ️ Technische Daten")
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-label'>Außen-Ø</div>
        <div class='metric-value'>{props['da']} mm</div>
        <hr style='margin: 8px 0;'>
        <div class='metric-label'>Wandstärke</div>
        <div class='metric-value'>{props['s']} mm</div>
        <hr style='margin: 8px 0;'>
        <div class='metric-label'>Bogenradius (3D)</div>
        <div class='metric-value'>{props['r']} mm</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💰 Stundensätze")
    st.session_state.settings['lohn'] = st.number_input("Montage (€/h)", value=st.session_state.settings['lohn'])

# --- MAIN TABS ---
st.title("PipeCraft V2.0 Enterprise")
tab_calc, tab_weld, tab_report = st.tabs(["📐 Vorrichter-Tools", "🔥 Schweiß-Manager", "📊 Projekt-Report"])

# --- TAB 1: VORRICHTER ---
with tab_calc:
    col_tools_1, col_tools_2 = st.columns([1, 1])
    
    with col_tools_1:
        st.subheader("Etagen-Rechner")
        l_in = st.number_input("Länge L (mm)", value=500.0, step=10.0)
        h_in = st.number_input("Höhe H (mm)", value=300.0, step=10.0)
        
        # Physics Call
        diag, ang, cut = EngineeringCore.calc_etage_2d(h_in, l_in, props['r'])
        
        st.success(f"➡️ **Sägelänge: {cut} mm**")
        c1, c2 = st.columns(2)
        c1.metric("Mitte-Mitte", f"{diag} mm")
        c2.metric("Winkel", f"{ang}°")
        
        if st.button("➕ Zur Liste hinzufügen", key="add_etage"):
            st.session_state.project_items.append({
                'cat': 'Vorfertigung',
                'desc': f"Etage DN{active_dn} (L={l_in}, H={h_in})",
                'qty': f"{cut} mm",
                'cost': 0.0 # Intern
            })
            st.toast("Hinzugefügt!")

    with col_tools_2:
        st.subheader("Visualisierung")
        fig = plot_etage_dynamic(l_in, h_in, cut, ang, active_dn)
        if fig: st.pyplot(fig)
        
        st.markdown("#### Passstück Gewicht")
        weight = EngineeringCore.calc_weight(props['da'], props['s'], cut)
        st.info(f"Gewicht (Stahl): **{weight} kg**")

# --- TAB 2: SCHWEISSEN ---
with tab_weld:
    st.subheader("Naht-Kalkulation (Volumenbasiert)")
    
    c_w1, c_w2 = st.columns(2)
    with c_w1:
        weld_method = st.selectbox("Verfahren", ["WIG", "MAG", "E-Hand"])
        weld_count = st.number_input("Anzahl Nähte", value=10, step=1)
        
        # Physics Call
        time_min, vol_tot = EngineeringCore.calc_weld_seam(active_dn, weld_count, weld_method)
        cost_calc = (time_min / 60) * st.session_state.settings['lohn']
        
    with c_w2:
        st.metric("Kalkulierte Zeit", f"{time_min} min")
        st.metric("Gesamtkosten", f"{cost_calc:.2f} €")
        st.caption(f"Basierend auf {vol_tot} cm³ Schweißgut")
        
        if st.button("➕ In Projekt übernehmen", key="add_weld"):
            st.session_state.project_items.append({
                'cat': 'Schweißen',
                'desc': f"DN{active_dn} {weld_method} Nähte",
                'qty': int(weld_count),
                'cost': cost_calc
            })
            st.toast("Kalkulation gespeichert!")

# --- TAB 3: REPORTING ---
with tab_report:
    st.subheader("Projektübersicht")
    
    if st.session_state.project_items:
        df_rep = pd.DataFrame(st.session_state.project_items)
        st.dataframe(df_rep, use_container_width=True)
        
        col_export_1, col_export_2 = st.columns(2)
        
        # Total Sum
        total_proj = sum(item['cost'] for item in st.session_state.project_items)
        st.metric("Gesamtsumme Projekt", f"{total_proj:.2f} €")
        
        if col_export_1.button("🗑️ Projekt leeren"):
            st.session_state.project_items = []
            st.rerun()
            
        if PDF_AVAILABLE:
            pdf_bytes = generate_pdf_report("Baustelle_XYZ", st.session_state.project_items)
            col_export_2.download_button("📄 PDF Bericht", data=pdf_bytes, file_name="Report.pdf", mime="application/pdf", type="primary")
    else:
        st.info("Noch keine Positionen im Projekt. Nutze die Tools in den anderen Tabs.")
