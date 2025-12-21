"""
PipeCraft Enterprise Edition (V31.0 - Clean Code)
-------------------------------------------------
Comprehensive Piping Engineering & Management Suite.

Modules:
1. Data Layer: Static Norms & Dynamic SQLite Database.
2. Logic Layer: Physics, Geometry, Costing, Fitting Management.
3. Presentation Layer: Streamlit UI with 3D Visualization.

Standards applied: PEP 8, Type Hinting, Explicit Control Flow.
Author: Senior Lead Software Engineer
"""

import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Explicit import required for 3D plotting in older matplotlib versions
from mpl_toolkits.mplot3d import Axes3D 
import sqlite3
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import List, Tuple, Any, Optional, Union, Dict

# -----------------------------------------------------------------------------
# 0. SYSTEM CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipeCraft")

# Optional PDF Support
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("FPDF Library not found. PDF Export features will be disabled.")

# Streamlit Configuration
st.set_page_config(
    page_title="PipeCraft V31.0 Enterprise",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS Styling
st.markdown("""
<style>
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
    
    /* Result Cards */
    .result-card-blue { 
        background-color: #eff6ff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #3b82f6; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        margin-bottom: 10px; 
        color: #1e3a8a; 
        font-size: 1rem; 
    }
    
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
    
    /* Detail Boxes */
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
    
    /* Weight/Warning Boxes */
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
    
    /* Input Field Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] { 
        border-radius: 8px; 
        border: 1px solid #cbd5e1; 
    }
    
    /* Button Styling */
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
# 1. DATA LAYER (STATIC DATA & REPOSITORIES)
# -----------------------------------------------------------------------------

# Raw Data for Piping Components (Norms)
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

# DataFrame Initialization with Integrity Check
try:
    df_pipe = pd.DataFrame(RAW_DATA)
except ValueError as e:
    st.error(f"FATAL ERROR: Inconsistent Data Arrays. Details: {e}")
    st.stop()

# Helper Constants & Lookup Tables
SCHRAUBEN_DB = { 
    "M12": [18, 60], "M16": [24, 130], "M20": [30, 250], "M24": [36, 420], 
    "M27": [41, 600], "M30": [46, 830], "M33": [50, 1100], "M36": [55, 1400], 
    "M39": [60, 1800], "M45": [70, 2700], "M52": [80, 4200] 
}

WS_LISTE = [2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 8.8, 10.0, 11.0, 12.5, 14.2, 16.0]

WS_STD_MAP = {
    25: 3.2, 32: 3.6, 40: 3.6, 50: 3.9, 65: 5.2, 80: 5.5, 100: 6.0, 
    125: 6.6, 150: 7.1, 200: 8.2, 250: 9.3, 300: 9.5, 350: 9.5, 400: 9.5, 
    450: 9.5, 500: 9.5
}

DB_NAME = "pipecraft.db"

# -----------------------------------------------------------------------------
# 2. BUSINESS LOGIC LAYER (ENGINES & MANAGERS)
# -----------------------------------------------------------------------------

# --- 2.1 Fitting Management (Smart Cut) ---

@dataclass
class SelectedFitting:
    """Represents a fitting selected by the user for deduction calculation."""
    type_name: str
    count: int
    deduction_single: float

class FittingManager:
    """
    Manages logic for determining deduction lengths (Z-measures) for various fittings.
    Central point of truth for fitting dimensions.
    """
    
    @staticmethod
    def get_deduction(type_name: str, row_data: pd.Series, pn_suffix: str = "_16", custom_angle: float = 45.0) -> float:
        """
        Retrieves the correct deduction length based on fitting type and pipe dimensions.
        """
        if type_name == "Bogen 90° (BA3)":
            return float(row_data['Radius_BA3'])
        
        elif type_name == "Bogen (Zuschnitt)":
            # Calculation: Radius * tan(alpha / 2)
            radius = float(row_data['Radius_BA3'])
            angle_rad = math.radians(custom_angle / 2)
            return radius * math.tan(angle_rad)
            
        elif type_name == "Flansch (Vorschweiß)":
            # Deduction is the flange leaf thickness (Blattstärke)
            return float(row_data[f'Flansch_b{pn_suffix}'])
            
        elif type_name == "T-Stück":
            # Standard Tee Height H
            return float(row_data['T_Stueck_H'])
            
        elif type_name == "Reduzierung (konz.)":
            # Standard Reducer Length L
            return float(row_data['Red_Laenge_L'])
            
        return 0.0

# --- 2.2 Inventory Logic ---

@dataclass
class InventoryItem:
    """Domain Model for an item in the warehouse."""
    article_id: str
    name: str
    price_per_unit: float
    current_stock: int
    reorder_point: int
    target_stock: int

    def calculate_reorder_qty(self) -> int:
        """
        Calculates how many items need to be ordered.
        Rule: Order only if current <= reorder_point. Fill up to target.
        """
        if self.current_stock <= self.reorder_point:
            return self.target_stock - self.current_stock
        return 0
        
    @property
    def stock_value(self) -> float:
        """Calculates total bound capital for this item."""
        return self.current_stock * self.price_per_unit

# --- 2.3 Physics Engine ---

class PhysicsEngine:
    """Calculates physical properties like weight and volume."""
    
    DENSITY_STEEL = 7.85  # kg/dm³
    DENSITY_CEMENT = 2.40 # kg/dm³
    
    @staticmethod
    def calculate_pipe_weight(dn_idx: int, ws: float, length_mm: float, is_zme: bool = False) -> float:
        """
        Calculates the total weight of a pipe segment.
        Includes optional cement lining (ZME).
        Formula: V = Length * Pi * (OuterRadius^2 - InnerRadius^2)
        """
        try:
            da_mm = df_pipe.iloc[dn_idx]['D_Aussen']
            
            # Conversion to decimeters (dm) for density calculation
            # 1 dm = 100 mm
            length_dm = length_mm / 100.0
            ra_dm = (da_mm / 2) / 100.0
            ri_stahl_dm = ra_dm - (ws / 100.0)
            
            # Calculate Steel Volume
            vol_stahl = math.pi * (ra_dm**2 - ri_stahl_dm**2) * length_dm
            weight = vol_stahl * PhysicsEngine.DENSITY_STEEL
            
            # Calculate Cement Volume (Optional)
            if is_zme:
                dn_val = df_pipe.iloc[dn_idx]['DN']
                # Estimating cement thickness based on DIN 2614
                if dn_val < 300:
                    cem_th_mm = 6.0
                elif dn_val < 600:
                    cem_th_mm = 9.0
                else:
                    cem_th_mm = 12.0
                    
                ri_cem_dm = ri_stahl_dm - (cem_th_mm / 100.0)
                
                if ri_cem_dm > 0:
                    vol_cem = math.pi * (ri_stahl_dm**2 - ri_cem_dm**2) * length_dm
                    weight += (vol_cem * PhysicsEngine.DENSITY_CEMENT)
                    
            return round(weight, 1)
            
        except Exception as e:
            logger.error(f"Error calculating weight: {e}")
            return 0.0

# --- 2.4 Geometry Engine ---

class GeometryEngine:
    """Calculates 3D coordinates and trigonometry for piping offsets."""
    
    @staticmethod
    def solve_offset_3d(h: float, l: float, b: float) -> Tuple[float, float]:
        """
        Solves a 3D piping offset (Etage).
        
        Args:
            h: Height (Vertical offset)
            l: Length (Horizontal offset)
            b: Breadth (Depth offset / "Sprung")
            
        Returns:
            Tuple containing:
            - Travel Length (Diagonal in 3D space)
            - Vertical Angle (Elevation angle relative to ground plane)
        """
        # Travel (3D Diagonal)
        travel = math.sqrt(h**2 + l**2 + b**2)
        
        # Spread (Projection on ground plane)
        spread = math.sqrt(l**2 + b**2)
        
        # Calculate Angle
        if spread == 0:
            angle = 90.0
        else:
            angle = math.degrees(math.atan(h / spread))
            
        return travel, angle

# --- 2.5 Cost Engine ---

class CostEngine:
    """Estimates time and cost for piping activities."""
    
    @staticmethod
    def estimate_welding_time(dn: int, ws: float, process: str) -> float:
        """
        Estimates welding time in minutes.
        Based on 'Inch-Diameter' rule adjusted by wall thickness and process.
        """
        zoll = dn / 25.0
        
        # Base minutes per inch
        if "WIG" in process:
            base_min = 12.0
        elif "CEL" in process:
            base_min = 4.5
        else:
            base_min = 8.0 # Standard for E-Hand / MAG
            
        # Wall thickness factor (Non-linear increase for thick walls)
        ws_factor = 1.0
        if ws > 6.0:
            ws_factor = ws / 6.0
            
        return (zoll * base_min * ws_factor)

# --- 2.6 Helpers ---

def get_schrauben_info(gewinde: str) -> List[Union[int, str]]:
    """Safe retrieval of bolt details."""
    return SCHRAUBEN_DB.get(gewinde, ["?", "?"])

def parse_abzuege(text: str) -> float:
    """
    Parses a mathematical string input safely.
    Example: '50+30' -> 80.0
    """
    try:
        if not text:
            return 0.0
        clean_text = text.replace(",", ".").replace(" ", "")
        
        # Security whitelist check
        allowed_chars = set("0123456789.+-*/()")
        if not set(clean_text).issubset(allowed_chars):
            return 0.0
            
        return float(pd.eval(clean_text))
    except Exception:
        return 0.0

# UI State Helper Functions to prevent IndexErrors
def get_ws_index(val: float) -> int:
    if val in WS_LISTE:
        return WS_LISTE.index(val)
    return 6 # Default index

def get_verf_index(val: str) -> int:
    opts = ["WIG", "E-Hand (CEL 70)", "WIG + E-Hand", "MAG"]
    if val in opts:
        return opts.index(val)
    return 0

def get_disc_idx(val: str) -> int:
    opts = ["125 mm", "180 mm", "230 mm"]
    if val in opts:
        return opts.index(val)
    return 0

def get_sys_idx(val: str) -> int:
    opts = ["Schrumpfschlauch (WKS)", "B80 Band (Einband)", "B50 + Folie (Zweiband)"]
    if val in opts:
        return opts.index(val)
    return 0

# -----------------------------------------------------------------------------
# 3. VISUALIZATION LAYER (2D & 3D PLOTTING)
# -----------------------------------------------------------------------------

class Visualizer:
    """Handles all matplotlib rendering."""
    
    @staticmethod
    def plot_true_3d_pipe(length: float, width: float, height: float, azim: int, elev: int) -> plt.Figure:
        """
        Renders an interactive-like 3D plot of the pipe segment.
        """
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        # Coordinates: Start (0,0,0) to End (L, B, H)
        xs = [0, length]
        ys = [0, width]
        zs = [0, height]
        
        # Draw the Pipe Vector
        ax.plot(xs, ys, zs, color='#ef4444', linewidth=5, solid_capstyle='round', label='Rohr')
        
        # Draw Bounding Box (Projection Lines) for spatial context
        # Ground projection
        ax.plot([0, length], [0, 0], [0, 0], 'k--', alpha=0.2)
        ax.plot([length, length], [0, width], [0, 0], 'k--', alpha=0.2)
        
        # Back wall projection
        ax.plot([0, length], [width, width], [0, 0], 'k--', alpha=0.1)
        
        # Height projection
        ax.plot([length, length], [width, width], [0, height], 'k--', alpha=0.3)
        
        # Markers for Start and End
        ax.scatter([0], [0], [0], color='black', s=50, label='Start')
        ax.scatter([length], [width], [height], color='#10b981', s=50, label='Ende')
        
        # Labels
        ax.set_xlabel('Länge (X)')
        ax.set_ylabel('Breite (Y)')
        ax.set_zlabel('Höhe (Z)')
        
        # Aspect Ratio Fix (Matplotlib 3D often distorts axes)
        max_dim = max(abs(length), abs(width), abs(height))
        if max_dim == 0:
            max_dim = 100
            
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        
        # Set Camera View
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_stutzen_curve(r_haupt: float, r_stutzen: float) -> plt.Figure:
        """Calculates and plots the intersection curve (unrolled) for a nozzle."""
        angles = range(0, 361, 5)
        try:
            # Formula for cylinder-cylinder intersection unrolling
            depths = [r_haupt - math.sqrt(r_haupt**2 - (r_stutzen * math.sin(math.radians(a)))**2) for a in angles]
        except ValueError:
            # Return empty figure if calculation is mathematically impossible (Stutzen > Haupt)
            return plt.figure()

        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.plot(angles, depths, color='#3b82f6', linewidth=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360)
        ax.axis('off') # Clean look
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

# -----------------------------------------------------------------------------
# 4. DATABASE REPOSITORY SERVICE
# -----------------------------------------------------------------------------

class DatabaseRepository:
    """
    Encapsulates all database operations.
    Uses 'with' statements for safe connection handling.
    """
    
    @staticmethod
    def init_tables():
        """Initializes all database tables."""
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            
            # 1. Rohrbuch (Documentation)
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
            
            # 2. Kalkulation (Calculation)
            c.execute('''CREATE TABLE IF NOT EXISTS kalkulation (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        typ TEXT, 
                        info TEXT, 
                        menge REAL, 
                        zeit_min REAL, 
                        kosten REAL, 
                        mat_text TEXT)''')
                        
            # 3. Inventory (Logistics)
            c.execute('''CREATE TABLE IF NOT EXISTS inventory (
                        article_id TEXT PRIMARY KEY, 
                        name TEXT, 
                        price REAL, 
                        current_stock INTEGER, 
                        reorder_point INTEGER, 
                        target_stock INTEGER)''')
            conn.commit()

    # --- INVENTORY OPERATIONS ---
    
    @staticmethod
    def add_inventory_item(item: InventoryItem) -> bool:
        """Adds a new item to inventory. Returns False if ID exists."""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.cursor().execute(
                    'INSERT INTO inventory (article_id, name, price, current_stock, reorder_point, target_stock) VALUES (?,?,?,?,?,?)', 
                    (item.article_id, item.name, item.price_per_unit, item.current_stock, item.reorder_point, item.target_stock)
                )
                return True
        except sqlite3.IntegrityError:
            return False

    @staticmethod
    def get_inventory_item(article_id: str) -> Optional[InventoryItem]:
        """Retrieves a specific item."""
        with sqlite3.connect(DB_NAME) as conn:
            row = conn.cursor().execute('SELECT * FROM inventory WHERE article_id=?', (article_id,)).fetchone()
            if row:
                return InventoryItem(row[0], row[1], row[2], row[3], row[4], row[5])
            return None

    @staticmethod
    def get_all_inventory() -> List[InventoryItem]:
        """Retrieves all inventory items."""
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.cursor().execute('SELECT * FROM inventory').fetchall()
            return [InventoryItem(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    @staticmethod
    def update_stock(article_id: str, new_qty: int):
        """Updates the stock quantity of an item."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute('UPDATE inventory SET current_stock=? WHERE article_id=?', (new_qty, article_id))
            conn.commit()

    # --- GENERIC OPERATIONS ---

    @staticmethod
    def add(table: str, data: Tuple):
        """Generic add function for logs and calculation tables."""
        with sqlite3.connect(DB_NAME) as conn:
            placeholders = ",".join(["?"] * len(data))
            
            if table == "rohrbuch":
                cols = "iso, naht, datum, dimension, bauteil, laenge, charge, schweisser"
            else:
                cols = "typ, info, menge, zeit_min, kosten, mat_text"
                
            query = f'INSERT INTO {table} ({cols}) VALUES ({placeholders})'
            conn.cursor().execute(query, data)
            conn.commit()
    
    @staticmethod
    def get_all(table: str) -> pd.DataFrame:
        """Generic get function returning a DataFrame."""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()
    
    @staticmethod
    def delete(table: str, entry_id: int):
        """Generic delete function."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(f"DELETE FROM {table} WHERE id=?", (entry_id,))
            conn.commit()
    
    @staticmethod
    def clear(table: str):
        """Generic clear table function."""
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute(f"DELETE FROM {table}")
            conn.commit()

# --- EXPORT HELPERS ---

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Exports DataFrame to Excel bytes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

def export_to_pdf(df: pd.DataFrame) -> bytes:
    """Exports DataFrame to PDF bytes (Simple List Format)."""
    if not PDF_AVAILABLE:
        return b""
        
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'PipeCraft Report', 0, 1, 'C')
            self.ln(5)
            
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    for index, row in df.iterrows():
        try:
            # Construct a simple line string from the row
            line_text = " | ".join([str(val) for val in row.values])
            # Encode/Decode to handle unicode issues in standard FPDF
            safe_text = line_text.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 10, safe_text, 1, 1)
        except Exception:
            continue
            
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 5. INITIALIZATION & SESSION STATE
# -----------------------------------------------------------------------------

# Initialize DB Tables
DatabaseRepository.init_tables()

# Define Default Session State
DEFAULT_STATE = {
    'saw_mass': 1000.0, 
    'saw_gap': 4.0, 
    'saw_deduct': "0", 
    'saw_zme': False,
    'kw_dn': 200, 
    'kw_ws': 6.3, 
    'kw_verf': "WIG", 
    'kw_pers': 1, 
    'kw_anz': 1, 
    'kw_split': False, 
    'kw_factor': 1.0,
    'cut_dn': 200, 
    'cut_ws': 6.3, 
    'cut_disc': "125 mm", 
    'cut_anz': 1, 
    'cut_zma': False, 
    'cut_iso': False, 
    'cut_factor': 1.0,
    'iso_sys': "Schrumpfschlauch (WKS)", 
    'iso_dn': 200, 
    'iso_anz': 1, 
    'iso_factor': 1.0,
    'mon_dn': 200, 
    'mon_type': "Schieber", 
    'mon_anz': 1, 
    'mon_factor': 1.0,
    'reg_min': 60, 
    'reg_pers': 2, 
    'bogen_winkel': 45, 
    'view_azim': 45, 
    'view_elev': 30,
    'p_lohn': 60.0, 
    'p_stahl': 2.5, 
    'p_dia': 45.0, 
    'p_cel': 0.40, 
    'p_draht': 15.0, 
    'p_gas': 0.05, 
    'p_wks': 25.0, 
    'p_kebu1': 15.0, 
    'p_kebu2': 12.0, 
    'p_primer': 12.0, 
    'p_machine': 15.0
}

# Load Defaults into Session State
if 'store' not in st.session_state:
    st.session_state.store = DEFAULT_STATE.copy()

# Initialize Fitting Cart for "Smart Cut"
if 'fitting_list' not in st.session_state:
    st.session_state.fitting_list = []

# Callbacks
def save_val(key):
    """Saves widget value to persistent store."""
    st.session_state.store[key] = st.session_state[f"_{key}"]

def get_val(key):
    """Retrieves value from persistent store."""
    return st.session_state.store.get(key, DEFAULT_STATE.get(key))

def update_kw_dn():
    """Callback: Updates DN and auto-sets person count for welding."""
    st.session_state.store['kw_dn'] = st.session_state['_kw_dn']
    if st.session_state.store['kw_dn'] >= 300:
        st.session_state.store['kw_pers'] = 2
    else:
        st.session_state.store['kw_pers'] = 1

# -----------------------------------------------------------------------------
# 6. UI IMPLEMENTATION (MAIN)
# -----------------------------------------------------------------------------

# Sidebar Logic
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=50) 
st.sidebar.markdown("### Menü")
selected_dn_global = st.sidebar.selectbox("Nennweite (Global)", df_pipe['DN'], index=8, key="global_dn") 
selected_pn = st.sidebar.radio("Druckstufe", ["PN 16", "PN 10"], index=0, key="global_pn") 

# Retrieve Global Context
row = df_pipe[df_pipe['DN'] == selected_dn_global].iloc[0]
standard_radius = float(row['Radius_BA3'])
suffix = "_16" if selected_pn == "PN 16" else "_10"

st.title("PipeCraft V31.0")
st.caption(f"🔧 Aktive Konfiguration: DN {selected_dn_global} | {selected_pn} | Radius: {standard_radius} mm")

# Main Tabs
tab_buch, tab_werk, tab_proj, tab_info, tab_lager = st.tabs(["📘 Tabellenbuch", "📐 Werkstatt", "📝 Rohrbuch", "💰 Kalkulation", "📦 Lager"])

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
# TAB 2: WERKSTATT (ENGINEERING TOOLS)
# -----------------------------------------------------------------------------
with tab_werk:
    tool_mode = st.radio("Werkzeug wählen:", ["📏 Säge (Passstück)", "🔄 Bogen (Zuschnitt)", "🔥 Stutzen (Schablone)", "📐 Etage (Versatz)"], horizontal=True, label_visibility="collapsed", key="tool_mode_nav")
    st.divider()
    
    # --- SUB-TOOL: SMART CUT / SÄGE ---
    if "Säge" in tool_mode:
        st.subheader("Smart Cut: Passstück Berechnung")
        
        # Basis-Informationen
        c_main1, c_main2 = st.columns(2)
        iso_mass = c_main1.number_input("Gesamtmaß (Iso)", value=get_val('saw_mass'), step=10.0, key="_saw_mass", on_change=save_val, args=('saw_mass',))
        spalt = c_main2.number_input("Wurzelspalt (pro Naht)", value=get_val('saw_gap'), key="_saw_gap", on_change=save_val, args=('saw_gap',))
        
        st.divider()
        st.markdown("#### Formteile hinzufügen (Warenkorb)")
        
        # Formteil-Auswahl UI
        c_add1, c_add2, c_add3, c_add4 = st.columns([2, 1, 1, 1])
        
        fitting_options = [
            "Bogen 90° (BA3)", 
            "Bogen (Zuschnitt)", 
            "Flansch (Vorschweiß)", 
            "T-Stück", 
            "Reduzierung (konz.)"
        ]
        fitting_type = c_add1.selectbox("Bauteil", fitting_options)
        
        # Spezielle UI für Bogen-Zuschnitt
        angle_input = 45.0
        if fitting_type == "Bogen (Zuschnitt)":
            angle_input = c_add2.number_input("Winkel °", value=45.0, step=1.0)
        else:
            c_add2.write("Standard")
            
        count_input = c_add3.number_input("Anzahl", value=1, min_value=1)
        
        # Add Button Logic
        if c_add4.button("Hinzufügen (+)", type="secondary"):
            deduction = FittingManager.get_deduction(fitting_type, row, suffix, angle_input)
            item_name = f"{fitting_type}"
            if fitting_type == "Bogen (Zuschnitt)":
                item_name += f" [{angle_input}°]"
            
            # Objekt in Liste speichern
            new_fitting = SelectedFitting(item_name, count_input, deduction)
            st.session_state.fitting_list.append(new_fitting)
            st.rerun()

        # Liste Anzeigen & Berechnen
        total_deduction = 0.0
        total_gaps = 0
        
        if st.session_state.fitting_list:
            st.markdown("---")
            for idx, item in enumerate(st.session_state.fitting_list):
                col_i1, col_i2, col_i3 = st.columns([4, 2, 1])
                
                sub_total = item.deduction_single * item.count
                total_deduction += sub_total
                total_gaps += item.count # Annahme: 1 Teil = 1 Naht
                
                with col_i1:
                    st.markdown(f"**{item.count}x** {item.type_name}")
                with col_i2:
                    st.caption(f"Abzug: {round(sub_total, 1)} mm")
                with col_i3:
                    if st.button("🗑️", key=f"del_{idx}"):
                        st.session_state.fitting_list.pop(idx)
                        st.rerun()
            
            if st.button("Alle löschen", type="primary"):
                st.session_state.fitting_list = []
                st.rerun()
        
        # Finale Berechnung
        total_spalt_deduction = total_gaps * spalt
        final_cut = iso_mass - total_deduction - total_spalt_deduction
        
        st.markdown("---")
        
        if final_cut < 0:
            st.error(f"Fehler: Die Summe der Abzüge ({round(total_deduction+total_spalt_deduction,1)} mm) ist größer als das Isomaß!")
        else:
            c_res1, c_res2 = st.columns(2)
            c_res1.markdown(f"<div class='result-card-green'>Sägelänge: {round(final_cut, 1)} mm</div>", unsafe_allow_html=True)
            c_res2.info(f"Abzüge: {round(total_deduction,1)} mm (Teile) + {round(total_spalt_deduction,1)} mm (Spalte)")

        # Gewichtsberechnung
        dn_idx = df_pipe[df_pipe['DN'] == selected_dn_global].index[0]
        std_ws = WS_STD_MAP.get(selected_dn_global, 4.0)
        c_zme = st.checkbox("ZME (Beton innen)?", value=get_val('saw_zme'), key="_saw_zme", on_change=save_val, args=('saw_zme',))
        
        kg = 0
        if final_cut > 0:
            kg = PhysicsEngine.calculate_pipe_weight(dn_idx, std_ws, final_cut, c_zme)
            
        st.markdown(f"<div class='weight-box'>⚖️ Gewicht Passstück: ca. {kg} kg</div>", unsafe_allow_html=True)

    # --- SUB-TOOL: BOGEN ---
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

    # --- SUB-TOOL: STUTZEN ---
    elif "Stutzen" in tool_mode:
        st.subheader("Stutzen Schablone")
        c_st1, c_st2 = st.columns(2)
        dn_stutzen = c_st1.selectbox("DN Stutzen", df_pipe['DN'], index=6, key="stutz_dn1")
        dn_haupt = c_st2.selectbox("DN Hauptrohr", df_pipe['DN'], index=9, key="stutz_dn2")
        
        if dn_stutzen > dn_haupt:
            st.error("Fehler: Stutzen darf nicht größer als Hauptrohr sein.")
        else:
            r_k = df_pipe[df_pipe['DN'] == dn_stutzen].iloc[0]['D_Aussen'] / 2
            r_g = df_pipe[df_pipe['DN'] == dn_haupt].iloc[0]['D_Aussen'] / 2
            
            c_tab, c_plot = st.columns([1, 2])
            
            # Berechne Werte für Tabelle
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

    # --- SUB-TOOL: 3D ETAGE ---
    elif "Etage" in tool_mode:
        st.subheader("3D Etagen Berechnung & Visualisierung")
        et_type = st.radio("Typ", ["2D (Einfach)", "3D (Kastenmaß)", "3D (Fix-Winkel)"], horizontal=True, key="et_type")
        spalt_et = st.number_input("Spalt", 4, key="et_gap")
        
        c_calc, c_vis = st.columns([1, 1.5])
        weight_l = 0.0
        
        with c_vis:
            st.caption("📷 Kamera Steuerung")
            v1, v2 = st.columns(2)
            azim = v1.slider("Horizontal", 0, 360, get_val('view_azim'), key="_view_azim", on_change=save_val, args=('view_azim',))
            elev = v2.slider("Vertikal", 0, 90, get_val('view_elev'), key="_view_elev", on_change=save_val, args=('view_elev',))

        # Calculation Switches
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
        
        # Etagen-Gewicht
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
            DatabaseRepository.add("rohrbuch", (iso, naht, datum.strftime("%d.%m.%Y"), f"DN {dn_sel}", bauteil, laenge, charge, schweisser))
            st.success("Gespeichert!")
    
    df_rb = DatabaseRepository.get_all("rohrbuch")
    st.dataframe(df_rb, use_container_width=True)
    
    with st.expander("Zeile löschen"):
        if not df_rb.empty:
            opts = {f"ID {r['id']}: {r['iso']} {r['naht']}": r['id'] for i, r in df_rb.iterrows()}
            if opts:
                sel = st.selectbox("Wähle Eintrag:", list(opts.keys()), key="rb_del_sel")
                if st.button("Löschen", key="rb_del_btn"):
                    DatabaseRepository.delete("rohrbuch", opts[sel])
                    st.rerun()

# -----------------------------------------------------------------------------
# TAB 4: KALKULATION
# -----------------------------------------------------------------------------
with tab_info:
    # --- Settings ---
    with st.expander("💶 Preis-Datenbank (Einstellungen)"):
        c_io1, c_io2 = st.columns(2)
        try:
            json_data = json.dumps(st.session_state.store)
            c_io1.download_button("💾 Einstellungen speichern", data=json_data, file_name="config.json", mime="application/json")
        except:
            pass
        
        uploaded_file = c_io2.file_uploader("📂 Einstellungen laden", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                st.session_state.store.update(data)
                st.success("Geladen!")
                st.rerun()
            except:
                st.error("Fehler")
        
        st.divider()
        c_p1, c_p2 = st.columns(2)
        st.session_state.store['p_lohn'] = c_p1.number_input("Lohn (€/h)", value=get_val('p_lohn'), key="_p_lohn", on_change=save_val, args=('p_lohn',))
        st.session_state.store['p_machine'] = c_p2.number_input("Maschine (€/h)", value=get_val('p_machine'), key="_p_machine", on_change=save_val, args=('p_machine',))

    # --- Mode Switch ---
    kalk_sub_mode = st.radio("Ansicht:", ["Eingabe & Rechner", "📊 Projekt Status / Export"], horizontal=True, label_visibility="collapsed")
    st.divider()

    if kalk_sub_mode == "Eingabe & Rechner":
        calc_task = st.radio("Tätigkeit", ["🔥 Fügen", "✂️ Trennen", "🔧 Montage", "🛡️ Isolierung", "🚗 Regie"], horizontal=True, key="calc_mode")
        st.markdown("---")
        
        p_lohn = get_val('p_lohn')
        p_machine = get_val('p_machine')

        # 4.1 FÜGEN
        if "Fügen" in calc_task:
            c1, c2, c3 = st.columns(3)
            k_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('kw_dn')), key="_kw_dn", on_change=update_kw_dn)
            k_ws = c2.selectbox("WS", WS_LISTE, index=get_ws_index(get_val('kw_ws')), key="_kw_ws", on_change=save_val, args=('kw_ws',))
            k_verf = c3.selectbox("Verfahren", ["WIG", "E-Hand (CEL 70)", "MAG"], index=get_verf_index(get_val('kw_verf')), key="_kw_verf", on_change=save_val, args=('kw_verf',))
            
            c4, c5 = st.columns(2)
            if get_val('kw_dn') >= 300:
                st.info("ℹ️ Großrohr (≥ DN 300): Team=2")
            
            pers_count = c4.number_input("Pers.", value=get_val('kw_pers'), min_value=1, key="_kw_pers", on_change=save_val, args=('kw_pers',))
            anz = c5.number_input("Anzahl", value=get_val('kw_anz'), min_value=1, key="_kw_anz", on_change=save_val, args=('kw_anz',))
            
            factor = st.slider("Zeit-Faktor", 0.5, 2.0, get_val('kw_factor'), 0.1, key="_kw_factor", on_change=save_val, args=('kw_factor',))
            
            dur = (CostEngine.estimate_welding_time(k_dn, k_ws, k_verf) / pers_count) * factor
            cost = (dur/60 * (pers_count*(p_lohn+p_machine))) * anz
            
            m1, m2 = st.columns(2)
            m1.metric("Zeit Total", f"{int(dur*anz)} min")
            m2.metric("Kosten Total", f"{round(cost, 2)} €")
            
            if st.button("Hinzufügen"):
                DatabaseRepository.add("kalkulation", ("Fügen", f"DN {k_dn} {k_verf}", anz, dur*anz, cost, "-"))
                st.rerun()

        # 4.2 TRENNEN
        elif "Trennen" in calc_task:
            c1, c2 = st.columns(2)
            c_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('cut_dn')), key="_cut_dn", on_change=save_val, args=('cut_dn',))
            anz = c2.number_input("Anzahl", value=get_val('cut_anz'), min_value=1, key="_cut_anz", on_change=save_val, args=('cut_anz',))
            
            c3, c4 = st.columns(2)
            zma = c3.checkbox("ZMA", value=get_val('cut_zma'), key="_cut_zma", on_change=save_val, args=('cut_zma',))
            iso = c4.checkbox("Mantel", value=get_val('cut_iso'), key="_cut_iso", on_change=save_val, args=('cut_iso',))
            factor = st.slider("Zeit-Faktor", 0.5, 2.0, get_val('cut_factor'), 0.1, key="_cut_factor", on_change=save_val, args=('cut_factor',))
            
            # Logic
            zoll = c_dn / 25.0
            dur = (zoll * 0.5 * (3.0 if zma else 1.0) * (1.3 if iso else 1.0) * factor) * anz
            cost = (dur/60 * p_lohn) + (anz * 2.5) # Pauschale Scheiben
            
            m1, m2 = st.columns(2)
            m1.metric("Zeit", f"{int(dur)} min")
            m2.metric("Kosten", f"{round(cost, 2)} €")
            
            if st.button("Hinzufügen"):
                DatabaseRepository.add("kalkulation", ("Trennen", f"DN {c_dn}", anz, dur, cost, "Scheiben"))
                st.rerun()

        # 4.3 MONTAGE
        elif "Montage" in calc_task:
            c1, c2 = st.columns(2)
            m_dn = c1.selectbox("DN", df_pipe['DN'], index=df_pipe['DN'].tolist().index(get_val('mon_dn')), key="_mon_dn", on_change=save_val, args=('mon_dn',))
            m_anz = c2.number_input("Anzahl", value=get_val('mon_anz'), key="_mon_anz", on_change=save_val, args=('mon_anz',))
            
            factor = st.slider("Zeit-Faktor", 0.5, 2.0, get_val('mon_factor'), key="_mon_factor", on_change=save_val, args=('mon_factor',))
            
            bolts = df_pipe[df_pipe['DN'] == m_dn].iloc[0][f'Lochzahl{suffix}']
            dur = ((bolts * 2.5) + 20) * m_anz * factor
            cost = (dur/60) * (p_lohn + p_machine)
            
            m1, m2 = st.columns(2)
            m1.metric("Zeit", f"{int(dur)} min")
            m2.metric("Kosten", f"{round(cost, 2)} €")
            
            if st.button("Hinzufügen"):
                DatabaseRepository.add("kalkulation", ("Montage", f"DN {m_dn}", m_anz, dur, cost, f"{bolts*2} Schr."))
                st.rerun()

        # 4.4 REGIE
        elif "Regie" in calc_task:
            c1, c2 = st.columns(2)
            t = c1.number_input("Minuten", value=get_val('reg_min'), step=15, key="_reg_min", on_change=save_val, args=('reg_min',))
            p = c2.number_input("Personen", value=get_val('reg_pers'), min_value=1, key="_reg_pers", on_change=save_val, args=('reg_pers',))
            
            cost = (t/60 * p_lohn) * p
            st.metric("Kosten", f"{round(cost, 2)} €")
            
            if st.button("Hinzufügen", key="reg_add"):
                DatabaseRepository.add("kalkulation", ("Regie", f"{p} Pers.", 1, t, cost, "-"))
                st.rerun()

    elif kalk_sub_mode == "📊 Projekt Status / Export":
        st.header("Projekt Übersicht")
        df_k = DatabaseRepository.get_all("kalkulation")
        if not df_k.empty:
            st.dataframe(df_k, use_container_width=True)
            st.metric("Total", f"{round(df_k['kosten'].sum(), 2)} €")
            
            if st.button("Reset", type="primary"):
                DatabaseRepository.clear("kalkulation")
                st.rerun()
            
            c1, c2 = st.columns(2)
            c1.download_button("Excel", export_to_excel(df_k), "kalk.xlsx")
            if PDF_AVAILABLE:
                c2.download_button("PDF", export_to_pdf(df_k), "report.pdf")
        else:
            st.info("Leer")

# -----------------------------------------------------------------------------
# TAB 5: LAGER & LOGISTIK
# -----------------------------------------------------------------------------
with tab_lager:
    st.subheader("📦 Lagerhaltung")
    col_inv1, col_inv2 = st.columns([1, 2])
    
    # Create Article
    with col_inv1:
        st.markdown("##### Artikel anlegen")
        with st.form("new_art"):
            a_id = st.text_input("ID")
            a_name = st.text_input("Name")
            a_price = st.number_input("Preis", 0.0)
            a_stock = st.number_input("Bestand", 0)
            a_min = st.number_input("Min", 10)
            a_max = st.number_input("Max", 100)
            
            if st.form_submit_button("Speichern"):
                if a_id and a_name:
                    item = InventoryItem(a_id, a_name, a_price, a_stock, a_min, a_max)
                    if DatabaseRepository.add_inventory_item(item):
                        st.success("OK")
                    else:
                        st.error("Fehler: ID existiert")
                else:
                    st.warning("ID/Name fehlen")
    
    # Booking & Overview
    with col_inv2:
        st.markdown("##### Buchung")
        inv_items = DatabaseRepository.get_all_inventory()
        
        if inv_items:
            opts = {f"{i.article_id} {i.name}": i for i in inv_items}
            sel = st.selectbox("Artikel", list(opts.keys()))
            item = opts[sel]
            
            amt = st.number_input("Menge", 1, key="inv_amt")
            c_b1, c_b2 = st.columns(2)
            
            if c_b1.button("Eingang"):
                DatabaseRepository.update_stock(item.article_id, item.current_stock + amt)
                st.rerun()
                
            if c_b2.button("Ausgang"):
                if item.current_stock >= amt:
                    DatabaseRepository.update_stock(item.article_id, item.current_stock - amt)
                    st.rerun()
                else:
                    st.error("Bestand zu niedrig")
    
    st.divider()
    
    if inv_items:
        data = []
        total_inv_val = 0.0
        
        for i in inv_items:
            reorder = i.calculate_reorder_qty()
            status = "🟢"
            if i.current_stock <= i.reorder_point:
                status = f"🔴 ({reorder})"
            elif i.current_stock > i.target_stock:
                status = "🟠"
            
            total_inv_val += i.stock_value
            data.append({
                "ID": i.article_id, 
                "Name": i.name, 
                "Bestand": i.current_stock, 
                "Status": status, 
                "Wert": f"{i.stock_value:.2f}"
            })
            
        st.dataframe(pd.DataFrame(data), use_container_width=True)
        st.metric("Lagerwert", f"{total_inv_val:,.2f} €")
