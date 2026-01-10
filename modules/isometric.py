import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# Optional Imports
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PDF_AVAILABLE = False
    class FPDF: pass # Dummy

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PLOTLY_AVAILABLE = False

# Import existing Visualizer and Exporter content from current file first...
# (This is a simplified approach - in reality we'd read the existing file)

class IsometricDrawer:
    """Generates isometric pipe drawings from 3D waypoints."""
    
    @staticmethod
    def _to_iso(x, y, z):
        """Convert 3D coordinates to 2D isometric projection."""
        iso_x = (x - z) * math.cos(math.radians(30))
        iso_y = y + (x + z) * math.sin(math.radians(30))
        return iso_x, iso_y
    
    @staticmethod
    def draw_iso_route(waypoints, pipe_dn=100, show_dimensions=True):
        """
        Creates isometric drawing from waypoints.
        waypoints: [{"x": float, "y": float, "z": float}, ...]
        """
        if len(waypoints) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect('equal')
        
        # Convert all waypoints to iso coords
        iso_points = []
        for wp in waypoints:
            x, y, z = wp.get("x", 0), wp.get("y", 0), wp.get("z", 0)
            iso_x, iso_y = IsometricDrawer._to_iso(x, y, z)
            iso_points.append((iso_x, iso_y, x, y, z))
        
        # Draw pipe segments
        pipe_width = max(2, pipe_dn / 50)
        
        for i in range(len(iso_points) - 1):
            p1 = iso_points[i]
            p2 = iso_points[i + 1]
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color='#2563eb', linewidth=pipe_width, solid_capstyle='round')
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color='#1e40af', linewidth=pipe_width + 2, alpha=0.3, solid_capstyle='round')
        
        # Draw waypoint markers
        for i, (iso_x, iso_y, x, y, z) in enumerate(iso_points):
            ax.plot(iso_x, iso_y, 'o', color='#dc2626', markersize=8, zorder=10)
            ax.text(iso_x, iso_y + 50, f'P{i+1}', ha='center', fontsize=10, fontweight='bold')
        
        # Add dimensions
        if show_dimensions:
            for i in range(len(iso_points) - 1):
                p1 = iso_points[i]
                p2 = iso_points[i + 1]
                
                dx = p2[2] - p1[2]
                dy = p2[3] - p1[3]
                dz = p2[4] - p1[4]
                distance = (dx**2 + dy**2 + dz**2)**0.5
                
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                
                ax.text(mid_x, mid_y - 30, f'{distance:.0f}mm', 
                       ha='center', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel('X / Z Achse', fontsize=10)
        ax.set_ylabel('Y Achse', fontsize=10)
        ax.set_title('Isometrische Rohr-Zeichnung', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.close(fig)
        return fig
