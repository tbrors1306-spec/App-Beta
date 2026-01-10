"""
ISO Drawing Canvas utilities
Generates isometric grid backgrounds for technical drawings
"""
import numpy as np
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import math

class ISOGridGenerator:
    """Generates isometric grid backgrounds for technical drawings."""
    
    @staticmethod
    def create_iso_triangle_grid(width=1200, height=800, grid_size=20, line_color=(200, 230, 230), bg_color=(255, 255, 255)):
        """
        Creates proper isometric triangle grid (like real ISO paper).
        
        Args:
            width: Canvas width
            height: Canvas height
            grid_size: Distance between grid points
            line_color: RGB for grid lines (light cyan like ISO paper)
            bg_color: RGB for background  
        Returns:
            PIL Image
        """
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Isometric triangle grid parameters
        # 30° angles, equilateral triangles
        spacing = grid_size
        h_spacing = spacing
        v_spacing = spacing * math.sin(math.radians(60))  # Height of equilateral triangle
        
        # Draw horizontal lines
        y = 0
        while y < height + v_spacing:
            draw.line([(0, y), (width, y)], fill=line_color, width=1)
            y += v_spacing
        
        # Draw left-slanting lines (120°)
        x_start = -height
        while x_start < width:
            x1, y1 = x_start, 0
            x2 = x_start + height * math.tan(math.radians(30))
            y2 = height
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)
            x_start += h_spacing
        
        # Draw right-slanting lines (60°)
        x_start = 0
        while x_start < width + height:
            x1, y1 = x_start, 0
            x2 = x_start - height * math.tan(math.radians(30))
            y2 = height
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)
            x_start += h_spacing
        
        return img
    
    @staticmethod
    def create_iso_grid(width=1200, height=800, grid_size=20, line_color=(220, 220, 220), bg_color=(255, 255, 255)):
        """Legacy method - redirects to triangle grid"""
        return ISOGridGenerator.create_iso_triangle_grid(width, height, grid_size, line_color, bg_color)
    
    @staticmethod
    def image_to_base64(img):
        """Convert PIL Image to base64 string for Streamlit."""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
