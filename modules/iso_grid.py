"""
ISO Drawing Canvas utilities
Generates isometric grid backgrounds for technical drawings
"""
import numpy as np
from PIL import Image, ImageDraw
import base64
from io import BytesIO

class ISOGridGenerator:
    """Generates isometric grid backgrounds for technical drawings."""
    
    @staticmethod
    def create_iso_grid(width=1200, height=800, grid_size=20, line_color=(220, 220, 220), bg_color=(255, 255, 255)):
        """
        Creates isometric grid background image.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            grid_size: Distance between grid points
            line_color: RGB tuple for grid lines
            bg_color: RGB tuple for background
            
        Returns:
            PIL Image object
        """
        # Create blank image
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw vertical lines (for reference)
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=line_color, width=1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=line_color, width=1)
        
        # Draw 30° isometric lines (left-to-right, going up)
        # tan(30°) = 0.577
        angle_30_slope = 0.577
        
        # Lines starting from left edge, going right-up
        for y_start in range(0, height + width, grid_size):
            x1, y1 = 0, y_start
            x2 = width
            y2 = y_start - int(width * angle_30_slope)
            if y2 < -width:
                continue
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)
        
        # Draw 150° isometric lines (left-to-right, going down)
        # This is mirror of 30°
        for y_start in range(-width, height, grid_size):
            x1, y1 = 0, y_start
            x2 = width
            y2 = y_start + int(width * angle_30_slope)
            if y2 > height + width:
                continue
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)
        
        return img
    
    @staticmethod
    def image_to_base64(img):
        """Convert PIL Image to base64 string for Streamlit."""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
