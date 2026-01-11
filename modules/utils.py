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

class Visualizer:
    @staticmethod
    def plot_stutzen(dn_haupt, dn_stutzen, df_pipe):
        row_h = df_pipe[df_pipe['DN'] == dn_haupt].iloc[0]
        row_s = df_pipe[df_pipe['DN'] == dn_stutzen].iloc[0]
        r_main = row_h['D_Aussen'] / 2; r_stub = row_s['D_Aussen'] / 2
        
        if r_stub > r_main:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "FEHLER: Stutzen > Hauptrohr", ha='center', va='center', color='red', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.close(fig)
            return fig

        angles = range(0, 361, 5); depths = []
        for a in angles:
            try:
                term = r_stub * math.sin(math.radians(a))
                depths.append(r_main - math.sqrt(r_main**2 - term**2))
            except ValueError: depths.append(0)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(angles, depths, color='#3b82f6', lw=2)
        ax.fill_between(angles, depths, color='#eff6ff', alpha=0.5)
        ax.set_xlim(0, 360); ax.set_ylabel("Tiefe (mm)"); ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.close(fig) 
        return fig
    @staticmethod
    def plot_2d_offset(run: float, offset: float):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        x = [0, run, run*1.5] 
        y = [0, offset, offset]
        ax.plot([0, run], [0, offset], color='#dc2626', linewidth=3, label='Rohrachse') 
        ax.plot([run, run*1.5], [offset, offset], color='black', linewidth=3) 
        ax.plot([-50, 0], [0, 0], color='black', linewidth=3) 
        ax.plot([0, run], [0, 0], linestyle='--', color='gray', alpha=0.7) 
        ax.plot([run, run], [0, offset], linestyle='--', color='gray', alpha=0.7) 
        ax.text(run/2, -offset*0.1 if offset!=0 else -10, f"Länge: {run:.0f}", ha='center', color='blue')
        ax.text(run + (run*0.05), offset/2, f"H: {offset:.0f}", va='center', color='blue')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        plt.close(fig)
        return fig
    @staticmethod
    def plot_rolling_offset_3d_room(roll: float, run: float, set_val: float):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        P0 = np.array([0, 0, 0])
        P1 = np.array([roll, run, set_val])
        max_dim = max(abs(roll), abs(run), abs(set_val), 100)
        xx, yy = np.meshgrid(np.linspace(-max_dim*0.2, roll*1.2, 2), np.linspace(-max_dim*0.2, run*1.2, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
        ax.plot([0, 0], [-run*0.3, 0], [0, 0], color='gray', linewidth=4, alpha=0.6)
        ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], color='#dc2626', linewidth=5, label='Passstück')
        ax.plot([P1[0], P1[0]], [P1[1], P1[1]+run*0.3], [P1[2], P1[2]], color='gray', linewidth=4, alpha=0.6)
        ax.scatter([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], color='#1e3a8a', s=100, label='Naht/Flansch')
        ax.plot([P1[0], P1[0]], [P1[1], P1[1]], [0, P1[2]], 'b--', linewidth=1, label='Höhe (Set)')
        ax.plot([0, P1[0]], [P1[1], P1[1]], [0, 0], 'g--', linewidth=1, label='Seite (Roll)')
        ax.set_xlabel('Seite (Roll)')
        ax.set_ylabel('Länge (Run)')
        ax.set_zlabel('Höhe (Set)')
        try: 
            ax.set_box_aspect([roll if roll>10 else 100, run if run>10 else 100, set_val if set_val>10 else 100])
        except Exception: 
            pass 
        ax.legend(loc='upper left', fontsize='small')
        plt.close(fig)
        return fig
    @staticmethod
    def plot_rotation_gauge(roll: float, set_val: float, rotation_angle: float):
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
        theta = math.radians(rotation_angle)
        ax.arrow(0, 0, theta, 0.9, head_width=0.1, head_length=0.1, fc='#ef4444', ec='#ef4444', length_includes_head=True)
        ax.set_theta_zero_location("N") 
        ax.set_theta_direction(-1)      
        ax.set_rticks([])               
        ax.set_rlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Verdrehung: {rotation_angle:.1f}°", va='bottom', fontsize=10, fontweight='bold')
        ax.text(math.radians(90), 1.2, "R", ha='center', fontweight='bold')
        ax.text(math.radians(270), 1.2, "L", ha='center', fontweight='bold')
        plt.close(fig)
        return fig
    @staticmethod
    def plot_segment_schematic(mid_back: float, mid_belly: float, od: float, angle: float):
        fig, ax = plt.subplots(figsize=(6, 3))
        height = od
        top_len = mid_back
        bot_len = mid_belly
        x_top = [-top_len/2, top_len/2]
        x_bot = [-bot_len/2, bot_len/2]
        y_top = [height/2, height/2]
        y_bot = [-height/2, -height/2]
        ax.plot(x_top, y_top, 'r-', linewidth=3, label='Rücken')
        ax.plot(x_bot, y_bot, 'b-', linewidth=3, label='Bauch')
        ax.plot([x_top[0], x_bot[0]], [y_top[0], y_bot[0]], 'k--', linewidth=1)
        ax.plot([x_top[1], x_bot[1]], [y_top[1], y_bot[1]], 'k--', linewidth=1)
        ax.annotate(f"{top_len:.1f}", xy=(0, height/2 + height*0.1), ha='center', color='red', fontweight='bold')
        ax.annotate(f"{bot_len:.1f}", xy=(0, -height/2 - height*0.2), ha='center', color='blue', fontweight='bold')
        ax.set_title(f"Mittelstück ({angle:.1f}° Schnitt)", fontsize=10)
        ax.set_xlim(-top_len/2 - 50, top_len/2 + 50)
        ax.set_ylim(-height, height)
        ax.axis('off')
        plt.close(fig)
        return fig

    @staticmethod
    def plot_cutting_plan(bars):
        """
        Visualizes the cutting plan.
        bars: List of OptBar objects
        """
        if not bars: return None
        
        num_bars = len(bars)
        fig, ax = plt.subplots(figsize=(10, max(2, num_bars * 0.8)))
        
        y_pos = range(num_bars)
        bar_height = 0.6
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#6366f1']
        
        for i, bar in enumerate(bars):
            # Base bar (faint outline for stock length)
            ax.barh(i, bar.length, height=bar_height, color='#f1f5f9', edgecolor='#cbd5e1', linewidth=1)
            
            x_start = 0
            for j, cut in enumerate(bar.cuts):
                color = colors[j % len(colors)]
                ax.barh(i, cut.length, height=bar_height, left=x_start, color=color, edgecolor='white', alpha=0.9)
                
                # Label if long enough
                if cut.length > bar.length * 0.05:
                   ax.text(x_start + cut.length/2, i, f"{cut.length:.0f}", ha='center', va='center', color='white', fontsize=8, fontweight='bold')
                else: 
                   # if very small, maybe don't label or label above?
                   pass
                   
                x_start += cut.length
                
                # Simulate small gap for saw width if needed, but visually we just stack them
                # In calculation we accounted for saw width, so x_start includes it effectively if we assume visualized length is purely material?
                # Optimization logic tracked raw usage. Let's assume passed cut.length is the component length.
                # Visualization of gap is nice but tricky. Let's stick to stacking.
            
            # Label remaining waste?
            if bar.waste > 0:
                 ax.text(bar.length, i, f"Rest: {bar.waste:.1f}", ha='right', va='center', color='#94a3b8', fontsize=8, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Stange {b.id}" for b in bars])
        ax.set_xlabel("Länge (mm)")
        ax.set_xlim(0, max(b.length for b in bars) * 1.05)
        ax.invert_yaxis()  # Top to bottom
        plt.tight_layout()
        plt.close(fig)
        return fig

    @staticmethod
    def plot_rolling_offset_interactive(roll, set_val, run_length, dn):
        """Creates interactive 3D plot using Plotly with explicit dimensions"""
        if not PLOTLY_AVAILABLE:
            return None
            
        # Coordinates
        # Start: (0, 0, 0)
        # End: (Roll, Run, Set)
        
        # Calculate travel length for title
        travel = (roll**2 + set_val**2 + run_length**2)**0.5
        
        fig = go.Figure()
        
        # 1. The Pipe (Diagonal Travel) - Red Thick Line
        fig.add_trace(go.Scatter3d(
            x=[0, roll], y=[0, run_length], z=[0, set_val],
            mode='lines+markers',
            line=dict(color='#dc2626', width=10),
            marker=dict(size=8, color='red'),
            name='Rohrweg (Hypotenuse)',
            hovertemplate='<b>Rohrweg</b><br>Länge: %{text:.1f}mm<extra></extra>',
            text=[0, travel]
        ))
        
        # 2. Wireframe Box (Dimensions)
        # Path: Origin -> Roll -> Run -> Set
        
        # Roll Component (X)
        fig.add_trace(go.Scatter3d(
            x=[0, roll], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='blue', width=4, dash='solid'),
            text=['', f'Roll: {roll}mm'],
            textposition='top center',
            name='Roll (Seite)'
        ))
        
        # Run Component (Y)
        fig.add_trace(go.Scatter3d(
            x=[roll, roll], y=[0, run_length], z=[0, 0],
            mode='lines+text',
            line=dict(color='gray', width=4, dash='solid'),
            text=['', f'Run: {run_length:.1f}mm'],
            textposition='middle right',
            name='Run (Länge)'
        ))
        
        # Set Component (Z)
        fig.add_trace(go.Scatter3d(
            x=[roll, roll], y=[run_length, run_length], z=[0, set_val],
            mode='lines+text',
            line=dict(color='green', width=4, dash='solid'),
            text=['', f'Set: {set_val}mm'],
            textposition='middle left',
            name='Set (Höhe)'
        ))
        
        # Helper lines to complete the box
        # Vertical drop from end point to ground
        fig.add_trace(go.Scatter3d(
            x=[roll, roll], y=[run_length, run_length], z=[0, set_val],
            mode='lines',
            line=dict(color='lightgray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Line from Roll end to projected end on ground
        fig.add_trace(go.Scatter3d(
            x=[roll, roll], y=[0, run_length], z=[set_val, set_val],
            mode='lines',
            line=dict(color='lightgray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add floor grid
        max_dim = max(roll, run_length, set_val, 100) * 1.2
        
        fig.update_layout(
            title=dict(
                text=f'<b>3D Rolling Offset</b><br><sub>Travel: {travel:.1f}mm</sub>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='<b>Roll (Seite)</b>', range=[-10, max_dim]),
                yaxis=dict(title='<b>Run (Länge)</b>', range=[-10, max_dim]),
                zaxis=dict(title='<b>Set (Höhe)</b>', range=[-10, max_dim]),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=600,
            showlegend=True,
            legend=dict(x=0.7, y=0.1)
        )
        
        return fig

class Exporter:
    @staticmethod
    def clean_text_for_pdf(text: str) -> str:
        if not isinstance(text, str): return str(text)
        replacements = {
            "€": "EUR", "–": "-", "—": "-", "„": '"', "“": '"', "”": '"', "’": "'", "‘": "'"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    @staticmethod
    def to_excel(df):
        output = BytesIO()
        export_df = df.drop(columns=['✏️', 'Löschen', 'id', 'Auswahl', 'project_id', 'dn_clean', 'charge'], errors='ignore')
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Daten')
        return output.getvalue()

    @staticmethod
    def to_pdf_final_report(df_log, project_name, meta_data=None):
        if not PDF_AVAILABLE: return b""
        if meta_data is None: meta_data = {}
        
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, Exporter.clean_text_for_pdf("FERTIGUNGSBESCHEINIGUNG"), 0, 1, 'C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 6, "Rohrleitungsbau / Anlagenbau", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(0, 8, "1. PROJEKTDATEN", 1, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 10)
        
        def row_cell(lbl, val):
            pdf.cell(60, 8, Exporter.clean_text_for_pdf(lbl), 1)
            pdf.cell(0, 8, Exporter.clean_text_for_pdf(str(val)), 1, 1)

        row_cell("Projekt / Baustelle:", project_name)
        row_cell("Auftrags-Nr. / Ticket:", meta_data.get('order_no', '-'))
        row_cell("Anlagenteil / System:", meta_data.get('system_name', '-'))
        row_cell("Datum der Fertigstellung:", datetime.now().strftime('%d.%m.%Y'))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, Exporter.clean_text_for_pdf("2. PRÜFERGEBNISSE & QUALITÄTSSICHERUNG"), 1, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 10)
        
        rt_state = "JA / i.O." if meta_data.get('check_rt') else "Nicht gefordert"
        dim_state = "JA / i.O." if meta_data.get('check_dim') else "Nein"
        iso_state = "JA / i.O." if meta_data.get('check_iso') else "Nein"
        
        row_cell("Zerstörungsfreie Prüfung (RT):", rt_state)
        row_cell("Maßhaltigkeit geprüft:", dim_state)
        row_cell("Isometrie revidiert (As-Built):", iso_state)
        row_cell("Materialzeugnisse (APZ) vorh.:", "Siehe Anlage")
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, Exporter.clean_text_for_pdf("Hiermit wird bestätigt, dass die oben genannten Rohrleitungen fachgerecht nach den geltenden Regeln der Technik und den vorliegenden Isometrien gefertigt wurden. Alle Schweißnähte wurden, soweit gefordert, einer Röntgenprüfung (RT) unterzogen und für in Ordnung befunden."))
        pdf.ln(15)

        y_sig = pdf.get_y()
        pdf.line(10, y_sig, 200, y_sig)
        pdf.ln(2)
        
        col_w = 63
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(col_w, 5, "Ersteller / Fachfirma", 0, 0, 'C')
        pdf.cell(col_w, 5, "Bauleitung / Supervisor", 0, 0, 'C')
        pdf.cell(col_w, 5, "Abnahme / TÜV", 0, 1, 'C')
        
        pdf.ln(15) 
        
        pdf.cell(col_w, 0, "", "B") 
        pdf.cell(col_w, 0, "", "B")
        pdf.cell(col_w, 0, "", "B")
        pdf.ln(2)
        pdf.set_font("Arial", '', 7)
        pdf.cell(col_w, 4, "Datum / Unterschrift", 0, 0, 'C')
        pdf.cell(col_w, 4, "Datum / Unterschrift", 0, 0, 'C')
        pdf.cell(col_w, 4, "Datum / Unterschrift / Stempel", 0, 1, 'C')

        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "ANLAGE 1: Material-Rückverfolgbarkeit", 0, 1, 'L')
        pdf.ln(5)
        
        df_log['charge_apz'] = df_log['charge_apz'].fillna('OHNE NACHWEIS').replace('', 'OHNE NACHWEIS')
        groups = df_log.groupby('charge_apz')
        pdf.set_font("Arial", size=10)
        for apz, group in groups:
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, f"Charge / APZ: {Exporter.clean_text_for_pdf(apz)}", 1, 1, 'L', fill=True)
            pdf.set_font("Arial", size=9)
            agg = group.groupby(['dimension', 'bauteil']).size().reset_index(name='count')
            for _, row in agg.iterrows():
                txt = f"   {row['count']}x {row['bauteil']} {row['dimension']}"
                isos = group[(group['dimension']==row['dimension']) & (group['bauteil']==row['bauteil'])]['iso'].unique()
                iso_txt = ", ".join(isos[:3])
                if len(isos) > 3: iso_txt += "..."
                pdf.cell(90, 6, Exporter.clean_text_for_pdf(txt), 1)
                pdf.cell(0, 6, f"Verbaut in: {Exporter.clean_text_for_pdf(iso_txt)}", 1, 1)
            pdf.ln(2)

        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "ANLAGE 2: Detailliertes Rohrbuch", 0, 1, 'L')
        pdf.ln(5)
        
        cols = ["ISO", "Naht", "DN", "Bauteil", "Schweißer"]
        widths = [40, 30, 30, 60, 30]
        pdf.set_font("Arial", 'B', 9)
        for i, c in enumerate(cols): pdf.cell(widths[i], 8, c, 1)
        pdf.ln()
        
        pdf.set_font("Arial", size=9)
        for _, row in df_log.iterrows():
            vals = [str(row.get(k.lower(), '')) if k.lower() != 'dn' else str(row.get('dimension','')) for k in cols]
            for i, v in enumerate(vals):
                pdf.cell(widths[i], 7, Exporter.clean_text_for_pdf(v[:25]), 1)
            pdf.ln()

        return pdf.output(dest='S').encode('latin-1')

    @staticmethod
    def to_pdf_sawlist(df, project_name="Unbekannt"):
        if not PDF_AVAILABLE: return b""
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Saegeauftrag: {Exporter.clean_text_for_pdf(project_name)}", 0, 1, 'L')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 5, f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}", 0, 1, 'L')
        pdf.ln(5)
        cols = ["Bezeichnung", "Rohmass", "Saegemass", "Info", "Zeit"]
        keys = ["name", "raw_length", "cut_length", "details", "timestamp"]
        widths = [60, 30, 30, 80, 30]
        pdf.set_font("Arial", 'B', 10)
        for i, c in enumerate(cols): pdf.cell(widths[i], 8, c, 1)
        pdf.ln()
        pdf.set_font("Arial", size=10)
        for _, row in df.iterrows():
            for i, k in enumerate(keys):
                val = str(row.get(k, ''))
                if isinstance(row.get(k), float): val = f"{row.get(k):.1f}"
                try: pdf.cell(widths[i], 8, Exporter.clean_text_for_pdf(val), 1)
                except (UnicodeEncodeError, Exception): pdf.cell(widths[i], 8, "?", 1)
            pdf.ln()
        return pdf.output(dest='S').encode('latin-1')
