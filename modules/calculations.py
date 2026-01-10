import math
import re
import pandas as pd
from typing import Dict, List

class PipeCalculator:
    PN_MAP = {
        "PN 16": "_16",
        "PN 10": "_10",
        "PN 6": "_10",
        "PN 25": "_16", 
        "PN 40": "_16" 
    }

    def __init__(self, df: pd.DataFrame): self.df = df
    
    def get_row(self, dn: int) -> pd.Series:
        row = self.df[self.df['DN'] == dn]
        return row.iloc[0] if not row.empty else self.df.iloc[0]
        
    def get_deduction(self, f_type: str, dn: int, pn: str, angle: float = 90.0) -> float:
        row = self.get_row(dn)
        suffix = self.PN_MAP.get(pn, "_10")
        
        if "Bogen 90°" in f_type: return float(row['Radius_BA3'])
        if "Zuschnitt" in f_type: return float(row['Radius_BA3']) * math.tan(math.radians(angle / 2))
        if "Flansch" in f_type: return float(row[f'Flansch_b{suffix}'])
        if "T-Stück" in f_type: return float(row['T_Stueck_H'])
        if "Reduzierung" in f_type: return float(row['Red_Laenge_L'])
        return 0.0
        
    def calculate_bend_details(self, dn: int, angle: float) -> Dict[str, float]:
        row = self.get_row(dn)
        r = float(row['Radius_BA3'])
        da = float(row['D_Aussen'])
        rad = math.radians(angle)
        return {"vorbau": r * math.tan(rad / 2), "bogen_aussen": (r + da/2) * rad, "bogen_mitte": r * rad, "bogen_innen": (r - da/2) * rad}
        
    def calculate_stutzen_coords(self, dn_haupt: int, dn_stutzen: int) -> pd.DataFrame:
        r_main = self.get_row(dn_haupt)['D_Aussen'] / 2
        r_stub = self.get_row(dn_stutzen)['D_Aussen'] / 2
        if r_stub > r_main: raise ValueError("Stutzen > Hauptrohr")
        table_data = []
        for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]:
            try:
                term = r_stub * math.sin(math.radians(angle))
                t_val = r_main - math.sqrt(r_main**2 - term**2)
            except ValueError: t_val = 0
            u_val = (r_stub * 2 * math.pi) * (angle / 360)
            table_data.append({"Winkel": f"{angle}°", "Tiefe (mm)": round(t_val, 1), "Umfang (mm)": round(u_val, 1)})
        return pd.DataFrame(table_data)
        
    def calculate_2d_offset(self, dn: int, offset: float, angle: float) -> Dict[str, float]:
        row = self.get_row(dn)
        r = float(row['Radius_BA3'])
        rad = math.radians(angle)
        try:
            hypotenuse = offset / math.sin(rad)
            run = offset / math.tan(rad)
        except ZeroDivisionError: return {"error": "Winkel 0"}
        z_mass = r * math.tan(rad / 2)
        return {"hypotenuse": hypotenuse, "run": run, "z_mass_single": z_mass, "cut_length": hypotenuse - (2*z_mass), "offset": offset, "angle": angle}
        
    def calculate_rolling_offset(self, dn: int, roll: float, set_val: float, height: float = 0.0) -> Dict[str, float]:
        diag_base = math.sqrt(roll**2 + set_val**2)
        travel = math.sqrt(diag_base**2 + height**2)
        if travel != 0 and -1 <= (diag_base / travel) <= 1:
            required_angle = math.degrees(math.acos(diag_base / travel))
        else:
            required_angle = 0
        return {"diag_base": diag_base, "travel": travel, "angle_calc": required_angle, 
                "run_length": diag_base, "set": set_val, "roll": roll}
        
    def calculate_segment_bend(self, dn: int, radius: float, num_segments: int, total_angle: float = 90.0) -> Dict[str, float]:
        row = self.get_row(dn)
        od = float(row['D_Aussen'])
        if num_segments < 2: return {"error": "Min. 2 Segmente"}
        miter_angle = total_angle / (2 * (num_segments - 1))
        tan_alpha = math.tan(math.radians(miter_angle))
        len_center = 2 * radius * tan_alpha
        len_back = 2 * (radius + od/2) * tan_alpha
        len_belly = 2 * (radius - od/2) * tan_alpha
        end_back = (radius + od/2) * tan_alpha
        end_belly = (radius - od/2) * tan_alpha
        end_center = radius * tan_alpha
        return {"miter_angle": miter_angle, "mid_back": len_back, "mid_belly": len_belly, "mid_center": len_center, "end_back": end_back, "end_belly": end_belly, "end_center": end_center, "od": od}

    @staticmethod
    def apply_tolerance_stack(cut_length: float, num_welds: int, shrinkage_per_weld: float = 2.0) -> dict:
        """
        Adjusts cut length to compensate for weld shrinkage.
        """
        total_compensation = num_welds * shrinkage_per_weld
        adjusted_length = cut_length + total_compensation
        
        return {
            "original": cut_length,
            "adjusted": adjusted_length,
            "compensation": total_compensation,
            "num_welds": num_welds,
            "shrinkage_per_weld": shrinkage_per_weld
        }

    def calculate_multi_level_offset(self, waypoints: list) -> dict:
        """
        Calculates chained rolling offsets through multiple waypoints.
        """
        if len(waypoints) < 2:
            return {"error": "Mindestens 2 Wegpunkte benötigt"}
        
        segments = []
        total_travel = 0
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            delta_roll = end["roll"] - start["roll"]
            delta_set = end["set"] - start["set"]
            
            # Calculate this segment using simple math (no dn needed for basic offset)
            diag_base = (delta_roll**2 + delta_set**2)**0.5
            
            # Determine angle
            if diag_base != 0:
                angle = abs(delta_roll / diag_base) * 90 if abs(delta_roll) > abs(delta_set) else abs(delta_set / diag_base) * 90
            else:
                angle = 0
            
            segments.append({
                "segment": i + 1,
                "from": f"P{i+1}",
                "to": f"P{i+2}",
                "roll": delta_roll,
                "set": delta_set,
                "travel": diag_base,
                "angle": angle
            })
            
            total_travel += diag_base
        
        return {
            "segments": segments,
            "total_travel": total_travel,
            "num_segments": len(segments)
        }

class MaterialManager:
    @staticmethod
    def parse_dn(dim_str: str) -> int:
        if not dim_str: return 0
        try:
            match = re.search(r'\d+', str(dim_str))
            if match: return int(match.group())
            return 0
        except (KeyError, IndexError, ValueError): return 0
    @staticmethod
    def generate_mto(df_log: pd.DataFrame) -> pd.DataFrame:
        if df_log.empty: return pd.DataFrame()
        df = df_log.copy()
        df['dn_clean'] = df['dimension'].apply(MaterialManager.parse_dn)
        linear_items = ['Rohrstoß', 'Passstück', 'Rohr']
        df_linear = df[df['bauteil'].isin(linear_items)].copy()
        if not df_linear.empty:
            df_linear['menge'] = pd.to_numeric(df_linear['laenge'], errors='coerce').fillna(0) / 1000.0
            mto_linear = df_linear.groupby(['dn_clean', 'bauteil'])['menge'].sum().reset_index()
            mto_linear['Einheit'] = 'm'
        else:
            mto_linear = pd.DataFrame(columns=['dn_clean', 'bauteil', 'menge', 'Einheit'])
        df_count = df[~df['bauteil'].isin(linear_items)].copy()
        if not df_count.empty:
            mto_count = df_count.groupby(['dn_clean', 'bauteil']).size().reset_index(name='menge')
            mto_count['Einheit'] = 'Stk'
        else:
            mto_count = pd.DataFrame(columns=['dn_clean', 'bauteil', 'menge', 'Einheit'])
        mto_final = pd.concat([mto_linear, mto_count], ignore_index=True)
        mto_final['Dimension'] = mto_final['dn_clean'].apply(lambda x: f"DN {x}")
        mto_final = mto_final.rename(columns={'bauteil': 'Beschreibung', 'menge': 'Menge'})
        mto_final = mto_final[['Dimension', 'Beschreibung', 'Menge', 'Einheit']].sort_values(['Dimension', 'Beschreibung'])
        return mto_final

class HandbookCalculator:
    BOLT_DATA = {"M12": [19, 85, 55], "M16": [24, 210, 135], "M20": [30, 410, 265], "M24": [36, 710, 460], "M27": [41, 1050, 680], "M30": [46, 1420, 920], "M33": [50, 1930, 1250], "M36": [55, 2480, 1600], "M39": [60, 3200, 2080], "M45": [70, 5000, 3250], "M52": [80, 7700, 5000]}
    @staticmethod
    def calculate_weight(od, wall, length):
        if wall <= 0: return {"steel": 0, "water": 0, "total": 0}
        id_mm = od - (2*wall)
        vol_s = (math.pi*(od**2 - id_mm**2)/4)/1000000
        vol_w = (math.pi*(id_mm**2)/4)/1000000
        return {"kg_per_m_steel": vol_s*7850, "total_steel": vol_s*7850*(length/1000), "total_filled": (vol_s*7850 + vol_w*1000)*(length/1000), "volume_l": vol_w*(length/1000)*1000}
    @staticmethod
    def get_bolt_length(t1, t2, bolt, washers=2, gasket=2.0):
        try:
            d = int(bolt.replace("M", ""))
            l = t1 + t2 + gasket + (washers*4) + (d*0.8) + max(6, d*0.4)
            rem = l % 5
            return int(l + (5-rem) if rem != 0 else l)
        except (KeyError, IndexError, ValueError): return 0
