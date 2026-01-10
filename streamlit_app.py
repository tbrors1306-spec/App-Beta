import streamlit as st
import pandas as pd
import json
import logging
import html
import time
from dataclasses import asdict
from datetime import datetime

from modules.database import DatabaseRepository, DB_NAME
from modules.models import FittingItem, SavedCut
from modules.calculations import PipeCalculator, MaterialManager, HandbookCalculator
from modules.utils import Visualizer, Exporter, PDF_AVAILABLE
from modules.optimization import CuttingOptimizer, CutRequest
from modules.isometric import IsometricDrawer
from modules.ui import init_app_state, render_smart_input, render_sidebar_projects

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipeCraft_V3_5_Final")

st.set_page_config(
    page_title="PipeCraft v3.5",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; background-color: #f8fafc; }
    h1, h2, h3, h4, h5 { font-family: 'Segoe UI', sans-serif; font-weight: 600; color: #1e293b; }
    div.row-widget.stRadio > div { flex-direction: row; align-items: stretch; }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] { 
        background-color: #ffffff; border: 1px solid #e2e8f0; padding: 10px 20px; border-radius: 8px; margin-right: 10px;
    }
    .machine-header-saw { border-bottom: 4px solid #f97316; color: #f97316; padding: 5px 0; font-weight: 700; font-size: 1.2rem; margin-bottom: 15px; text-transform: uppercase; }
    .machine-header-geo { border-bottom: 4px solid #0ea5e9; color: #0ea5e9; padding: 5px 0; font-weight: 700; font-size: 1.2rem; margin-bottom: 15px; text-transform: uppercase; }
    .machine-header-doc { border-bottom: 4px solid #64748b; color: #64748b; padding: 5px 0; font-weight: 700; font-size: 1.2rem; margin-bottom: 15px; text-transform: uppercase; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; }
    div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px; }
    .project-tag { font-family: 'Segoe UI', sans-serif; font-weight: 600; color: #475569; padding: 8px 12px; background-color: #e2e8f0; border-radius: 6px; margin-bottom: 20px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_pipe_data():
    with open('data/pipe_dimensions.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def render_smart_saw(calc: PipeCalculator, df: pd.DataFrame, current_dn: int, pn: str):
    st.markdown('<div class="machine-header-saw">🪚 SMARTE SÄGE</div>', unsafe_allow_html=True)
    
    proj_name = st.session_state.get('active_project_name', 'Unbekannt')
    safe_proj_name = html.escape(proj_name)
    active_pid = st.session_state.get('active_project_id', 1)
    is_archived = st.session_state.get('project_archived', 0)

    st.markdown(f"<div class='project-tag'>📍 PROJEKT: {safe_proj_name}</div>", unsafe_allow_html=True)

    if is_archived:
        st.info("Projekt ist abgeschlossen. Keine neuen Schnitte möglich.")
        return

    # Init saved cuts clean up
    if st.session_state.saved_cuts:
        try: _ = st.session_state.saved_cuts[0].fittings
        except AttributeError: st.session_state.saved_cuts = []

    # Transfer logic: Update the widget state directly if a transfer exists
    if 'transfer_cut_length' in st.session_state:
        new_val = st.session_state.pop('transfer_cut_length')
        st.session_state['saw_raw_input'] = float(new_val)
        st.toast("✅ Maß aus Geometrie übernommen!", icon="📏")
        
    c_calc, c_list = st.columns([1.3, 1.7])

    # --- LINKER BEREICH: RECHNER ---
    with c_calc:
        with st.container(border=True):
            
            # 1. DAS EINGABE-FORMULAR (Ganz oben)
            st.markdown("**1. Schnitt & Bauteile**")
            with st.form(key="combined_saw_form"):
                cut_name = st.text_input("Bezeichnung / Spool", placeholder="z.B. Strang A - 01")
                # Removed 'value=default_raw' and rely on key state or default 0.0
                raw_len = st.number_input("Schnittmaß (Roh) [mm]", min_value=0.0, step=10.0, format="%.1f", key="saw_raw_input")
                
                cg1, cg2, cg3 = st.columns(3)
                gap = cg1.number_input("Spalt (mm)", value=3.0, step=0.5)
                dicht_anz = cg2.number_input("Dichtungen", 0, 5, 0)
                dicht_thk = cg3.number_input("Dicke", 0.0, 5.0, 2.0)

                st.markdown("---")
                st.caption("Optional: Fitting hinzufügen")
                
                cf1, cf2 = st.columns([1.5, 1])
                f_type = cf1.selectbox("Typ", ["Bogen 90° (BA3)", "Bogen (Zuschnitt)", "Flansch (Vorschweiß)", "T-Stück", "Reduzierung"], label_visibility="collapsed")
                
                try: 
                    default_dn_idx = df['DN'].tolist().index(current_dn)
                except ValueError: 
                    default_dn_idx = 0
                f_dn = cf2.selectbox("DN", df['DN'], index=default_dn_idx, label_visibility="collapsed")
                
                cf3, cf4 = st.columns([1, 1])
                f_cnt = cf3.number_input("Anzahl", 1, 10, 1)
                f_ang = 90.0
                if "Zuschnitt" in f_type: 
                    f_ang = cf4.slider("Winkel", 0, 90, 45)
                else:
                    cf4.markdown("") # Spacer

                st.markdown("<br>", unsafe_allow_html=True)
                
                col_btn_add, col_btn_calc = st.columns(2)
                
                # Button A: Fügt Bauteil hinzu UND berechnet
                submitted_add = col_btn_add.form_submit_button("➕ Bauteil dazu", type="secondary", use_container_width=True)
                
                # Button B: Nur Berechnen
                submitted_calc = col_btn_calc.form_submit_button("🔄 Berechnen", type="primary", use_container_width=True)

            # --- LOGIK NACH DEM FORMULAR-SUBMIT ---
            
            # Fall A: Bauteil hinzufügen
            if submitted_add:
                deduct = calc.get_deduction(f_type, f_dn, pn, f_ang)
                uid = f"{len(st.session_state.fitting_list)}_{datetime.now().timestamp()}"
                nm = f"{f_type} DN{f_dn}" + (f" ({f_ang}°)" if "Zuschnitt" in f_type else "")
                st.session_state.fitting_list.append(FittingItem(uid, nm, f_cnt, deduct, f_dn))
                st.toast(f"✅ {nm} hinzugefügt!", icon="➕")

            # Fall B oder A: Berechnen
            if submitted_add or submitted_calc:
                sum_fit = sum(i.total_deduction for i in st.session_state.fitting_list)
                sum_gap = sum(i.count for i in st.session_state.fitting_list) * gap
                sum_gskt = dicht_anz * dicht_thk
                total = sum_fit + sum_gap + sum_gskt
                final = raw_len - total
                
                st.session_state.last_calc_result = {
                    "final": final, "raw": raw_len, "total_deduct": total,
                    "info": f"Teile -{sum_fit:.1f} | Spalte -{sum_gap:.1f} | Dicht. -{sum_gskt:.1f}"
                }

            # 2. LISTE DER BEREITS GEWÄHLTEN BAUTEILE (JETZT HIER UNTERHALB)
            if st.session_state.fitting_list:
                st.divider()
                st.markdown("###### 🛒 Enthaltene Teile:")
                for i, item in enumerate(st.session_state.fitting_list):
                    with st.container():
                        cr1, cr2, cr3 = st.columns([3, 1.5, 0.5])
                        cr1.text(f"{item.count}x {item.name}")
                        cr2.text(f"-{item.total_deduction:.1f}")
                        if cr3.button("🗑️", key=f"d_{item.id}", help="Entfernen"):
                            st.session_state.fitting_list.pop(i)
                            st.rerun()
                
                if st.button("Alle Teile entfernen", type="secondary", key="clear_fits"):
                    st.session_state.fitting_list = []
                    st.rerun()

            # 3. ERGEBNIS & SPEICHERN
            if 'last_calc_result' in st.session_state:
                res = st.session_state.last_calc_result
                st.divider()
                
                if res['final'] < 0:
                    st.error(f"⚠️ Negativmaß! ({res['final']:.1f} mm)")
                else:
                    st.metric("Sägelänge (Z)", f"{res['final']:.1f} mm")
                    st.caption(res['info'])
                    
                    # Tolerance Stack Calculator
                    with st.expander("⚠️ Schweißnaht-Schrumpfung berücksichtigen", expanded=False):
                        st.caption("Kompensiert die Schrumpfung durch Schweißnähte (typisch: 1-3mm pro Naht)")
                        tc1, tc2 = st.columns(2)
                        num_welds = tc1.number_input("Anzahl Nähte", min_value=1, max_value=10, value=2, step=1)
                        shrinkage = tc2.number_input("Schrumpfung/Naht (mm)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
                        
                        tol_result = calc.apply_tolerance_stack(res['final'], num_welds, shrinkage)
                        
                        st.divider()
                        tm1, tm2 = st.columns(2)
                        tm1.metric("Original", f"{tol_result['original']:.1f} mm")
                        tm2.metric("Korrigiert", f"{tol_result['adjusted']:.1f} mm", delta=f"+{tol_result['compensation']:.1f} mm")
                        st.caption(f"📏 Für {tol_result['num_welds']} Nähte à {tol_result['shrinkage_per_weld']}mm")
                    
                    if st.button("💾 IN LISTE SPEICHERN", type="primary", use_container_width=True):
                        final_name = cut_name if cut_name.strip() else f"Schnitt"
                        current_fittings_copy = list(st.session_state.fitting_list)
                        new_id = int(time.time() * 1000)
                        
                        new_cut = SavedCut(new_id, final_name, res['raw'], res['final'], 
                                         f"{len(current_fittings_copy)} Teile", 
                                         datetime.now().strftime("%H:%M"), 
                                         current_fittings_copy)
                        
                        st.session_state.saved_cuts.append(new_cut)
                        st.session_state.fitting_list = [] 
                        del st.session_state.last_calc_result
                        
                        st.toast("✅ Schnitt gespeichert!", icon="💾")
                        time.sleep(0.5)
                        st.rerun()

    # --- RECHTER BEREICH: LISTE ---
    with c_list:
        st.markdown("#### 📋 Schnittliste")
        action_bar = st.container()

        if not st.session_state.saved_cuts:
            st.info("Noch keine Schnitte vorhanden.")
            with action_bar:
                st.button("🗑️ Löschen", disabled=True, use_container_width=True)
        else:
            data = [asdict(c) for c in st.session_state.saved_cuts]
            df_s = pd.DataFrame(data)
            if 'Auswahl' not in df_s.columns: df_s['Auswahl'] = False
            
            df_display = df_s[['Auswahl', 'name', 'raw_length', 'cut_length', 'details', 'id']]
            
            edited_df = st.data_editor(
                df_display, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Auswahl": st.column_config.CheckboxColumn("☑️", width="small", default=False),
                    "name": st.column_config.TextColumn("Bez.", width="medium"), 
                    "raw_length": st.column_config.NumberColumn("Roh", format="%.0f"), 
                    "cut_length": st.column_config.NumberColumn("Säge", format="%.1f", width="medium"), 
                    "details": st.column_config.TextColumn("Info", width="small"), 
                    "id": None
                },
                disabled=["name", "raw_length", "cut_length", "details", "id"], 
                key="saw_editor_v4"
            )
            
            selected_rows = edited_df[edited_df['Auswahl'] == True]
            selected_ids = selected_rows['id'].tolist()
            num_sel = len(selected_ids)
            
            with action_bar:
                btns_disabled = (num_sel == 0)
                col_del, col_trans, col_excel = st.columns([1, 1, 1])
                
                if col_del.button(f"🗑️ Löschen ({num_sel})", disabled=btns_disabled, type="secondary", use_container_width=True):
                    st.session_state.saved_cuts = [c for c in st.session_state.saved_cuts if c.id not in selected_ids]
                    st.toast(f"🗑️ {num_sel} Einträge gelöscht!", icon="🗑️")
                    time.sleep(0.5)
                    st.rerun()
                
                if col_trans.button(f"📝 Übertragen ({num_sel})", disabled=btns_disabled, type="primary", use_container_width=True):
                    count_pipes = 0
                    count_fits = 0
                    for cut in st.session_state.saved_cuts:
                        if cut.id in selected_ids:
                            DatabaseRepository.add_entry({
                                "iso": cut.name, "naht": "", "datum": datetime.now().strftime("%d.%m.%Y"),
                                "dimension": f"DN {current_dn}", "bauteil": "Rohrstoß", "laenge": cut.cut_length,
                                "charge": "", "charge_apz": "", "schweisser": "", "project_id": active_pid
                            })
                            count_pipes += 1
                            for fit in cut.fittings:
                                fit_name_clean = fit.name.split(" DN")[0]
                                for _ in range(fit.count):
                                    DatabaseRepository.add_entry({
                                        "iso": cut.name, "naht": "", "datum": datetime.now().strftime("%d.%m.%Y"),
                                        "dimension": f"DN {fit.dn}", "bauteil": fit_name_clean, "laenge": 0.0,
                                        "charge": "", "charge_apz": "", "schweisser": "", "project_id": active_pid
                                    })
                                    count_fits += 1
                    
                    st.toast(f"✅ {count_pipes} Rohre und {count_fits} Fittings übertragen!", icon="📦")
                    time.sleep(0.5)

                fname_base = f"Saege_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
                excel_data = Exporter.to_excel(df_s)
                col_excel.download_button("📥 Excel (Alle)", excel_data, f"{fname_base}.xlsx", use_container_width=True)

            # --- OPTIMIZER BLOCK ---
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            with st.expander("✂️ Schnitt-Optimierung (Verschnitt-Minimierung)", expanded=False):
                st.caption("Berechnet die optimale Aufteilung der gewählten Schnitte auf Stangen.")
                
                c_opt1, c_opt2 = st.columns(2)
                stock_len = c_opt1.number_input("Stangenlänge (mm)", value=6000.0, step=500.0)
                saw_width = c_opt2.number_input("Sägeblatt (mm)", value=3.0, step=0.5)
                
                if st.button("🚀 Optimierung starten", disabled=btns_disabled, use_container_width=True):
                    # Gather cuts
                    requests = []
                    for cut in st.session_state.saved_cuts:
                        if cut.id in selected_ids: 
                            requests.append(CutRequest(id=cut.name, length=cut.cut_length))
                    
                    if not requests:
                        st.error("Bitte Schnitte auswählen!")
                    else:
                        result_bars = CuttingOptimizer.solve_ffd(requests, stock_len, saw_width)
                        st.session_state.opt_results = result_bars
                        st.toast("Optimierung fertig!")

                if 'opt_results' in st.session_state and st.session_state.opt_results:
                    bars = st.session_state.opt_results
                    total_waste = sum(b.waste for b in bars)
                    
                    st.divider()
                    st.markdown("##### Ergebnis:")
                    m1, m2 = st.columns(2)
                    m1.metric("Benötigte Stangen", f"{len(bars)} Stk")
                    m2.metric("Gesamtabfall", f"{total_waste/1000:.2f} m")
                    
                    fig_opt = Visualizer.plot_cutting_plan(bars)
                    if fig_opt:
                        st.pyplot(fig_opt, use_container_width=True)
                    
                    with st.expander("Detailliste"):
                        for b in bars:
                            st.markdown(f"**Stange {b.id}** (Rest: {b.waste:.1f}mm)")
                            txts = [f"{c.length:.0f}" for c in b.cuts]
                            st.caption(" | ".join(txts))

            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            if st.button("Alles Reset (Liste leeren)", type="secondary"):
                st.session_state.saved_cuts = []
                st.rerun()

def render_geometry_tools(calc: PipeCalculator, df: pd.DataFrame):
    st.markdown('<div class="machine-header-geo">📐 GEOMETRIE & BERECHNUNG</div>', unsafe_allow_html=True)
    geo_tabs = st.tabs(["2D Etage (S-Schlag)", "3D Raum-Etage (Rolling)", "Bogen (Standard)", "🦞 Segment-Bogen", "Stutzen"])
    
    with geo_tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            with st.container(border=True):
                with st.form(key="geo_2d_form"):
                    dn = st.selectbox("Nennweite", df['DN'], index=5, key="2d_dn")
                    offset = st.number_input("Versprung (H) [mm]", value=500.0, step=10.0, key="2d_off")
                    angle = st.selectbox("Fittings (°)", [30, 45, 60], index=1, key="2d_ang")
                    submit_2d = st.form_submit_button("Berechnen 🚀", type="primary", use_container_width=True)
                
                if submit_2d:
                    res = calc.calculate_2d_offset(dn, offset, angle)
                    st.session_state.calc_res_2d = res 
        
        with c2:
            if 'calc_res_2d' in st.session_state:
                res = st.session_state.calc_res_2d
                if "error" in res: st.error(res["error"])
                else:
                    st.markdown("**Ergebnis**")
                    m1, m2 = st.columns(2)
                    m1.metric("Zuschnitt (Rohr)", f"{res['cut_length']:.1f} mm")
                    m2.metric("Etagenlänge", f"{res['hypotenuse']:.1f} mm")
                    st.info(f"Benötigter Platz (Länge): {res['run']:.1f} mm")
                    
                    if st.button("➡️ An Säge (2D)", key="btn_2d_saw"):
                        st.session_state.transfer_cut_length = res['cut_length']
                        st.toast("✅ Maß an Säge übertragen!", icon="📏")
                    
                    fig_2d = Visualizer.plot_2d_offset(res['run'], res['offset'])
                    st.pyplot(fig_2d, use_container_width=False)

    with geo_tabs[1]:
        col_in, col_out, col_vis = st.columns([1, 1, 1.5]) 
        with col_in:
            with st.container(border=True):
                st.markdown("**Eingabe**")
                with st.form(key="geo_3d_form"):
                    dn_roll = st.selectbox("Nennweite", df['DN'], index=5, key="3d_dn")
                    fit_angle = st.selectbox("Fitting Typ", [45, 60, 90], index=0, key="3d_ang")
                    set_val = st.number_input("Versprung Höhe (Set)", value=300.0, min_value=0.0, step=10.0)
                    roll_val = st.number_input("Versprung Seite (Roll)", value=400.0, min_value=0.0, step=10.0)
                    submit_3d = st.form_submit_button("Berechnen 🚀", type="primary", use_container_width=True)
                
                if submit_3d:
                    true_offset = (set_val**2 + roll_val**2)**0.5
                    rad_angle = (fit_angle * 3.14159) / 180
                    if rad_angle > 0:
                        travel_center = true_offset / (true_offset/true_offset if true_offset==0 else 1) # simple fallback
                        # Re-using calc logic properly
                        res_roll = calc.calculate_rolling_offset(dn_roll, roll_val, set_val)
                        travel_center = res_roll['travel']
                        # need deduction
                        deduct_single = calc.get_deduction(f"Bogen (Zuschnitt)", dn_roll, "PN 16", fit_angle) 
                        cut_len = travel_center - (2 * deduct_single)
                        run_length = (travel_center**2 - true_offset**2)**0.5 if travel_center > true_offset else 0
                        rot_angle = res_roll['angle_calc']
                    
                    st.session_state.calc_res_3d = {
                        "cut_len": cut_len, "travel_center": travel_center, 
                        "run_length": run_length, "rot_angle": rot_angle,
                        "roll_val": roll_val, "set_val": set_val
                    }

        with col_out:
            if 'calc_res_3d' in st.session_state:
                res = st.session_state.calc_res_3d
                st.markdown("**Ergebnis**")
                st.metric("Zuschnitt (Rohr)", f"{res['cut_len']:.1f} mm")
                st.caption(f"Einbaumaß (Mitte-Mitte): {res['travel_center']:.1f} mm")
                st.metric("Verdrehung", f"{res['rot_angle']:.1f} °", "aus der Senkrechten")
                
                if res['cut_len'] < 0: st.error("Nicht baubar! Fittings stoßen zusammen.")
                else:
                    if st.button("➡️ An Säge (3D)", key="btn_3d_saw"):
                        st.session_state.transfer_cut_length = res['cut_len']
                        st.toast("✅ Maß an Säge übertragen!", icon="📏")

        with col_vis:
            if 'calc_res_3d' in st.session_state:
                res = st.session_state.calc_res_3d
                st.markdown("**3D Simulation**")
                fig_3d = Visualizer.plot_rolling_offset_interactive(res['roll_val'], res['run_length'], res['set_val'])
                if fig_3d:
                     st.plotly_chart(fig_3d, use_container_width=True)
                else:
                     st.warning("Installiere `plotly` um 3D-Ansicht zu sehen.")
                     fig_static = Visualizer.plot_rolling_offset_3d_room(res['roll_val'], 0, res['set_val'])
                     st.pyplot(fig_static, use_container_width=False)
                with st.expander("Verdrehung (Wasserwaage)"):
                    fig_gauge = Visualizer.plot_rotation_gauge(res['roll_val'], res['set_val'], res['rot_angle'])
                    st.pyplot(fig_gauge, use_container_width=False)

        # NEW: Multi-Level Rolling Offset Sub-Section
        st.divider()
        st.markdown("#### 🔗 Multi-Level Rolling Offset (Kette)")
        st.caption("Berechne verkettete Offsets durch mehrere Wegpunkte (A→B→C→D)")
        
        if 'ml_waypoints' not in st.session_state:
            st.session_state.ml_waypoints = [{"roll": 0.0, "set": 0.0}, {"roll": 400.0, "set": 300.0}]
        
        col_wp, col_res = st.columns([1, 1.5])
        
        with col_wp:
            st.markdown("**Wegpunkte**")
            for i, wp in enumerate(st.session_state.ml_waypoints):
                with st.container(border=True):
                    wp_col1, wp_col2, wp_col3 = st.columns([2, 3, 3])
                    wp_col1.markdown(f"**P{i+1}**")
                    new_roll = wp_col2.number_input(f"Roll (Seite)", value=wp["roll"], step=10.0, key=f"ml_roll_{i}")
                    new_set = wp_col3.number_input(f"Set (Höhe)", value=wp["set"], step=10.0, key=f"ml_set_{i}")
                    st.session_state.ml_waypoints[i] = {"roll": new_roll, "set": new_set}
            
            ml_col1, ml_col2 = st.columns(2)
            if ml_col1.button("➕ Wegpunkt", use_container_width=True):
                last = st.session_state.ml_waypoints[-1]
                st.session_state.ml_waypoints.append({"roll": last["roll"] + 200, "set": last["set"] + 100})
                st.rerun()
            
            if ml_col2.button("➖ Entfernen", use_container_width=True, disabled=len(st.session_state.ml_waypoints) <= 2):
                st.session_state.ml_waypoints.pop()
                st.rerun()
            
            if st.button("🚀 Kette berechnen", type="primary", use_container_width=True):
                result = calc.calculate_multi_level_offset(st.session_state.ml_waypoints)
                st.session_state.ml_result = result
        
        with col_res:
            if 'ml_result' in st.session_state:
                result = st.session_state.ml_result
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.markdown("**Ergebnis**")
                    r1, r2 = st.columns(2)
                    r1.metric("Gesamt-Länge", f"{result['total_travel']:.1f} mm")
                    r2.metric("Segmente", result['num_segments'])
                    
                    st.divider()
                    st.markdown("**Segment-Details:**")
                    for seg in result["segments"]:
                        with st.container(border=True):
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.caption(f"**{seg['from']} → {seg['to']}**")
                            sc2.caption(f"Travel: {seg['travel']:.1f}mm")
                            sc3.caption(f"∠ {seg['angle']:.1f}°")

        # NEW: Isometric Drawing Generator
        st.divider()
        st.markdown("#### 📐 ISO-Zeichnung Generator")
        st.caption("Generiert automatisch isometrische Zeichnungen aus Wegpunkten")
        
        iso_col1, iso_col2 = st.columns([1, 1.5])
        
        with iso_col1:
            st.markdown("**Quelle wählen:**")
            use_multi_level = st.checkbox("Multi-Level Wegpunkte nutzen", value=True)
            
            if use_multi_level and 'ml_waypoints' in st.session_state:
                # Convert ml_waypoints format to 3D (assume y=cumulative for now)
                waypoints_3d = []
                cumulative_y = 0
                for i, wp in enumerate(st.session_state.ml_waypoints):
                    waypoints_3d.append({
                        "x": wp["roll"],
                        "y": cumulative_y,
                        "z": wp["set"]
                    })
                    cumulative_y += 500  # Default spacing
                    
                st.info(f"{len(waypoints_3d)} Wegpunkte aus Multi-Level")
            else:
                st.caption("Manuelle Eingabe (vereinfacht):")
                waypoints_3d = [
                    {"x": 0, "y": 0, "z": 0},
                    {"x": 400, "y": 500, "z": 300},
                    {"x": 200, "y": 1000, "z": 600}
                ]
            
            pipe_dn_iso = st.selectbox("Rohr-DN (Darstellung)", [50, 100, 150, 200], index=1)
            show_dims = st.checkbox("Maße anzeigen", value=True)
            
            if st.button("🎨 ISO-Zeichnung erstellen", type="primary", use_container_width=True):
                fig_iso = IsometricDrawer.draw_iso_route(waypoints_3d, pipe_dn_iso, show_dims)
                st.session_state.iso_figure = fig_iso
                st.toast("✅ Zeichnung erstellt!")
        
        with iso_col2:
            if 'iso_figure' in st.session_state and st.session_state.iso_figure:
                st.markdown("**Ergebnis:**")
                st.pyplot(st.session_state.iso_figure, use_container_width=True)
                
                # Export button
                buf = BytesIO()
                st.session_state.iso_figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    "📥 Als PNG Download",
                    buf,
                    file_name=f"iso_zeichnung_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )

    with geo_tabs[2]:
        with st.container(border=True):
            st.markdown("#### Standard Bogen-Rechner")
            with st.form(key="geo_bend_form"):
                cb1, cb2 = st.columns(2)
                angle = cb1.slider("Winkel", 0, 90, 45, key="gb_ang_std")
                dn_b = cb2.selectbox("DN", df['DN'], index=6, key="gb_dn_std")
                submit_bend = st.form_submit_button("Berechnen")
            
            if submit_bend:
                details = calc.calculate_bend_details(dn_b, angle)
                st.session_state.calc_res_bend = details
        
        if 'calc_res_bend' in st.session_state:
            details = st.session_state.calc_res_bend
            c_v, c_l = st.columns([1, 2])
            with c_v: st.metric("Vorbau (Z-Maß)", f"{details['vorbau']:.1f} mm")
            with c_l:
                cm1, cm2, cm3 = st.columns(3)
                cm1.metric("Rücken", f"{details['bogen_aussen']:.1f}")
                cm2.metric("Mitte", f"{details['bogen_mitte']:.1f}") 
                cm3.metric("Bauch", f"{details['bogen_innen']:.1f}")

    with geo_tabs[3]:
        st.info("🦞 Berechnung für Segmentbögen (Lobster Back) ohne Standard-Fittings.")
        with st.form(key="geo_seg_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                dn_seg = st.selectbox("Nennweite", df['DN'], index=8, key="seg_dn")
                radius_seg = st.number_input("Radius (R) [mm]", value=1000.0, step=10.0)
            with c2:
                num_seg = st.number_input("Anzahl Segmente (Ganze)", value=3, min_value=2, step=1)
                total_ang = st.number_input("Gesamtwinkel", value=90.0, step=5.0)
            with c3:
                submit_seg = st.form_submit_button("Berechnen 🦞", type="primary")
        
        if submit_seg:
            res = calc.calculate_segment_bend(dn_seg, radius_seg, num_seg, total_ang)
            st.session_state.calc_res_seg = res
        
        if 'calc_res_seg' in st.session_state:
            res = st.session_state.calc_res_seg
            if "error" in res:
                st.error(res["error"])
            else:
                st.divider()
                with st.container(border=True):
                    c_res1, c_res2 = st.columns([1, 1])
                    with c_res1:
                        st.markdown("#### Mittelstück (Ganz)")
                        st.metric("Rücken (L_aussen)", f"{res['mid_back']:.1f} mm")
                        st.metric("Bauch (L_innen)", f"{res['mid_belly']:.1f} mm")
                        st.caption(f"Schnittwinkel: {res['miter_angle']:.2f}°")
                    with c_res2:
                        st.markdown("#### Endstück (Halb)")
                        st.metric("Rücken bis Schnitt", f"{res['end_back']:.1f} mm")
                        st.metric("Bauch bis Schnitt", f"{res['end_belly']:.1f} mm")
                fig_seg = Visualizer.plot_segment_schematic(res['mid_back'], res['mid_belly'], res['od'], res['miter_angle'])
                st.pyplot(fig_seg, use_container_width=False)

    with geo_tabs[4]:
        with st.form(key="geo_noz_form"):
            c1, c2 = st.columns(2)
            dn_stub = c1.selectbox("DN Stutzen", df['DN'], index=5, key="gs_dn1")
            dn_main = c2.selectbox("DN Hauptrohr", df['DN'], index=8, key="gs_dn2")
            submit_noz = st.form_submit_button("Berechnen Stutzen")
        
        if submit_noz:
            try:
                df_c = calc.calculate_stutzen_coords(dn_main, dn_stub)
                fig = Visualizer.plot_stutzen(dn_main, dn_stub, df)
                st.session_state.calc_res_noz = (df_c, fig)
            except ValueError as e: st.error(str(e))
            
        if 'calc_res_noz' in st.session_state:
            df_c, fig = st.session_state.calc_res_noz
            c_r1, c_r2 = st.columns([1, 2])
            with c_r1: st.table(df_c)
            with c_r2: 
                if fig: st.pyplot(fig)
                else: st.error("⚠️ Geometrie ungültig")

def render_mto_tab(active_pid: int, proj_name: str):
    st.markdown('<div class="machine-header-doc">📦 MATERIAL MANAGER</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='project-tag'>📍 PROJEKT: {html.escape(proj_name)}</div>", unsafe_allow_html=True)
    df_log = DatabaseRepository.get_logbook_by_project(active_pid)
    if df_log.empty:
        st.info("Keine Daten im Rohrbuch. Das Materiallager ist leer.")
        return
    mto_df = MaterialManager.generate_mto(df_log)
    if not mto_df.empty:
        with st.container(border=True):
            total_items = len(mto_df)
            total_meters = mto_df[mto_df['Einheit']=='m']['Menge'].sum()
            c1, c2 = st.columns(2)
            c1.metric("Positionen", total_items, "verschiedene Artikel")
            c2.metric("Verrohrung gesamt", f"{total_meters:.1f} m", "Laufmeter")
        
        st.divider()
        fname = f"MTO_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        st.download_button("📥 MTO als Excel herunterladen", Exporter.to_excel(mto_df), fname, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.dataframe(mto_df, use_container_width=True, hide_index=True)

def render_logbook(df_pipe: pd.DataFrame):
    st.markdown('<div class="machine-header-doc">📝 ROHRBUCH</div>', unsafe_allow_html=True)
    
    proj_name = st.session_state.get('active_project_name', 'Unbekannt')
    active_pid = st.session_state.get('active_project_id', 1)
    is_archived = st.session_state.get('project_archived', 0)

    st.markdown(f"<div class='project-tag'>📍 PROJEKT: {html.escape(proj_name)} (ID: {active_pid})</div>", unsafe_allow_html=True)

    bulk_ids = st.session_state.get('bulk_edit_ids', [])
    
    if not is_archived:
        # --- MASSEN-BEARBEITUNG BLOCK ---
        if len(bulk_ids) > 1:
            st.warning(f"⚡ MASSEN-BEARBEITUNG: {len(bulk_ids)} Einträge ausgewählt")
            with st.container(border=True):
                with st.form("bulk_edit_form"):
                    c_bulk1, c_bulk2 = st.columns([1, 2])
                    target_field = c_bulk1.selectbox("Feld ändern:", ["Schweißer", "APZ / Charge", "ISO", "Datum"])
                    
                    new_value = ""
                    if target_field == "Datum":
                        d_val = c_bulk2.date_input("Neues Datum", datetime.now())
                        new_value = d_val.strftime("%d.%m.%Y")
                    else:
                        new_value = c_bulk2.text_input("Neuer Wert")
                    
                    submit_bulk = st.form_submit_button("🚀 Alle ändern", type="primary")
                
                if submit_bulk:
                    DatabaseRepository.bulk_update(bulk_ids, target_field, new_value)
                    st.toast(f"🚀 {len(bulk_ids)} Einträge aktualisiert!", icon="✅")
                    time.sleep(0.5)
                    st.session_state.bulk_edit_ids = []
                    st.session_state.logbook_select_all = False
                    st.session_state.logbook_key_counter += 1
                    st.rerun()
                
                if st.button("Abbrechen (Auswahl aufheben)"):
                    st.session_state.bulk_edit_ids = []
                    st.session_state.logbook_select_all = False
                    st.session_state.logbook_key_counter += 1
                    st.rerun()

        # --- EINZEL-BEARBEITUNG BLOCK ---
        else:
            if len(bulk_ids) == 1:
                if st.session_state.editing_id != bulk_ids[0]:
                    st.session_state.editing_id = bulk_ids[0]
            
            header_text = "Eintrag bearbeiten ✏️" if st.session_state.editing_id else "Neuer Eintrag ➕"
            
            with st.container(border=True):
                st.markdown(f"#### {header_text}")
                
                with st.form("logbook_entry_form"):
                    def_iso = st.session_state.last_iso if not st.session_state.editing_id else ""
                    def_sch = st.session_state.last_schweisser if not st.session_state.editing_id else ""
                    def_apz = st.session_state.last_apz if not st.session_state.editing_id else ""
                    def_dat = st.session_state.last_datum if not st.session_state.editing_id else datetime.now()

                    c1, c2, c3 = st.columns(3)
                    
                    current_iso = st.session_state.form_iso if st.session_state.editing_id else def_iso
                    iso_val = c1.text_input("ISO / Bez.", value=current_iso)

                    if 'form_naht' not in st.session_state: st.session_state.form_naht = ""
                    naht_val = c2.text_input("Naht", value=st.session_state.form_naht)
                    
                    if 'form_datum' not in st.session_state: st.session_state.form_datum = def_dat
                    if isinstance(st.session_state.form_datum, str):
                          try: st.session_state.form_datum = datetime.strptime(st.session_state.form_datum, "%d.%m.%Y").date()
                          except (ValueError, TypeError): st.session_state.form_datum = datetime.now().date()
                    dat_val = c3.date_input("Datum", value=st.session_state.form_datum)
                    
                    c4, c5, c6 = st.columns(3)
                    
                    if 'form_bauteil_idx' not in st.session_state: st.session_state.form_bauteil_idx = 0
                    bt_idx = st.session_state.form_bauteil_idx
                    bt_options = ["Rohrstoß", "Bogen", "Flansch", "T-Stück", "Reduzierung", "Stutzen", "Passstück", "Nippel", "Muffe"]
                    if bt_idx >= len(bt_options): bt_idx = 0
                    bt_val = c4.selectbox("Bauteil", bt_options, index=bt_idx)
                    
                    if 'form_dn_idx' not in st.session_state: st.session_state.form_dn_idx = 8
                    dn_idx = st.session_state.form_dn_idx
                    if dn_idx >= len(df_pipe): dn_idx = 8
                    dn_val = c5.selectbox("Dimension", df_pipe['DN'], index=dn_idx)
                    
                    final_dim_str = f"DN {dn_val}" 

                    if 'form_len' not in st.session_state: st.session_state.form_len = 0.0
                    len_val = c6.number_input("Länge (mm)", value=float(st.session_state.form_len), step=1.0) 
                    
                    c7, c8 = st.columns(2)
                    
                    current_apz = st.session_state.form_apz if st.session_state.editing_id else def_apz
                    apz_val = c7.text_input("APZ / Zeugnis", value=current_apz)
                    
                    current_sch = st.session_state.form_schweisser if st.session_state.editing_id else def_sch
                    sch_val = c8.text_input("Schweißer", value=current_sch)
                    
                    submit_entry = st.form_submit_button("SPEICHERN 💾", type="primary", use_container_width=True)

                if submit_entry:
                    if st.session_state.editing_id:
                        DatabaseRepository.update_full_entry(st.session_state.editing_id, {
                            "iso": iso_val, "naht": naht_val, "datum": dat_val.strftime("%d.%m.%Y"),
                            "dimension": final_dim_str, "bauteil": bt_val, "laenge": len_val,
                            "charge_apz": apz_val, "schweisser": sch_val
                        })
                        st.toast("✅ Eintrag aktualisiert!", icon="✏️")
                        st.session_state.editing_id = None
                        st.session_state.bulk_edit_ids = []
                        st.session_state.logbook_select_all = False
                        st.session_state.logbook_key_counter += 1
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        DatabaseRepository.add_entry({
                            "iso": iso_val, "naht": naht_val, "datum": dat_val.strftime("%d.%m.%Y"),
                            "dimension": final_dim_str, "bauteil": bt_val, "laenge": len_val,
                            "charge": "", "charge_apz": apz_val, "schweisser": sch_val,
                            "project_id": active_pid
                        })
                        st.session_state.last_iso = iso_val
                        st.session_state.last_apz = apz_val
                        st.session_state.last_schweisser = sch_val
                        st.session_state.last_datum = dat_val
                        st.toast("✅ Gespeichert!", icon="💾")
                        time.sleep(0.5)
                        st.rerun()
                
                if st.session_state.editing_id:
                    if st.button("Abbrechen", use_container_width=True):
                        st.session_state.editing_id = None
                        st.session_state.bulk_edit_ids = []
                        st.rerun()

    st.divider()
    
    df = DatabaseRepository.get_logbook_by_project(active_pid)
    
    if not df.empty:
        c_exp, c_sel_all, c_desel_all, _ = st.columns([1, 1, 1, 2])
        
        fname_base = f"Rohrbuch_{proj_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        c_exp.download_button("📥 Excel", Exporter.to_excel(df), f"{fname_base}.xlsx")
        
        if c_sel_all.button("☑️ Alle auswählen"):
            st.session_state.logbook_select_all = True
            st.session_state.logbook_key_counter += 1 
            st.rerun()
            
        if c_desel_all.button("☐ Keine"):
            st.session_state.logbook_select_all = False
            st.session_state.logbook_key_counter += 1
            st.rerun()
        
        st.markdown("### 📋 Einträge")
        
        df_display = df.copy()
        current_selection_state = st.session_state.get('logbook_select_all', False)
        df_display.insert(0, "Auswahl", current_selection_state)
        
        dynamic_key = f"logbook_editor_native_{st.session_state.get('logbook_key_counter', 0)}"
        
        edited_df = st.data_editor(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Auswahl": st.column_config.CheckboxColumn("☑️", width="small", default=current_selection_state),
                "id": None, 
                "project_id": None,
                "✏️": None,
                "Löschen": None,
                "iso": st.column_config.TextColumn("ISO", width="medium"),
                "naht": st.column_config.TextColumn("Naht", width="small"),
                "datum": st.column_config.TextColumn("Datum", width="small"),
                "bauteil": st.column_config.TextColumn("Bauteil", width="medium"),
                "dimension": st.column_config.TextColumn("Dimension", width="small"),
                "schweisser": st.column_config.TextColumn("Schweißer", width="small"),
                "charge_apz": st.column_config.TextColumn("APZ/Charge", width="medium"),
            },
            disabled=["iso", "naht", "datum", "dimension", "bauteil", "laenge", "charge", "charge_apz", "schweisser", "id", "project_id"],
            key=dynamic_key
        )
        
        selected_rows = edited_df[edited_df['Auswahl'] == True]
        selected_ids_list = selected_rows['id'].tolist()
        
        current_bulk_set = set(st.session_state.bulk_edit_ids)
        new_bulk_set = set(selected_ids_list)
        
        if current_bulk_set != new_bulk_set:
            st.session_state.bulk_edit_ids = selected_ids_list
            
            if len(selected_ids_list) == 1:
                sel_row = selected_rows.iloc[0].to_dict()
                st.session_state.editing_id = int(sel_row['id'])
                st.session_state.form_iso = sel_row.get('iso', '')
                st.session_state.form_naht = sel_row.get('naht', '')
                st.session_state.form_apz = sel_row.get('charge_apz', '')
                st.session_state.form_schweisser = sel_row.get('schweisser', '')
                try: 
                    d_str = sel_row.get('datum', datetime.now().strftime("%d.%m.%Y"))
                    st.session_state.form_datum = datetime.strptime(d_str, "%d.%m.%Y").date()
                except (ValueError, TypeError): st.session_state.form_datum = datetime.now().date()

            if len(selected_ids_list) != 1:
                st.session_state.editing_id = None
                
            st.rerun()

        if len(selected_ids_list) > 1:
             if st.button(f"🗑️ {len(selected_ids_list)} Einträge löschen", type="secondary"):
                DatabaseRepository.delete_entries(selected_ids_list)
                st.session_state.editing_id = None
                st.session_state.bulk_edit_ids = []
                st.session_state.logbook_select_all = False
                st.session_state.logbook_key_counter += 1
                st.toast(f"🗑️ {len(selected_ids_list)} Einträge gelöscht!")
                time.sleep(0.5)
                st.rerun()
    else:
        st.info(f"Keine Einträge für Projekt '{proj_name}'.")

def render_tab_handbook(calc: PipeCalculator, dn: int, pn: str):
    st.markdown('<div class="machine-header-doc">📚 SMART DATA</div>', unsafe_allow_html=True)
    row = calc.get_row(dn)
    suffix = "_16" if pn == "PN 16" else "_10"
    st.markdown(f"**DN {dn} / {pn}**")

    od = float(row['D_Aussen'])
    flange_b = float(row[f'Flansch_b{suffix}'])
    lk = float(row[f'LK_k{suffix}'])
    bolt = row[f'Schraube_M{suffix}']
    n_holes = int(row[f'Lochzahl{suffix}'])
    
    with st.container(border=True):
        st.markdown("##### 🏗️ Gewichte & Hydrotest")
        with st.form("handbook_weight_form"):
            c_in1, c_in2 = st.columns([1, 2])
            with c_in1:
                wt_input = st.number_input("Wandstärke (mm)", value=6.3, min_value=1.0, step=0.1)
                len_input = st.number_input("Rohrlänge (m)", value=6.0, step=0.5)
            
            submit_weight = st.form_submit_button("Berechnen")
        
        if submit_weight or True: # Initiale Berechnung erlauben
            w_data = HandbookCalculator.calculate_weight(od, wt_input, len_input * 1000)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Leergewicht (Stahl)", f"{w_data['total_steel']:.1f} kg", f"{w_data['kg_per_m_steel']:.1f} kg/m")
            mc2.metric("Gewicht Gefüllt", f"{w_data['total_filled']:.1f} kg", "für Hydrotest")
            mc3.metric("Füllvolumen", f"{w_data['volume_l']:.0f} Liter", "Wasserbedarf")

    c_geo1, c_geo2 = st.columns(2)
    with c_geo1:
        with st.container(border=True):
            st.markdown("##### 📐 Flansch")
            st.write(f"**Blatt:** {flange_b} mm | **Lochkreis:** {lk} mm")
            st.write(f"**Bohrung:** {n_holes} x {bolt}")
            progress_val = min(lk / (od + 100), 1.0)
            st.progress(progress_val, text="Lochkreis Verhältnis")

    with c_geo2:
        with st.container(border=True):
            st.markdown("##### 🔘 Dichtung (Check)")
            d_innen = od - (2*wt_input) 
            d_aussen = lk - (int(bolt.replace("M","")) * 1.5)
            st.info(f"ID: ~{d_innen:.0f} mm | AD: ~{d_aussen:.0f} mm | 2.0mm")

    st.divider()
    
    with st.container(border=True):
        st.markdown("#### 🔧 Montage & Drehmomente (8.8)")
        
        cb_col1, cb_col2 = st.columns([1, 2.5])
        
        with cb_col1:
            st.caption("Konfiguration")
            conn_type = st.radio("Typ", ["Fest-Fest", "Fest-Los", "Fest-Blind"], index=0, label_visibility="collapsed")
            use_washers = st.checkbox("2x U-Scheibe", value=True)
            is_lubed = st.toggle("Geschmiert (MoS2)", value=True)
            gasket_thk = st.number_input("Dichtung", value=2.0, step=0.5)
            
        with cb_col2:
            bolt_info = HandbookCalculator.BOLT_DATA.get(bolt, [0, 0, 0])
            sw, nm_dry, nm_lube = bolt_info
            
            t1 = flange_b
            t2 = flange_b 
            if "Los" in conn_type: t2 = flange_b + 5 
            elif "Blind" in conn_type: t2 = flange_b + (dn * 0.02)
                
            n_washers = 2 if use_washers else 0
            calc_len = HandbookCalculator.get_bolt_length(t1, t2, bolt, n_washers, gasket_thk)
            torque = nm_lube if is_lubed else nm_dry
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Bolzen", f"{bolt} x {calc_len}", f"{n_holes} Stk.")
            m2.metric("Schlüsselweite", f"SW {sw} mm", "Nuss/Ring")
            m3.metric("Drehmoment", f"{torque} Nm", "Geschmiert" if is_lubed else "Trocken")

def render_closeout_tab(active_pid: int, proj_name: str, is_archived: int):
    st.markdown('<div class="machine-header-doc">🏁 FERTIGSTELLUNG (HANDOVER)</div>', unsafe_allow_html=True)
    
    if is_archived:
        st.warning(f"Projekt '{proj_name}' ist abgeschlossen und archiviert.")
        if st.button("🔓 Projekt wiedereröffnen (Reopen)"):
            DatabaseRepository.toggle_archive_project(active_pid, False)
            st.session_state.project_archived = 0
            st.rerun()
            
        df_log = DatabaseRepository.get_logbook_by_project(active_pid)
        if not df_log.empty and PDF_AVAILABLE:
            st.divider()
            st.markdown("#### Dokumentation (Abruf)")
            meta_saved = st.session_state.get('last_handover_meta', {})
            pdf_data = Exporter.to_pdf_final_report(df_log, proj_name, meta_saved)
            st.download_button("📄 Fertigungsbescheinigung herunterladen", pdf_data, f"Fertigungsbescheinigung_{proj_name}.pdf", "application/pdf", type="primary")
        return

    st.info("Erstellung der Fertigungsbescheinigung für die Abnahme.")
    df_log = DatabaseRepository.get_logbook_by_project(active_pid)
    
    if 'ho_order' not in st.session_state: st.session_state.ho_order = ""
    if 'ho_sys' not in st.session_state: st.session_state.ho_sys = ""
    if 'ho_rt' not in st.session_state: st.session_state.ho_rt = True
    if 'ho_dim' not in st.session_state: st.session_state.ho_dim = False
    if 'ho_iso' not in st.session_state: st.session_state.ho_iso = False

    with st.container(border=True):
        with st.form(key="handover_form"):
            st.markdown("#### 1. Projektdaten für Deckblatt")
            c1, c2 = st.columns(2)
            in_order = c1.text_input("Auftrags-Nr. / Ticket", value=st.session_state.ho_order)
            in_sys = c2.text_input("Anlagenteil / System", value=st.session_state.ho_sys)
            
            st.markdown("#### 2. Qualitätssicherung (Bestätigung)")
            c_rt, c_dim, c_iso = st.columns(3)
            check_rt = c_rt.checkbox("ZfP: RT (Röntgen) i.O.", value=st.session_state.ho_rt)
            check_dim = c_dim.checkbox("Maßhaltigkeit geprüft", value=st.session_state.ho_dim)
            check_iso = c_iso.checkbox("Isometrie (As-Built)", value=st.session_state.ho_iso)
            
            submit_update = st.form_submit_button("💾 Daten für PDF übernehmen", type="primary")

    if submit_update:
        st.session_state.ho_order = in_order
        st.session_state.ho_sys = in_sys
        st.session_state.ho_rt = check_rt
        st.session_state.ho_dim = check_dim
        st.session_state.ho_iso = check_iso
        st.toast("Daten übernommen! Vorschau aktualisiert.", icon="📄")
    
    missing_apz = len(df_log[df_log['charge_apz'].astype(str).str.strip() == ''])
    missing_weld = len(df_log[df_log['schweisser'].astype(str).str.strip() == ''])
    ready = True
    
    if missing_apz > 0 or missing_weld > 0:
        st.warning(f"Hinweis: Es fehlen {missing_apz} APZs und {missing_weld} Schweißer-Einträge.")
        ready = False 

    st.divider()
    
    col_act, col_pdf = st.columns(2)
    
    meta_data = {
        "order_no": st.session_state.ho_order,
        "system_name": st.session_state.ho_sys,
        "check_rt": st.session_state.ho_rt,
        "check_dim": st.session_state.ho_dim,
        "check_iso": st.session_state.ho_iso
    }
    
    st.session_state.last_handover_meta = meta_data

    with col_act:
        force_close = False
        if not ready:
            force_close = st.checkbox("⚠️ Trotz fehlender Daten abschließen")
        
        if ready or force_close:
            if st.button("🏁 PROJEKT ARCHIVIEREN", type="secondary"):
                DatabaseRepository.toggle_archive_project(active_pid, True)
                st.session_state.project_archived = 1
                st.balloons()
                st.rerun()
        else:
            st.button("🏁 Abschließen", disabled=True)

    with col_pdf:
        if not df_log.empty and PDF_AVAILABLE:
            pdf_data = Exporter.to_pdf_final_report(df_log, proj_name, meta_data)
            st.caption(f"Vorschau Daten: Ticket '{meta_data['order_no']}' | System '{meta_data['system_name']}'")
            st.download_button(
                label="📄 PDF Bescheinigung herunterladen", 
                data=pdf_data, 
                file_name=f"Fertigungsbescheinigung_{proj_name}.pdf", 
                mime="application/pdf",
                type="primary"
            )

def main():
    init_app_state()
    DatabaseRepository.init_db()
    df_pipe = load_pipe_data()
    calc = PipeCalculator(df_pipe)

    render_sidebar_projects()

    with st.sidebar:
        st.subheader("⚙️ Setup")
        dn = st.selectbox("Nennweite", df_pipe['DN'], index=8)
        pn = st.radio("Druckstufe", ["PN 16", "PN 10"], horizontal=True)

    tabs = ["🪚 Smarte Säge", "📐 Geometrie", "📝 Rohrbuch", "📦 Material", "📚 Smart Data", "🏁 Handover"]
    
    if st.session_state.active_tab not in tabs:
        st.session_state.active_tab = tabs[0]
    
    selected_tab = st.radio("Menü", tabs, horizontal=True, label_visibility="collapsed", key="nav_radio", index=tabs.index(st.session_state.active_tab))
    
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
        st.rerun()

    if st.session_state.active_tab == "🪚 Smarte Säge":
        render_smart_saw(calc, df_pipe, dn, pn)
    elif st.session_state.active_tab == "📐 Geometrie":
        render_geometry_tools(calc, df_pipe)
    elif st.session_state.active_tab == "📝 Rohrbuch":
        render_logbook(df_pipe)
    elif st.session_state.active_tab == "📦 Material":
        render_mto_tab(st.session_state.active_project_id, st.session_state.active_project_name)
    elif st.session_state.active_tab == "📚 Smart Data":
        render_tab_handbook(calc, dn, pn)
    elif st.session_state.active_tab == "🏁 Handover":
        render_closeout_tab(st.session_state.active_project_id, st.session_state.active_project_name, st.session_state.project_archived)

if __name__ == "__main__":
    main()
