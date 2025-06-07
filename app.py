import streamlit as st
import uuid
import os
import shutil
from zipfile import ZipFile
from scipy.signal import convolve2d
import imageio

from ising_model import run_temperature_scan, run_hysteresis
from visualizer import (
    plot_magnetization_vs_temp,
    save_all_spin_snapshots,
    save_all_hysteresis_loops,
    save_final_hysteresis_snapshots,
    plot_hysteresis_loop,
    plot_binder_ratio,
    plot_spin_ratio,
    generate_local_analysis,
    create_spin_animation,
    plot_hysteresis_features
)

# Streamlit app layout
st.title("2D Ising Model Visualizer")
st.markdown("Upload your parameter settings, or run a default simulation.")

# Sidebar simulation controls
st.sidebar.header("Simulation Settings")
sim_type = st.sidebar.selectbox("Simulation Type", ["Temperature Scan", "Hysteresis Loop"])
spin_size = st.sidebar.slider("Spin Lattice Size (NxN)", min_value=16, max_value=128, value=32, step=16)
num_steps = st.sidebar.number_input("Simulation Steps", min_value=100, max_value=100000, value=5000, step=1000)
output_dir = os.path.join("outputs", str(uuid.uuid4()))
os.makedirs(output_dir, exist_ok=True)

# Simulation execution
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        if sim_type == "Temperature Scan":
            results = run_temperature_scan(spin_size, num_steps)
            st.session_state.results = results
            st.success(f"Temperature scan completed. {len(results)} data points generated.")
        elif sim_type == "Hysteresis Loop":
            results = run_hysteresis(spin_size, num_steps)
            st.session_state.hyst_data = results
            st.success(f"Hysteresis loop completed. {len(results)} temperature points processed.")

# Analysis Section
st.header("Analysis and Visualization")

if sim_type == "Temperature Scan" and "results" in st.session_state:
    results = st.session_state.results
    T_list = [r['T'] for r in results]
    M2_list = [r['M2'] for r in results]
    M4_list = [r['M4'] for r in results]
    M2_var = [r['M2_var'] for r in results]
    M4_var = [r['M4_var'] for r in results]
    up_ratio_list = [r['up_ratio'] for r in results]
    down_ratio_list = [r['down_ratio'] for r in results]
    A_hyst_list = [r.get('A_hyst', 0) for r in results]  # Default to 0 if not present
    Hc_list = [r.get('Hc', 0) for r in results]
    Hc_var = [r.get('Hc_var', 0) for r in results]

    if st.button("Plot Magnetization vs Temperature"):
        save_path = os.path.join(output_dir, "Magnetization_vs_Temperature.png")
        plot_magnetization_vs_temp(results, save_path)
        st.image(save_path, caption="Magnetization and Susceptibility vs Temperature")

    if st.button("Plot Binder Ratio"):
        save_path = os.path.join(output_dir, "Binder_Ratio_vs_Temperature.png")
        plot_binder_ratio(T_list, M2_list, M4_list, M2_var, M4_var, save_path)
        st.image(save_path, caption="Binder Ratio vs Temperature")

    if st.button("Plot Spin Ratio"):
        save_path = os.path.join(output_dir, "Spin_Ratio_vs_Temperature.png")
        plot_spin_ratio(T_list, up_ratio_list, down_ratio_list, save_path)
        st.image(save_path, caption="Up/Down Spin Ratio vs Temperature")

    if st.button("Generate Spin Snapshots"):
        snapshots_dir = os.path.join(output_dir, "spin_snapshots")
        save_all_spin_snapshots(results, snapshots_dir)
        st.success(f"Spin snapshots saved to {snapshots_dir}.")

    if st.button("Generate Local Order Analysis"):
        local_dir = os.path.join(output_dir, "local_order")
        os.makedirs(local_dir, exist_ok=True)
        for r in results:
            generate_local_analysis(r['spins_snapshot'], r['T'], local_dir)
        st.success(f"Local order analysis saved to {local_dir}.")

    if st.button("Download All Snapshots as ZIP"):
        zip_name = os.path.join(output_dir, "snapshots.zip")
        with ZipFile(zip_name, 'w') as zipf:
            snapshots_dir = os.path.join(output_dir, "spin_snapshots")
            for root, _, files in os.walk(snapshots_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=snapshots_dir)
                    zipf.write(file_path, arcname)
        st.download_button("Download Snapshots ZIP", data=open(zip_name, 'rb'), file_name="snapshots.zip")

elif sim_type == "Hysteresis Loop" and "hyst_data" in st.session_state:
    hyst_data = st.session_state.hyst_data
    T_list = [h['T'] for h in hyst_data]
    A_hyst_list = [h['A_hyst'] for h in hyst_data]
    Hc_list = [h['Hc'] for h in hyst_data]
    Hc_var = [h['Hc_var'] for h in hyst_data]

    if st.button("Plot Hysteresis Loops"):
        loops_dir = os.path.join(output_dir, "hysteresis_loops")
        save_all_hysteresis_loops(hyst_data, loops_dir)
        st.success(f"Hysteresis loops saved to {loops_dir}.")

    if st.button("Plot Hysteresis Features"):
        plot_hysteresis_features(T_list, A_hyst_list, Hc_list, Hc_var, save_path=output_dir)
        st.image(os.path.join(output_dir, "Coercive_Field_vs_T.png"), caption="Coercive Field vs Temperature")
        st.image(os.path.join(output_dir, "Hysteresis_Area_vs_T.png"), caption="Hysteresis Area vs Temperature")

    if st.button("Save Final Hysteresis Snapshots"):
        final_dir = os.path.join(output_dir, "final_hysteresis")
        save_final_hysteresis_snapshots(hyst_data, final_dir)
        st.success(f"Final hysteresis snapshots saved to {final_dir}.")

# GIF Animation Section (optional, but often useful)
if st.sidebar.checkbox("Generate Spin Animation (experimental)"):
    if sim_type == "Temperature Scan" and "results" in st.session_state:
        spin_matrices = [r['spins_snapshot'] for r in st.session_state.results]
        T_list = [r['T'] for r in st.session_state.results]
        gif_path = os.path.join(output_dir, "spin_animation.gif")
        create_spin_animation(spin_matrices, T_list, gif_path)
        st.image(gif_path, caption="Spin Configuration Animation")
    else:
        st.warning("Spin animation is only available after running a Temperature Scan.")

