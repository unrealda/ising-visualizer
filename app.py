import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
from ising_model import run_temperature_scan, run_hysteresis
from visualizer import (
    plot_magnetization_vs_temp,
    save_all_spin_snapshots,
    save_all_hysteresis_loops,
    save_final_hysteresis_snapshots
)

st.set_page_config(layout="wide")
st.title("2D Ising Model Visualizer - Wolff Algorithm")

if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False

st.sidebar.header("Simulation Parameters")
L = st.sidebar.slider("Lattice size (LxL)", 8, 64, 16)
Tmin = st.sidebar.number_input("Minimum Temperature", value=1.5)
Tmax = st.sidebar.number_input("Maximum Temperature", value=3.5)
nT = st.sidebar.slider("Number of Temperature Steps", 5, 100, 20)
Ntrial = st.sidebar.slider("MC Trials per Temperature", 10, 1000, 100)

run_hysteresis = st.sidebar.checkbox("Run Hysteresis Simulation")

if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        folder = "sim_output"
        os.makedirs(folder, exist_ok=True)
        
        results = run_temperature_scan(L, lattice, Ntrial, Tmin, Tmax, nT)
        st.session_state.simulation_done = True
        st.session_state.T_list = T_list
        st.session_state.M = M
        st.session_state.Chi = Chi
        st.session_state.spin_snapshots = spin_configs

        if run_hysteresis:
            gif_path = run_hysteresis_simulation(L, Tmin, Tmax, nT, Ntrial, folder)
            st.session_state.hysteresis_gif = gif_path

        st.success("Simulation complete!")

if st.session_state.simulation_done:
    T_list = st.session_state.T_list
    M = st.session_state.M
    Chi = st.session_state.Chi
    spin_configs = st.session_state.spin_snapshots

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Magnetization', color=color)
    ax1.plot(T_list, M, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Susceptibility', color=color)
    ax2.plot(T_list, Chi, 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    st.pyplot(fig)

    t_idx = st.slider("Select temperature snapshot to visualize", 0, len(T_list)-1, 0)
    snapshot = spin_configs[t_idx]
    T = T_list[t_idx]

    fig2 = generate_arrow_plot(snapshot, T)
    st.pyplot(fig2)

    buf = BytesIO()
    fig2.savefig(buf, format="png")
    st.download_button("Download Snapshot Image", data=buf.getvalue(), file_name="ising_snapshot.png", mime="image/png")

    if run_hysteresis and 'hysteresis_gif' in st.session_state:
        st.header("Hysteresis Animation")
        st.image(st.session_state.hysteresis_gif, caption="Magnetization Hysteresis Loop", use_column_width=True)

else:
    st.info("Set parameters and click 'Run Simulation' to begin.")
