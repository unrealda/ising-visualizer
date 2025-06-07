import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
from ising_model import run_ising_simulation
from visualizer import generate_arrow_plot

st.set_page_config(layout="wide")
st.title("2D Ising Model Visualizer - Wolff Algorithm")

st.sidebar.header("Simulation Parameters")
L = st.sidebar.slider("Lattice size (LxL)", 8, 64, 16)
Tmin = st.sidebar.number_input("Minimum Temperature", value=1.5)
Tmax = st.sidebar.number_input("Maximum Temperature", value=3.5)
nT = st.sidebar.slider("Number of Temperature Steps", 5, 100, 20)
Ntrial = st.sidebar.slider("MC Trials per Temperature", 10, 1000, 100)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation and generating plots..."):
        folder = "sim_output"
        os.makedirs(folder, exist_ok=True)
        T_list, spin_configs, M, Chi = run_ising_simulation(L, Tmin, Tmax, nT, Ntrial, folder)

        st.success("Simulation complete!")

        # Show magnetization and susceptibility plot
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

        # Select snapshot to visualize
        t_idx = st.slider("Select temperature snapshot to visualize", 0, nT-1, 0)
        fig2 = generate_arrow_plot(spin_configs[t_idx], T_list[t_idx])
        st.pyplot(fig2)

        # Download image button
        buf = BytesIO()
        fig2.savefig(buf, format="png")
        st.download_button("Download Snapshot Image", data=buf.getvalue(), file_name="ising_snapshot.png", mime="image/png")
else:
    st.info("Set parameters and click 'Run Simulation' to begin.")
