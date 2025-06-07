import streamlit as st
import numpy as np
from ising_model import simulate_ising, cluster_sizes, hysteresis_loop, monte_carlo_step, initialize_lattice
from visualizer import plot_magnetization_vs_temp, plot_cluster_distribution, plot_hysteresis, animate_hysteresis
import matplotlib.pyplot as plt
import tempfile
import matplotlib.animation as animation

st.set_page_config(layout="wide")
st.title("高级 Ising 模型蒙特卡洛模拟与磁滞回线动画")

L = st.sidebar.number_input("晶格边长 L", min_value=10, max_value=50, value=20, step=1)
lattice_type = st.sidebar.selectbox("晶格类型", ["Square", "Triangular"])

Tmin = st.sidebar.slider("最低温度 Tmin", 0.1, 5.0, 1.0)
Tmax = st.sidebar.slider("最高温度 Tmax", 0.1, 5.0, 3.0)
nT = st.sidebar.number_input("温度点数 nT", min_value=5, max_value=30, value=10)
n_steps = st.sidebar.number_input("每温度蒙特卡洛步数", min_value=100, max_value=1000, value=300)

st.header("磁化强度、磁化率与Binder比率计算")

if st.button("开始计算"):
    T_vals = np.linspace(Tmin, Tmax, nT)

    M_vals = []
    chi_vals = []
    binder_vals = []

    # 误差计算: 多次独立模拟
    runs = 3
    M_all = []
    chi_all = []
    binder_all = []

    progress_bar = st.progress(0)
    for idx, T in enumerate(T_vals):
        M_runs = []
        chi_runs = []
        binder_runs = []
        for _ in range(runs):
            _, M_avg, susceptibility, binder_cumulant, _ = simulate_ising(L, T, n_steps, 0.0, lattice_type)
            M_runs.append(M_avg)
            chi_runs.append(susceptibility)
            binder_runs.append(binder_cumulant)
        M_all.append(M_runs)
        chi_all.append(chi_runs)
        binder_all.append(binder_runs)
        progress_bar.progress((idx+1)/len(T_vals))

    # 计算平均和标准差
    M_arr = np.array(M_all)
    chi_arr = np.array(chi_all)
    binder_arr = np.array(binder_all)

    M_mean = np.mean(M_arr, axis=1)
    M_std = np.std(M_arr, axis=1)

    chi_mean = np.mean(chi_arr, axis=1)
    chi_std = np.std(chi_arr, axis=1)

    binder_mean = np.mean(binder_arr, axis=1)
    binder_std = np.std(binder_arr, axis=1)

    # 传入绘图函数，带误差条
    fig = plot_magnetization_vs_temp(
        T_vals,
        np.column_stack((M_mean, M_std)),
        np.column_stack((chi_mean, chi_std)),
        np.column_stack((binder_mean, binder_std)),
        lattice_type
    )
    st.pyplot(fig)

st.header("簇大小分布")

T_cluster = st.slider("簇大小分布温度 T", Tmin, Tmax, (Tmin+Tmax)/2)
if st.button("计算簇大小分布"):
    lattice, _, _, _, _ = simulate_ising(L, T_cluster, n_steps, 0.0, lattice_type)
    sizes = cluster_sizes(lattice, lattice_type)
    fig = plot_cluster_distribution(sizes)
    st.pyplot(fig)

st.header("磁滞回线模拟及动画")

T_hys = st.slider("磁滞回线温度 T", Tmin, Tmax, (Tmin+Tmax)/2)
n_eq = st.number_input("每场平衡步数", min_value=50, max_value=500, value=200)
n_meas = st.number_input("每场测量步数", min_value=50, max_value=500, value=200)

if st.button("生成磁滞回线并动画"):
    H_values = np.concatenate((np.linspace(-2, 2, 40), np.linspace(2, -2, 40)))

    magnetizations, coercivity, remanence, _ = hysteresis_loop(L, T_hys, H_values, n_eq, n_meas, lattice_type)

    # 生成动画用的晶格快照
    lattice_states = []
    lattice_tmp = initialize_lattice(L, lattice_type)
    for H in H_values:
        for _ in range(n_eq):
            lattice_tmp = monte_carlo_step(lattice_tmp, T_hys, H, lattice_type)
        lattice_states.append(lattice_tmp.copy())

    fig_hyst = plot_hysteresis(H_values, magnetizations, coercivity, remanence)
    st.pyplot(fig_hyst)

    ani = animate_hysteresis(H_values, magnetizations, lattice_states)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer='imagemagick')
    st.image(tmpfile.name)
