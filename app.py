import streamlit as st
import numpy as np
from ising_model import (
    simulate_ising, 
    cluster_sizes, 
    hysteresis_loop, 
    monte_carlo_step, 
    initialize_lattice
)
from visualizer import (
    plot_magnetization_vs_temp, 
    plot_cluster_distribution, 
    plot_hysteresis
)

st.set_page_config(page_title="二维Ising模型模拟", layout="wide")
st.title("二维Ising模型模拟与分析")

# -------- 参数设置 --------
st.sidebar.header("模型参数设置")
L = st.sidebar.number_input("晶格边长 L", 10, 50, 20)
lattice_type = st.sidebar.selectbox("晶格类型", ["Square", "Triangular"])

# -------- 磁化率计算 --------
st.header("磁化率、剩余磁化强度、Binder比率计算")
Tmin = st.slider("最低温度 Tmin", 0.1, 5.0, 1.0)
Tmax = st.slider("最高温度 Tmax", Tmin, 5.0, 3.0)
nT = st.number_input("温度点数 nT", 5, 30, 10)
n_steps = st.number_input("每温度蒙特卡洛步数", 100, 1000, 300)

if st.checkbox("启用磁化率计算", True):
    T_vals = np.linspace(Tmin, Tmax, nT)
    M_mean, M_std, chi_mean, binder_mean = [], [], [], []

    for T in T_vals:
        lattice, mags = simulate_ising(L, T, n_steps, 0.0, lattice_type)
        mags = np.array(mags)
        m = np.mean(mags)
        m2 = np.mean(mags**2)
        m4 = np.mean(mags**4)
        chi = L*L*(m2 - m**2)/T
        binder = 1 - m4/(3*m2**2)
        M_mean.append(m)
        M_std.append(np.std(mags))
        chi_mean.append(chi)
        binder_mean.append(binder)

    fig1 = plot_magnetization_vs_temp(T_vals, M_mean, M_std, chi_mean, binder_mean)
    st.pyplot(fig1)

# -------- 簇大小分布 --------
st.header("簇大小分布")
T_cluster = st.slider("簇大小分布温度 T", Tmin, Tmax, (Tmin + Tmax)/2)
if st.checkbox("启用簇大小分布计算", False):
    lattice, _, _, _, _ = simulate_ising(L, T_cluster, n_steps, 0.0, lattice_type)
    sizes = cluster_sizes(lattice, lattice_type)
    fig2 = plot_cluster_distribution(sizes)
    st.pyplot(fig2)

# -------- 磁滞回线 --------
st.header("磁滞回线及动画")
T_hys = st.slider("磁滞回线温度 T", Tmin, Tmax, (Tmin + Tmax)/2)
n_eq = st.number_input("每场平衡步数", 50, 500, 200)
n_meas = st.number_input("每场测量步数", 50, 500, 200)

if st.checkbox("启用磁滞回线计算", False):
    H_values = np.concatenate((np.linspace(-2, 2, 40), np.linspace(2, -2, 40)))
    magnetizations, coercivity, remanence, _ = hysteresis_loop(L, T_hys, H_values, n_eq, n_meas, lattice_type)
    fig3 = plot_hysteresis(H_values, magnetizations, coercivity, remanence)
    st.pyplot(fig3)
