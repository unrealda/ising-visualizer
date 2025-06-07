import streamlit as st
import numpy as np
from ising_model import simulate_ising, cluster_sizes, hysteresis_loop
from visualizer import plot_magnetization_vs_temp, plot_cluster_distribution, plot_hysteresis

st.set_page_config(page_title="Ising模型模拟", layout="wide")
st.title("二维Ising模型模拟与分析")

# 初始化session_state缓存结构
if 'M_data' not in st.session_state:
    st.session_state.M_data = None
if 'cluster_data' not in st.session_state:
    st.session_state.cluster_data = None
if 'hysteresis_data' not in st.session_state:
    st.session_state.hysteresis_data = None

# -------- 参数设置 --------
st.sidebar.header("模型参数设置")
L = st.sidebar.number_input("晶格边长 L", 10, 50, 20)
lattice_type = st.sidebar.selectbox("晶格类型", ["Square", "Triangular"])

# 磁化率参数
st.sidebar.subheader("磁化率计算参数")
Tmin = st.sidebar.slider("最低温度 Tmin", 0.1, 5.0, 1.0)
Tmax = st.sidebar.slider("最高温度 Tmax", Tmin, 5.0, 3.0)
nT = st.sidebar.number_input("温度点数 nT", 5, 30, 10)
n_steps = st.sidebar.number_input("每温度蒙特卡洛步数", 100, 1000, 300)

# 簇大小分布参数
st.sidebar.subheader("簇大小分布参数")
T_cluster = st.sidebar.slider("簇大小分布温度 T", Tmin, Tmax, (Tmin + Tmax) / 2)

# 磁滞回线参数
st.sidebar.subheader("磁滞回线参数")
T_hys = st.sidebar.slider("磁滞回线温度 T", Tmin, Tmax, (Tmin + Tmax) / 2)
n_eq = st.sidebar.number_input("每场平衡步数", 50, 500, 200)
n_meas = st.sidebar.number_input("每场测量步数", 50, 500, 200)

# -------- 磁化率计算 --------
st.header("磁化率、剩余磁化强度、Binder比率计算")
if st.button("计算磁化率及相关特征"):
    T_vals = np.linspace(Tmin, Tmax, nT)
    M_mean = []
    M_std = []
    chi_mean = []
    chi_std = []
    binder_mean = []
    binder_std = []
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
        chi_std.append(0)  # 这里简化，没做误差估计
        binder_mean.append(binder)
        binder_std.append(0)
    st.session_state.M_data = (T_vals, M_mean, M_std, chi_mean, chi_std, binder_mean, binder_std)
    st.success("磁化率计算完成！")

if st.session_state.M_data is not None:
    T_vals, M_mean, M_std, chi_mean, chi_std, binder_mean, binder_std = st.session_state.M_data
    fig1 = plot_magnetization_vs_temp(T_vals, M_mean, M_std, chi_mean, binder_mean)
    st.pyplot(fig1)

# -------- 簇大小分布计算 --------
st.header("簇大小分布")
if st.button("计算簇大小分布"):
    lattice, _, _, _, _ = simulate_ising(L, T_cluster, n_steps, 0.0, lattice_type)
    sizes = cluster_sizes(lattice, lattice_type)
    st.session_state.cluster_data = sizes
    st.success("簇大小分布计算完成！")

if st.session_state.cluster_data is not None:
    fig2 = plot_cluster_distribution(st.session_state.cluster_data)
    st.pyplot(fig2)

# -------- 磁滞回线及动画 --------
st.header("磁滞回线及动画")
if st.button("生成磁滞回线"):
    H_values = np.concatenate((np.linspace(-2, 2, 40), np.linspace(2, -2, 40)))
    magnetizations, coercivity, remanence, lattice = hysteresis_loop(L, T_hys, H_values, n_eq, n_meas, lattice_type)
    st.session_state.hysteresis_data = (H_values, magnetizations, coercivity, remanence)
    st.success("磁滞回线生成完成！")

if st.session_state.hysteresis_data is not None:
    H_values, magnetizations, coercivity, remanence = st.session_state.hysteresis_data
    fig3 = plot_hysteresis(H_values, magnetizations, coercivity, remanence)
    st.pyplot(fig3)
