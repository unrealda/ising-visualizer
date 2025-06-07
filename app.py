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
    plot_hysteresis_features,
    plot_with_errorbars
)

st.set_page_config(page_title="Ising Model (Wolff Algorithm)", layout="wide")
st.title("\U0001F9BE Wolff 算法模拟二维伊辛模型")

# Session state初始化
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
    st.session_state['has_run'] = False
    st.session_state['advanced'] = False

# 缓存目录设置
base_cache_dir = ".streamlit_cache"
os.makedirs(base_cache_dir, exist_ok=True)
tmpdir = os.path.join(base_cache_dir, st.session_state['session_id'])

# 侧边栏参数
with st.sidebar:
    st.header("参数设置")
    L = st.number_input("格子边长 L", min_value=4, max_value=128, value=8)
    lattice = st.selectbox("晶格类型", ["square", "triangular"])
    mode = st.radio("模拟模式", ["快速预览", "高精度模拟"])
    
    if mode == "快速预览":
        Ntrial = 30
        nT = 5
    else:
        Ntrial = st.number_input("每温度试验次数", min_value=10, max_value=1000, value=100)
        nT = st.number_input("温度步数", min_value=2, max_value=100, value=10)
    
    Tmin = st.number_input("最低温度 Tmin", min_value=0.1, value=1.0, step=0.1)
    Tmax = st.number_input("最高温度 Tmax", min_value=0.1, value=3.5, step=0.1)
    
    st.session_state['advanced'] = st.checkbox("启用MATLAB完整分析", False)
    if st.session_state['advanced']:
        st.session_state['order_thresh'] = st.slider("有序域阈值", 0.5, 1.0, 0.8)
    
    run_button = st.button("开始模拟")
    if st.button("清空缓存并重置"):
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        st.session_state.clear()
        st.rerun()

# 模拟函数
@st.cache_data(show_spinner=False)
def simulate_and_generate(L, lattice, Ntrial, Tmin, Tmax, nT, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    
    # 基础模拟
    results = run_temperature_scan(L, lattice, Ntrial, Tmin, Tmax, nT)
    T_list = [r['T'] for r in results]
    hyst_data = run_hysteresis(L, lattice, T_list, Ntrial=100)

    # 基础可视化
    plot_magnetization_vs_temp(results, os.path.join(tmpdir, "magnetization_vs_T.png"))
    spin_dir = os.path.join(tmpdir, "spin_snapshots")
    hyst_dir = os.path.join(tmpdir, "hysteresis_loops")
    final_dir = os.path.join(tmpdir, "final_hyst_frames")
    final_hyst_plot_dir = os.path.join(tmpdir, "final_hyst_plot_frames")
    
    save_all_spin_snapshots(results, spin_dir)
    save_all_hysteresis_loops(hyst_data, hyst_dir)
    save_final_hysteresis_snapshots(hyst_data, final_dir)
    
    # 高级分析
    if st.session_state['advanced']:
        plot_binder_ratio(results, os.path.join(tmpdir, "binder_ratio.png"))
        plot_spin_ratio(results, os.path.join(tmpdir, "spin_ratio.png"))
        
        error_dir = os.path.join(tmpdir, "error_analysis")
        plot_with_errorbars(results, error_dir)
        
        local_order_dir = os.path.join(tmpdir, "local_order")
        for res in results:
            generate_local_analysis(
                res['spins_snapshot'], 
                res['T'],
                st.session_state.order_thresh,
                local_order_dir
            )
        
        create_spin_animation(
            [r['spins_snapshot'] for r in results],
            [r['T'] for r in results],
            os.path.join(tmpdir, "spin_animation.gif")
        )
        
        plot_hysteresis_features(
            [h['T'] for h in hyst_data],
            {
                'A_hyst': [h['loop_area'] for h in hyst_data],
                'M_r': [h['M_r'] for h in hyst_data],
                'H_c': [h['H_c'] for h in hyst_data]
            },
            os.path.join(tmpdir, "hysteresis_features.png")
        )
    
    return results, hyst_data

# 运行模拟
if run_button:
    with st.spinner("正在运行模拟，请稍候..."):
        results, hyst_data = simulate_and_generate(L, lattice, Ntrial, Tmin, Tmax, nT, tmpdir)
        st.session_state['has_run'] = True

# 结果显示
if st.session_state.get('has_run', False):
    # 基础结果
    st.subheader("磁化率与温度关系图")
    st.image(os.path.join(tmpdir, "magnetization_vs_T.png"), use_container_width=True)
    
    cols = st.columns(2)
    with cols[0]:
        st.subheader("\u2191/\u2193 自旋分布图")
        spin_files = sorted(os.listdir(os.path.join(tmpdir, "spin_snapshots")))
        idx_spin = st.slider("温度帧", 0, len(spin_files)-1, 0, key="spin_slider")
        st.image(os.path.join(tmpdir, "spin_snapshots", spin_files[idx_spin]))
    
    with cols[1]:
        st.subheader("磁滞回线图")
        hyst_files = sorted(os.listdir(os.path.join(tmpdir, "hysteresis_loops")))
        idx_hyst = st.slider("温度帧", 0, len(hyst_files)-1, 0, key="hyst_slider")
        st.image(os.path.join(tmpdir, "hysteresis_loops", hyst_files[idx_hyst]))
    
    # 高级结果
    if st.session_state['advanced']:
        st.divider()
        
        # Binder比率和自旋比例
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Binder比率 $U_4$")
            st.image(os.path.join(tmpdir, "binder_ratio.png"))
        with cols[1]:
            st.subheader("自旋比例")
            st.image(os.path.join(tmpdir, "spin_ratio.png"))
        
        # 误差分析
        st.subheader("误差分析")
        tabs = st.tabs(["磁化强度", "磁化率", "Binder比率"])
        with tabs[0]:
            st.image(os.path.join(tmpdir, "error_analysis", "Magnetization_with_errorbar.png"))
        with tabs[1]:
            st.image(os.path.join(tmpdir, "error_analysis", "Chi_with_errorbar.png"))
        with tabs[2]:
            st.image(os.path.join(tmpdir, "error_analysis", "Binder_U4_with_errorbar.png"))
        
        # 局部有序度
        st.subheader("局部有序度分析")
        order_dir = os.path.join(tmpdir, "local_order")
        order_files = sorted([f for f in os.listdir(order_dir) if "Order" in f])
        selected = st.select_slider("选择温度", options=order_files, 
                                  format_func=lambda x: f"T={x.split('_')[2][1:-4]}")
        cols = st.columns(2)
        with cols[0]:
            st.image(os.path.join(order_dir, selected))
        with cols[1]:
            st.image(os.path.join(order_dir, selected.replace("Order", "Mask")))
        
        # 动态演示
        st.subheader("自旋动态演化")
        st.image(os.path.join(tmpdir, "spin_animation.gif"))
        
        st.subheader("磁滞特征分析")
        st.image(os.path.join(tmpdir, "hysteresis_features.png"))
    
    # 下载按钮
    zip_path = os.path.join(tmpdir, "results.zip")
    with ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if any(file.endswith(ext) for ext in ['.png', '.gif']):
                    zipf.write(os.path.join(root, file), 
                              os.path.relpath(os.path.join(root, file), tmpdir))
    
    with open(zip_path, "rb") as f:
        st.download_button("下载所有结果", f, "ising_results.zip")
