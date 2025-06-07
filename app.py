import streamlit as st
import uuid
import os
import shutil
from zipfile import ZipFile

from ising_model import run_temperature_scan, run_hysteresis, compute_coercive_field_stats
from visualizer import (
    plot_magnetization_vs_temp,
    save_all_spin_snapshots,
    save_all_hysteresis_loops,
    save_final_hysteresis_snapshots,
    plot_hysteresis_loop,
    plot_binder_cumulant_vs_temp,
    plot_coercive_field_vs_temp,
)

st.set_page_config(page_title="Ising Model (Wolff Algorithm)", layout="wide")
st.title("\U0001F9BE Wolff 算法模拟二维伊辛模型")

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
    st.session_state['has_run'] = False

base_cache_dir = ".streamlit_cache"
os.makedirs(base_cache_dir, exist_ok=True)
tmpdir = os.path.join(base_cache_dir, st.session_state['session_id'])

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
    run_button = st.button("开始模拟")

    if st.button("清空缓存并重置"):
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        st.session_state.clear()
        st.success("缓存已清除，请手动刷新页面或重新开始模拟。")

@st.cache_data(show_spinner=False)
def simulate_and_generate(L, lattice, Ntrial, Tmin, Tmax, nT, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)

    # 运行温度扫描，包含Binder统计
    results = run_temperature_scan(L, lattice, Ntrial, Tmin, Tmax, nT)
    T_list = [r['T'] for r in results]

    # 运行磁滞回线模拟
    hyst_data = run_hysteresis(L, lattice, T_list, Ntrial=100)

    # 绘图并保存
    plot_magnetization_vs_temp(results, save_path=os.path.join(tmpdir, "magnetization_vs_T.png"))
    plot_binder_cumulant_vs_temp(results, save_path=os.path.join(tmpdir, "binder_cumulant_vs_T.png"))

    Hc_list, Hc_err_list = compute_coercive_field_stats(hyst_data)
    plot_coercive_field_vs_temp(T_list, Hc_list, Hc_err_list, save_path=os.path.join(tmpdir, "coercive_field_vs_T.png"))

    spin_dir = os.path.join(tmpdir, "spin_snapshots")
    hyst_dir = os.path.join(tmpdir, "hysteresis_loops")
    final_dir = os.path.join(tmpdir, "final_hyst_frames")
    final_hyst_plot_dir = os.path.join(tmpdir, "final_hyst_plot_frames")

    save_all_spin_snapshots(results, spin_dir)
    save_all_hysteresis_loops(hyst_data, hyst_dir)
    save_final_hysteresis_snapshots(hyst_data, final_dir)

    os.makedirs(final_hyst_plot_dir, exist_ok=True)
    final = hyst_data[-1]
    H_vals = final['H_vals']
    M_vals = final['M_vals']
    for i in range(1, len(H_vals) + 1):
        plot_hysteresis_loop(H_vals[:i], M_vals[:i], final['T'], save_path=os.path.join(final_hyst_plot_dir, f'frame_{i:03d}.png'))

    return results, hyst_data

if run_button:
    with st.spinner("正在运行模拟，请稍候..."):
        results, hyst_data = simulate_and_generate(L, lattice, Ntrial, Tmin, Tmax, nT, tmpdir)
        st.session_state['has_run'] = True

if st.session_state.get('has_run', False):
    # 磁化率曲线
    st.subheader("磁化率与温度关系图")
    st.image(os.path.join(tmpdir, "magnetization_vs_T.png"), use_container_width=True)

    # Binder比率曲线
    st.subheader("Binder 比率与温度关系图")
    st.image(os.path.join(tmpdir, "binder_cumulant_vs_T.png"), use_container_width=True)

    # 矫顽力曲线
    st.subheader("矫顽力与温度关系图")
    st.image(os.path.join(tmpdir, "coercive_field_vs_T.png"), use_container_width=True)

    # 自旋分布图（温度滑动预览）
    st.subheader("\u2191/\u2193 自旋分布图（温度滑动预览）")
    spin_dir = os.path.join(tmpdir, "spin_snapshots")
    spin_files = sorted(os.listdir(spin_dir))
    idx_spin = st.slider("选择温度帧 (箭头图)", 0, len(spin_files) - 1, 0)
    st.image(os.path.join(spin_dir, spin_files[idx_spin]), caption=spin_files[idx_spin])

    # 磁滞回线图（温度滑动预览）
    st.subheader("磁滞回线图（温度滑动预览）")
    hyst_dir = os.path.join(tmpdir, "hysteresis_loops")
    hyst_files = sorted(os.listdir(hyst_dir))
    idx_hyst = st.slider("选择温度帧 (磁滞图)", 0, len(hyst_files) - 1, 0)
    st.image(os.path.join(hyst_dir, hyst_files[idx_hyst]), caption=hyst_files[idx_hyst])

    # 最终温度下磁滞过程形成图（双图）
    st.subheader("最终温度下磁滞过程形成图")
    final_dir = os.path.join(tmpdir, "final_hyst_frames")
    final_hyst_plot_dir = os.path.join(tmpdir, "final_hyst_plot_frames")
    final_spin_files = sorted(os.listdir(final_dir))
    final_plot_files = sorted(os.listdir(final_hyst_plot_dir))
    idx_final = st.slider("选择帧 (最终温度磁滞形成)", 0, len(final_spin_files) - 1, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(final_dir, final_spin_files[idx_final]), caption="自旋图帧")
    with col2:
        st.image(os.path.join(final_hyst_plot_dir, final_plot_files[idx_final]), caption="磁滞回线帧")

    # 下载所有图像按钮
    zip_path = os.path.join(tmpdir, "ising_results.zip")
    if not os.path.exists(zip_path):
        with ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(".png"):
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, tmpdir)
                        zipf.write(abs_path, arcname=rel_path)

    with open(zip_path, "rb") as f:
        st.download_button("\U0001F4E5 下载所有图像 (ZIP)", f, file_name="ising_results.zip")
