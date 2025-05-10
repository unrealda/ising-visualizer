import streamlit as st
import tempfile
import shutil
import os
from ising_model import run_temperature_scan, run_hysteresis
from visualizer import (
    plot_magnetization_vs_temp,
    save_all_spin_snapshots,
    save_all_hysteresis_loops,
    save_final_hysteresis_snapshots
)
from zipfile import ZipFile

st.set_page_config(page_title="Ising Model (Wolff Algorithm)", layout="wide")
st.title("\U0001F9BE Wolff 算法模拟二维伊辛模型")

with st.sidebar:
    st.header("参数设置")
    L = st.number_input("格子边长 L", min_value=4, max_value=128, value=10)
    lattice = st.selectbox("晶格类型", ["square", "triangular"])
    Ntrial = st.number_input("每温度试验次数", min_value=10, max_value=1000, value=100)
    Tmin = st.number_input("最低温度 Tmin", min_value=0.1, value=1.0, step=0.1)
    Tmax = st.number_input("最高温度 Tmax", min_value=0.1, value=3.5, step=0.1)
    nT = st.number_input("温度步数", min_value=2, max_value=100, value=10)

    run_button = st.button("开始模拟")

if run_button or 'results' not in st.session_state:
    with st.spinner("正在运行模拟，请稍候..."):
        tmpdir = tempfile.mkdtemp()

        # 运行模拟
        results = run_temperature_scan(L, lattice, Ntrial, Tmin, Tmax, nT)
        T_list = [r['T'] for r in results]
        hyst_data = run_hysteresis(L, lattice, T_list, Ntrial=100)

        # 输出图像
        plot_magnetization_vs_temp(results, save_path=os.path.join(tmpdir, "magnetization_vs_T.png"))
        spin_dir = os.path.join(tmpdir, "spin_snapshots")
        hyst_dir = os.path.join(tmpdir, "hysteresis_loops")
        final_dir = os.path.join(tmpdir, "final_hyst_frames")

        save_all_spin_snapshots(results, spin_dir)
        save_all_hysteresis_loops(hyst_data, hyst_dir)
        save_final_hysteresis_snapshots(hyst_data, final_dir)

        # 保存至 session_state
        st.session_state['results'] = results
        st.session_state['hyst_data'] = hyst_data
        st.session_state['tmpdir'] = tmpdir
        st.session_state['spin_dir'] = spin_dir
        st.session_state['hyst_dir'] = hyst_dir
        st.session_state['final_dir'] = final_dir

# 使用缓存的图像
results = st.session_state['results']
hyst_data = st.session_state['hyst_data']
tmpdir = st.session_state['tmpdir']
spin_dir = st.session_state['spin_dir']
hyst_dir = st.session_state['hyst_dir']
final_dir = st.session_state['final_dir']

# 图 1: 磁化率曲线
st.subheader("磁化率与温度关系图")
st.image(os.path.join(tmpdir, "magnetization_vs_T.png"), use_column_width=True)

# 图 2: 箭头图（温度）
st.subheader("\u2191/\u2193 自旋分布图（温度滑动预览）")
spin_files = sorted(os.listdir(spin_dir))
idx_spin = st.slider("选择温度帧 (箭头图)", 0, len(spin_files) - 1, 0)
st.image(os.path.join(spin_dir, spin_files[idx_spin]), caption=spin_files[idx_spin])

# 图 3: 磁滞图
st.subheader("磁滞回线图（温度滑动预览）")
hyst_files = sorted(os.listdir(hyst_dir))
idx_hyst = st.slider("选择温度帧 (磁滞图)", 0, len(hyst_files) - 1, 0)
st.image(os.path.join(hyst_dir, hyst_files[idx_hyst]), caption=hyst_files[idx_hyst])

# 图 4: 最终温度逐帧磁滞图
st.subheader("最终温度下磁滞过程形成图")
final_files = sorted(os.listdir(final_dir))
idx_final = st.slider("选择帧 (最终温度磁滞形成)", 0, len(final_files) - 1, 0)
st.image(os.path.join(final_dir, final_files[idx_final]), caption=final_files[idx_final])

# 下载按钮
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
