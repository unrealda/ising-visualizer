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

if run_button:
    with st.spinner("正在运行模拟，请稍候..."):
        tmpdir = tempfile.mkdtemp()

        # 运行模拟
        results = run_temperature_scan(L, lattice, Ntrial, Tmin, Tmax, nT)
        T_list = [r['T'] for r in results]
        hyst_data = run_hysteresis(L, lattice, T_list, Ntrial=100)

        # 输出图像
        plot_magnetization_vs_temp(results, save_path=os.path.join(tmpdir, "magnetization_vs_T.png"))
        save_all_spin_snapshots(results, os.path.join(tmpdir, "spin_snapshots"))
        save_all_hysteresis_loops(hyst_data, os.path.join(tmpdir, "hysteresis_loops"))
        save_final_hysteresis_snapshots(hyst_data, os.path.join(tmpdir, "final_hyst_frames"))

        # 展示主要图像
        st.subheader("磁化率与温度关系图")
        st.image(os.path.join(tmpdir, "magnetization_vs_T.png"), use_column_width=True)

        st.subheader("自旋分布图 (每温度一帧)")
        for f in sorted(os.listdir(os.path.join(tmpdir, "spin_snapshots"))):
            st.image(os.path.join(tmpdir, "spin_snapshots", f), caption=f)

        st.subheader("磁滞回线图")
        for f in sorted(os.listdir(os.path.join(tmpdir, "hysteresis_loops"))):
            st.image(os.path.join(tmpdir, "hysteresis_loops", f), caption=f)

        st.subheader("最终温度下的磁滞过程帧")
        for f in sorted(os.listdir(os.path.join(tmpdir, "final_hyst_frames"))):
            st.image(os.path.join(tmpdir, "final_hyst_frames", f), caption=f)

        # 打包为 zip
        zip_path = os.path.join(tmpdir, "ising_results.zip")
        with ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(".png"):
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, tmpdir)
                        zipf.write(abs_path, arcname=rel_path)

        with open(zip_path, "rb") as f:
            st.download_button("\U0001F4E5 下载所有图像 (ZIP)", f, file_name="ising_results.zip")

        shutil.rmtree(tmpdir)
