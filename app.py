import os
import streamlit as st
from ising_model import run_temperature_scan, run_hysteresis, compute_coercive_field_stats
from visualizer import (
    plot_magnetization_vs_temp,
    save_all_spin_snapshots,
    save_all_hysteresis_loops,
    save_final_hysteresis_snapshots,
    plot_binder_ratio_vs_temp,
    plot_coercive_field_vs_temp
)

@st.cache_data(show_spinner=True)
def simulate_and_generate(L, lattice, Ntrial, Tmin, Tmax, nT, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)

    results = run_temperature_scan(L, lattice, Ntrial, Tmin, Tmax, nT)
    T_list = [r['T'] for r in results]

    hyst_data = run_hysteresis(L, lattice, T_list, Ntrial=100)

    Hc_list, Hc_err_list = compute_coercive_field_stats(hyst_data)

    st.write(f"T_list length: {len(T_list)}")
    st.write(f"Hc_list length: {len(Hc_list)}")
    st.write(f"Hc_err_list length: {len(Hc_err_list)}")

    # 确保长度匹配，截断为最小长度
    min_len = min(len(T_list), len(Hc_list), len(Hc_err_list))
    T_list_trim = T_list[:min_len]
    Hc_list_trim = Hc_list[:min_len]
    Hc_err_list_trim = Hc_err_list[:min_len]

    # 保存图片
    plot_magnetization_vs_temp(results, save_path=os.path.join(tmpdir, "magnetization_vs_T.png"))
    plot_binder_ratio_vs_temp(results, save_path=os.path.join(tmpdir, "binder_ratio_vs_T.png"))
    plot_coercive_field_vs_temp(T_list_trim, Hc_list_trim, Hc_err_list_trim, save_path=os.path.join(tmpdir, "coercive_field_vs_T.png"))

    save_all_spin_snapshots(results, os.path.join(tmpdir, "spin_snapshots"))
    save_all_hysteresis_loops(hyst_data, os.path.join(tmpdir, "hysteresis_loops"))
    save_final_hysteresis_snapshots(hyst_data, os.path.join(tmpdir, "final_hysteresis_snapshots"))

    return results, hyst_data

def main():
    st.title("Ising Model Monte Carlo Simulation")

    L = st.sidebar.number_input("Lattice size L", min_value=4, max_value=200, value=16, step=4)
    lattice = st.sidebar.selectbox("Lattice type", ("square", "triangular"))
    Ntrial = st.sidebar.number_input("MC steps per temperature", min_value=10, max_value=10000, value=500, step=10)
    Tmin = st.sidebar.number_input("Min temperature Tmin", min_value=0.01, max_value=10.0, value=1.5, step=0.01, format="%.2f")
    Tmax = st.sidebar.number_input("Max temperature Tmax", min_value=0.01, max_value=10.0, value=3.5, step=0.01, format="%.2f")
    nT = st.sidebar.number_input("Number of temperature points", min_value=5, max_value=200, value=30, step=1)

    if Tmin >= Tmax:
        st.error("Tmin must be less than Tmax!")
        return

    tmpdir = "tmp_output"

    if st.button("Run Simulation"):
        with st.spinner("Simulating..."):
            results, hyst_data = simulate_and_generate(L, lattice, Ntrial, Tmin, Tmax, nT, tmpdir)

        st.success("Simulation completed!")

        st.subheader("Magnetization vs Temperature")
        st.image(os.path.join(tmpdir, "magnetization_vs_T.png"))

        st.subheader("Binder Cumulant vs Temperature")
        st.image(os.path.join(tmpdir, "binder_ratio_vs_T.png"))

        st.subheader("Coercive Field vs Temperature")
        st.image(os.path.join(tmpdir, "coercive_field_vs_T.png"))

        st.subheader("Sample Spin Snapshot")
        sample = results[len(results)//2]
        st.image(os.path.join(tmpdir, "spin_snapshots", f"spin_T{sample['T']:.3f}.png"))

        st.subheader("Sample Hysteresis Loop")
        sample_hyst = hyst_data[len(hyst_data)//2]
        st.image(os.path.join(tmpdir, "hysteresis_loops", f"hysteresis_T{sample_hyst['T']:.3f}.png"))

if __name__ == "__main__":
    main()
