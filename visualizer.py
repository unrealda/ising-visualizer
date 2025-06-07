import matplotlib.pyplot as plt
import numpy as np
import os
def plot_magnetization_vs_temp(results, save_path=None):
    T_list = [r['T'] for r in results]
    M = [r['M'] for r in results]
    Mvar = [r['Mvar'] for r in results]
    Chi = [r['chi'] for r in results]
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.errorbar(T_list, M, yerr=np.sqrt(Mvar), fmt='x--', color='blue', label='Magnetization')
    ax2.plot(T_list, Chi, 'o-r', label='Susceptibility ($\chi$)')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Magnetization', color='blue')
    ax2.set_ylabel('Susceptibility $\chi$', color='red')
    ax1.grid(True)
    chi_peak = max(Chi)
    idx_peak = Chi.index(chi_peak)
    ax2.plot(T_list[idx_peak], chi_peak, 'kp', markerfacecolor='green')
    ax2.text(T_list[idx_peak] + 0.1, chi_peak, f'Peak at T = {T_list[idx_peak]:.2f}', color='green')
    plt.title("Magnetization and Susceptibility vs Temperature")
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_spin_snapshot(spin_matrix, T, save_path=None):
    cmap = plt.get_cmap('bwr')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(spin_matrix, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Spin Configuration at T = {T:.2f}")
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_hysteresis_loop(H_vals, M_vals, T, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(H_vals, M_vals, 'o-', lw=1.5)
    ax.set_xlabel('External Field H')
    ax.set_ylabel('Magnetization')
    ax.set_title(f"Hysteresis Loop at T = {T:.2f}")
    ax.grid(True)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def save_all_spin_snapshots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for res in results:
        spin_matrix = res['spins_snapshot']
        T = res['T']
        fname = os.path.join(output_dir, f'spin_T{T:.3f}.png')
        plot_spin_snapshot(spin_matrix, T, fname)

def save_all_hysteresis_loops(hyst_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for h in hyst_data:
        plot_hysteresis_loop(h['H_vals'], h['M_vals'], h['T'],
                             os.path.join(output_dir, f'hysteresis_T{h["T"]:.3f}.png'))

def save_final_hysteresis_snapshots(hyst_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    last = hyst_data[-1]
    for i, frame in enumerate(last['final_frames']):
        fname = os.path.join(output_dir, f'final_hyst_frame_{i:03d}.png')
        plot_spin_snapshot(frame, T=last['T'], save_path=fname)
