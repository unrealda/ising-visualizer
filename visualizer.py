import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import convolve2d
import imageio

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

def generate_local_analysis(spin_matrix, T, save_dir):
    kernel = np.ones((3,3))
    local_sum = convolve2d(spin_matrix, kernel, mode='same', boundary='wrap')
    local_order = local_sum / 9
    plt.figure(figsize=(5,5))
    plt.imshow(local_order, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Local Order at T={T:.2f}")
    plt.savefig(os.path.join(save_dir, f'Local_Order_T{T:.3f}.png'))
    plt.close()
    threshold = 0.8
    mask = np.abs(local_order) >= threshold
    plt.figure(figsize=(5,5))
    plt.imshow(mask, cmap='gray')
    plt.title(f"Ordered Domains (|M|≥{threshold})")
    plt.savefig(os.path.join(save_dir, f'Local_Mask_T{T:.3f}.png'))
    plt.close()
    return np.mean(np.abs(local_order))

def plot_binder_ratio(T_list, M2_list, M4_list, M2_var, M4_var, save_path=None):
    binder_ratio = 1 - np.array(M4_list) / (3 * np.array(M2_list)**2)
    error = np.sqrt(
        (np.array(M4_var)/(3*np.array(M2_list)**2))**2 +
        (2*np.array(M4_list)*np.array(M2_var)/(3*np.array(M2_list)**3))**2
    )
    plt.figure(figsize=(8,5))
    plt.errorbar(T_list, binder_ratio, yerr=error, fmt='o-', color='purple', capsize=4)
    plt.xlabel('Temperature')
    plt.ylabel('Binder Ratio')
    plt.title('Binder Ratio vs Temperature')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_spin_ratio(T_list, up_ratio_list, down_ratio_list, save_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(T_list, up_ratio_list, 'r^-', label='Up Spins')
    plt.plot(T_list, down_ratio_list, 'bv-', label='Down Spins')
    plt.xlabel('Temperature')
    plt.ylabel('Spin Ratio')
    plt.title('Up/Down Spin Ratio vs Temperature')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_hysteresis_features(T_list, A_hyst_list, Hc_list, Hc_var, save_path=None):
    plt.figure(figsize=(8,5))
    plt.errorbar(T_list, Hc_list, yerr=np.sqrt(Hc_var), fmt='o-', color='green', capsize=4, label='Coercive Field Hc')
    plt.xlabel('Temperature')
    plt.ylabel('Coercive Field')
    plt.title('Coercive Field vs Temperature')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'Coercive_Field_vs_T.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(T_list, A_hyst_list, 'o-', color='orange', label='Hysteresis Area')
    plt.xlabel('Temperature')
    plt.ylabel('Hysteresis Area')
    plt.title('Hysteresis Area vs Temperature')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'Hysteresis_Area_vs_T.png'), bbox_inches='tight')
    plt.close()

def create_spin_animation(spin_matrices, T_list, save_path):
    fig, ax = plt.subplots(figsize=(8,8))
    plt.axis('off')
    frames = []
    for i, (spins, T) in enumerate(zip(spin_matrices, T_list)):
        ax.clear()
        up_count = np.sum(spins > 0)
        down_count = spins.size - up_count
        for y in range(spins.shape[0]):
            for x in range(spins.shape[1]):
                if spins[y,x] > 0:
                    ax.text(x, y, r'$\uparrow$', ha='center', va='center', 
                           color='r', fontsize=14)
                else:
                    ax.text(x, y, r'$\downarrow$', ha='center', va='center',
                           color='b', fontsize=14)
        ax.text(0.1, 1.05, f'T = {T:.3f}', transform=ax.transAxes,
                fontsize=13, fontweight='bold')
        ax.text(0.5, 1.05, f'↑: {up_count}', transform=ax.transAxes,
                color='r', fontsize=13, fontweight='bold')
        ax.text(0.7, 1.05, f'↓: {down_count}', transform=ax.transAxes,
                color='b', fontsize=13, fontweight='bold')
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
    imageio.mimsave(save_path, frames, duration=0.3, loop=0)
