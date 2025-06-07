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

def generate_local_analysis(spin_matrix, T, save_dir):
    """移植MATLAB的局部有序度三图生成"""
    kernel = np.ones((3,3))
    local_sum = convolve2d(spin_matrix, kernel, mode='same', boundary='wrap')
    local_order = local_sum / 9
    # 热图
    plt.figure(figsize=(5,5))
    plt.imshow(local_order, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Local Order at T={T:.2f}")
    plt.savefig(os.path.join(save_dir, f'Local_Order_T{T:.3f}.png'))
    plt.close()
    # 高有序区域掩膜
    threshold = 0.8
    mask = np.abs(local_order) >= threshold
    plt.figure(figsize=(5,5))
    plt.imshow(mask, cmap='gray')
    plt.title(f"Ordered Domains (|M|≥{threshold})")
    plt.savefig(os.path.join(save_dir, f'Local_Mask_T{T:.3f}.png'))
    plt.close()
    # 返回平均局部有序度 (用于曲线绘制)
    return np.mean(np.abs(local_order))

def plot_hysteresis_features(T_list, features, save_path):
    """移植MATLAB的plot_hysteresis_features函数"""
    # 数据平滑处理 (与MATLAB的smooth函数一致)
    from scipy.interpolate import make_interp_spline
    T_smooth = np.linspace(min(T_list), max(T_list), 100)
    
    # 回线面积
    A_smooth = make_interp_spline(T_list, features['A_hyst'])(T_smooth)
    plt.figure(figsize=(8,5))
    plt.plot(T_list, features['A_hyst'], 'bo-', label='Raw')
    plt.plot(T_smooth, A_smooth, 'r-', lw=2, label='Smoothed')
    plt.xlabel('Temperature')
    plt.ylabel('Hysteresis Area')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Hysteresis_Area_vs_T.png'))
    plt.close()
    
    # 剩余磁化 (同样处理其他两个图)
    # ...类似实现...

def create_spin_animation(spin_matrices, T_list, save_path):
    """移植MATLAB的make_arrow_animation_1逻辑"""
    fig, ax = plt.subplots(figsize=(8,8))
    plt.axis('off')
    
    frames = []
    for i, (spins, T) in enumerate(zip(spin_matrices, T_list)):
        ax.clear()
        up_count = np.sum(spins > 0)
        down_count = spins.size - up_count
        
        # 绘制箭头 (与MATLAB完全一致)
        for y in range(spins.shape[0]):
            for x in range(spins.shape[1]):
                if spins[y,x] > 0:
                    ax.text(x, y, r'$\uparrow$', ha='center', va='center', 
                           color='r', fontsize=14)
                else:
                    ax.text(x, y, r'$\downarrow$', ha='center', va='center',
                           color='b', fontsize=14)
        
        # 添加统计信息 (位置计算与MATLAB一致)
        ax.text(0.1, 1.05, f'T = {T:.3f}', transform=ax.transAxes,
                fontsize=13, fontweight='bold')
        ax.text(0.5, 1.05, f'↑: {up_count}', transform=ax.transAxes,
                color='r', fontsize=13, fontweight='bold')
        ax.text(0.7, 1.05, f'↓: {down_count}', transform=ax.transAxes,
                color='b', fontsize=13, fontweight='bold')
        
        # 生成帧
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
    
    # 保存GIF (参数与MATLAB一致)
    imageio.mimsave(save_path, frames, duration=300, loop=0)
