import matplotlib.pyplot as plt
import numpy as np
import os

def plot_magnetization_vs_temp(results, save_path=None):
    """
    绘制磁化强度与磁化率随温度变化，支持保存到文件。
    
    results: List[dict]，每个dict包含'T', 'M', 'Mvar', 'Chi'等字段
    save_path: str或None，指定保存路径，None则不保存
    """
    T = np.array([r['T'] for r in results])
    M = np.array([r['M'] for r in results])
    Mvar = np.array([r['Mvar'] for r in results])
    Chi = np.array([r['Chi'] for r in results])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(T, M, yerr=np.sqrt(Mvar), fmt='o-', color='tab:blue', label='磁化强度 M')
    ax1.set_xlabel('温度 T')
    ax1.set_ylabel('磁化强度 M', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(T, Chi, 'r--o', label='磁化率 χ')
    ax2.set_ylabel('磁化率 χ', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('磁化强度与磁化率随温度变化')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def save_all_spin_snapshots(results, out_dir):
    """
    生成所有温度对应的自旋分布箭头图并保存
    
    results: List[dict]，每个dict至少含 'T', 'spin_config' (2D np.array)
    out_dir: 输出文件夹路径
    """
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        T = r['T']
        spins = r['spin_config']  # 2D numpy array
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(spins, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f'自旋分布图 T={T:.2f}')
        ax.axis('off')
        plt.tight_layout()
        filepath = os.path.join(out_dir, f'spin_T_{T:.2f}.png')
        plt.savefig(filepath, dpi=150)
        plt.close(fig)

def save_all_hysteresis_loops(hyst_data, out_dir):
    """
    生成所有温度对应的磁滞回线图并保存
    
    hyst_data: List[dict]，每个dict包含 'T', 'H_vals', 'M_vals'
    out_dir: 输出文件夹路径
    """
    os.makedirs(out_dir, exist_ok=True)
    for h in hyst_data:
        T = h['T']
        H = h['H_vals']
        M = h['M_vals']
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(H, M, 'o-', linewidth=1.5)
        ax.set_title(f'磁滞回线 T={T:.2f}')
        ax.set_xlabel('外部磁场 H')
        ax.set_ylabel('磁化强度 M')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True)
        plt.tight_layout()
        filepath = os.path.join(out_dir, f'hysteresis_T_{T:.2f}.png')
        plt.savefig(filepath, dpi=150)
        plt.close(fig)

def save_final_hysteresis_snapshots(hyst_data, out_dir):
    """
    保存最终温度下磁滞过程每步的自旋分布（假设数据已包含逐帧spin_config）
    
    hyst_data: List[dict]，最后一个dict应包含 'spin_frames': List[np.array]
    out_dir: 保存路径
    """
    os.makedirs(out_dir, exist_ok=True)
    final_data = hyst_data[-1]
    spin_frames = final_data.get('spin_frames', [])
    for i, spins in enumerate(spin_frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(spins, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f'最终温度磁滞自旋帧 {i+1}')
        ax.axis('off')
        plt.tight_layout()
        filepath = os.path.join(out_dir, f'final_spin_frame_{i+1:03d}.png')
        plt.savefig(filepath, dpi=150)
        plt.close(fig)

def plot_hysteresis_loop(H_vals, M_vals, T, save_path=None):
    """
    单步绘制磁滞回线进展图，方便做成动画帧
    
    H_vals, M_vals: 当前进度数据数组
    T: 当前温度
    save_path: 若指定，则保存到文件
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(H_vals, M_vals, 'o-', linewidth=1.5, color='purple')
    ax.set_title(f'磁滞回线进展 T={T:.2f}')
    ax.set_xlabel('外部磁场 H')
    ax.set_ylabel('磁化强度 M')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
