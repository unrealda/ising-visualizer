import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def plot_magnetization_vs_temp(T, M, susceptibility, binder, lattice_type='Square'):
    fig, ax = plt.subplots(1, 3, figsize=(18,5))

    # 磁化强度
    ax[0].errorbar(T, M[:,0], yerr=M[:,1], fmt='ro-', label="磁化强度 M")
    ax[0].set_xlabel('温度 T')
    ax[0].set_ylabel('磁化强度 M')
    ax[0].set_title(f'{lattice_type} 晶格磁化强度')
    ax[0].grid(True)

    # 磁化率
    ax[1].errorbar(T, susceptibility[:,0], yerr=susceptibility[:,1], fmt='bo-', label="磁化率 χ")
    ax[1].set_xlabel('温度 T')
    ax[1].set_ylabel('磁化率 χ')
    ax[1].set_title(f'{lattice_type} 晶格磁化率')
    ax[1].grid(True)

    # Binder比率
    ax[2].errorbar(T, binder[:,0], yerr=binder[:,1], fmt='go-', label="Binder比率 U4")
    ax[2].set_xlabel('温度 T')
    ax[2].set_ylabel('Binder比率 U4')
    ax[2].set_title(f'{lattice_type} 晶格Binder比率')
    ax[2].grid(True)

    plt.tight_layout()
    return fig

def plot_cluster_distribution(sizes):
    fig, ax = plt.subplots(figsize=(8,5))
    counts, bins, patches = ax.hist(sizes, bins=30, color='purple', alpha=0.7)
    ax.set_xlabel("簇大小")
    ax.set_ylabel("频数")
    ax.set_title("簇大小分布直方图")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_hysteresis(H, M, coercivity, remanence):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(H, M, 'r-o', label="磁化强度 M")
    ax.axvline(coercivity, color='b', linestyle='--', label=f'矫顽力 Hc={coercivity:.3f}')
    ax.axhline(remanence, color='g', linestyle='--', label=f'剩余磁化 Mr={remanence:.3f}')
    ax.set_xlabel('外磁场 H')
    ax.set_ylabel('磁化强度 M')
    ax.set_title('磁滞回线')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def animate_hysteresis(H_values, M_values, lattice_states, pause=200):
    """生成磁滞回线动画，红蓝箭头表示自旋方向"""
    fig, ax = plt.subplots(figsize=(6,6))
    L = lattice_states[0].shape[0]

    def update(frame):
        ax.clear()
        ax.set_title(f'Hysteresis Loop: H={H_values[frame]:.2f}, M={M_values[frame]:.3f}')
        ax.set_xticks([])
        ax.set_yticks([])
        lattice = lattice_states[frame]
        for i in range(L):
            for j in range(L):
                if lattice[i,j] == 1:
                    ax.arrow(j, L-1-i, 0, 0.3, head_width=0.2, head_length=0.2, fc='red', ec='red')
                else:
                    ax.arrow(j, L-1-i, 0, -0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        ax.set_xlim(-0.5, L-0.5)
        ax.set_ylim(-0.5, L-0.5)

    ani = animation.FuncAnimation(fig, update, frames=len(H_values), interval=pause)
    return ani
