import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def get_neighbors(index, L):
    row = index // L
    col = index % L
    neighbors = [
        ((row - 1) % L) * L + col,  # 上
        ((row + 1) % L) * L + col,  # 下
        row * L + (col - 1) % L,    # 左
        row * L + (col + 1) % L     # 右
    ]
    return neighbors

def run_ising_simulation(L, Tmin, Tmax, nT, Ntrial, folder):
    N = L * L
    T_list = np.linspace(Tmin, Tmax, nT)
    M_list = []
    Chi_list = []
    spin_snapshots = []

    for T in T_list:
        beta = 1.0 / T
        p = 1 - np.exp(-2 * beta)
        S = np.random.choice([-1, 1], size=N)

        magnetizations = []

        for _ in range(Ntrial):
            k = np.random.randint(N)
            cluster = [k]
            pocket = [k]

            while pocket:
                s = pocket.pop(np.random.randint(len(pocket)))
                neighbors = get_neighbors(s, L)
                for n in neighbors:
                    if S[n] == S[s] and n not in cluster:
                        if np.random.rand() < p:
                            cluster.append(n)
                            pocket.append(n)

            for site in cluster:
                S[site] *= -1

            magnetizations.append(np.sum(S))

        M_avg = np.mean(np.abs(magnetizations)) / N
        M2_avg = np.mean(np.array(magnetizations) ** 2) / (N ** 2)
        Chi = (N / T) * (M2_avg - M_avg ** 2)

        M_list.append(M_avg)
        Chi_list.append(Chi)

        S2D = S.reshape((L, L))
        spin_snapshots.append(S2D)

        np.savetxt(f"{folder}/snapshot_T{T:.3f}.dat", S2D, fmt='%d')

    return T_list, spin_snapshots, M_list, Chi_list

def run_hysteresis_simulation(L, Tmin, Tmax, nT, Ntrial, folder):
    N = L * L
    T_list = np.linspace(Tmin, Tmax, nT)
    final_T = T_list[-1]
    beta = 1.0 / final_T
    p = 1 - np.exp(-2 * beta)

    field_range = np.linspace(-1.0, 1.0, 30)
    magnetization_list = []

    S = np.random.choice([-1, 1], size=N)
    snapshots = []

    for H in np.concatenate([field_range, field_range[::-1]]):
        magnetizations = []

        for _ in range(Ntrial):
            k = np.random.randint(N)
            cluster = [k]
            pocket = [k]

            while pocket:
                s = pocket.pop(np.random.randint(len(pocket)))
                neighbors = get_neighbors(s, L)
                for n in neighbors:
                    if S[n] == S[s] and n not in cluster:
                        if np.random.rand() < p:
                            cluster.append(n)
                            pocket.append(n)

            for site in cluster:
                S[site] *= -1

            magnetizations.append(np.sum(S))

        M_avg = np.mean(magnetizations) / N
        magnetization_list.append((H, M_avg))

        S2D = S.reshape((L, L))
        snapshots.append(S2D.copy())

    # 生成 GIF
    gif_path = os.path.join(folder, "hysteresis.gif")
    images = []

    for snap in snapshots:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(snap, cmap='bwr', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        buf = os.path.join(folder, "temp.png")
        fig.savefig(buf)
        images.append(imageio.v2.imread(buf))
        plt.close(fig)

    imageio.mimsave(gif_path, images, duration=0.2)
    return gif_path
