import numpy as np
import random

def build_neighbors(L, lattice_type="square"):
    N = L * L
    neighbors = [[] for _ in range(N)]

    def index(row, col):
        return (row % L) * L + (col % L)

    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if lattice_type == "square":
                nbrs = [
                    index(i, j + 1),
                    index(i + 1, j),
                    index(i, j - 1),
                    index(i - 1, j),
                ]
            elif lattice_type == "triangular":
                nbrs = [
                    index(i, j + 1),
                    index(i + 1, j),
                    index(i, j - 1),
                    index(i - 1, j),
                    index(i + 1, j - 1),
                    index(i - 1, j + 1),
                ]
            else:
                raise ValueError("Invalid lattice type")
            neighbors[idx] = nbrs
    return np.array(neighbors, dtype=object)

def wolff_update(spins, neighbors, beta, H=0.0):
    N = len(spins)
    visited = np.zeros(N, dtype=bool)
    seed = random.randint(0, N - 1)
    cluster_spin = spins[seed]

    # 建议修改p_add去掉外场影响，更符合物理
    p_add = 1 - np.exp(-2 * beta)

    cluster = [seed]
    stack = [seed]
    visited[seed] = True

    while stack:
        current = stack.pop()
        for nbr in neighbors[current]:
            if not visited[nbr] and spins[nbr] == cluster_spin:
                if random.random() < p_add:
                    visited[nbr] = True
                    cluster.append(nbr)
                    stack.append(nbr)

    spins[visited] *= -1
    return spins, len(cluster), np.sum(spins)

def run_temperature_scan(L, lattice_type, Ntrial, Tmin, Tmax, nT):
    N = L * L
    neighbors = build_neighbors(L, lattice_type)
    T_list = np.linspace(Tmin, Tmax, nT)
    results = []

    for T in T_list:
        beta = 1.0 / T
        spins = np.random.choice([-1, 1], size=N)
        magnetizations = np.zeros(Ntrial)
        cluster_sizes = np.zeros(Ntrial)

        for i in range(Ntrial):
            spins, clust_size, mag = wolff_update(spins, neighbors, beta)
            magnetizations[i] = mag
            cluster_sizes[i] = clust_size

        M = np.mean(np.abs(magnetizations)) / N
        M2 = np.mean(magnetizations ** 2) / N**2
        Mvar = np.var(magnetizations / N) / N
        chi = (N / T) * (M2 - M**2)

        # 新增计算Binder比率及误差
        U4, U4_err = compute_binder_cumulant(magnetizations / N)

        results.append({
            'T': T,
            'M': M,
            'Mvar': Mvar,
            'chi': chi,
            'mean_cluster_size': np.mean(cluster_sizes),
            'binder_cumulant': U4,
            'binder_error': U4_err,
            'spins_snapshot': spins.copy().reshape((L, L))
        })

    return results

def run_hysteresis(L, lattice_type, T_list, Ntrial=100):
    N = L * L
    neighbors = build_neighbors(L, lattice_type)
    hysteresis_data = []

    for T in T_list:
        beta = 1 / T
        record_final = abs(T - T_list[-1]) < 1e-8
        final_gif_frames = []

        if abs(T - T_list[np.argmax(T_list)]) < 0.3:
            H_vals = list(np.arange(-1, 1.05, 0.05)) + list(np.arange(0.95, -1.05, -0.05))
        else:
            H_vals = list(np.arange(-1, 1.2, 0.2)) + list(np.arange(0.8, -1.2, -0.2))

        spins = np.ones(N, dtype=int)
        M_H = []

        for H in H_vals:
            for _ in range(Ntrial):
                spins, _, _ = wolff_update(spins, neighbors, beta, H)
            M_H.append(np.sum(spins) / N)

            if record_final:
                final_gif_frames.append(spins.copy().reshape((L, L)))

        hysteresis_data.append({
            'T': T,
            'H_vals': H_vals,
            'M_vals': M_H,
            'final_frames': final_gif_frames if record_final else None
        })

    return hysteresis_data

def compute_binder_cumulant(magnetizations):
    """
    输入：磁化样本数组（归一化）
    输出：Binder比率及bootstrap估计误差
    """
    M2 = np.mean(magnetizations**2)
    M4 = np.mean(magnetizations**4)
    U4 = 1 - M4 / (3 * M2**2)

    nboot = 1000
    boot_vals = []
    N = len(magnetizations)
    for _ in range(nboot):
        sample = np.random.choice(magnetizations, size=N, replace=True)
        M2b = np.mean(sample**2)
        M4b = np.mean(sample**4)
        U4b = 1 - M4b / (3 * M2b**2)
        boot_vals.append(U4b)
    U4_err = np.std(boot_vals)

    return U4, U4_err

def find_coercive_field(H_vals, M_vals):
    """
    从磁滞回线中线性插值找到M=0的H值（矫顽力）
    """
    for i in range(len(M_vals)-1):
        if M_vals[i] * M_vals[i+1] < 0:
            H1, H2 = H_vals[i], H_vals[i+1]
            M1, M2 = M_vals[i], M_vals[i+1]
            Hc = H1 - M1 * (H2 - H1) / (M2 - M1)
            return Hc
    return None

def compute_coercive_field_stats(hysteresis_data, T_index=0):
    """
    输入：磁滞回线数据列表及温度索引
    输出：平均矫顽力和误差估计（示例基于单组数据）
    """
    Hc_list = []
    data = hysteresis_data[T_index]
    H_vals = data['H_vals']
    M_vals = data['M_vals']

    Hc = find_coercive_field(H_vals, M_vals)
    if Hc is not None:
        Hc_list.append(Hc)

    if len(Hc_list) > 0:
        mean_Hc = np.mean(Hc_list)
        std_Hc = np.std(Hc_list)
    else:
        mean_Hc, std_Hc = None, None

    return mean_Hc, std_Hc
