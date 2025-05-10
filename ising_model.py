import numpy as np

# 运行伊辛模型模拟（带有Wolff算法）
def run_ising_simulation(L, Tmin, Tmax, nT, Ntrial, folder):
    N = L * L
    T_list = np.linspace(Tmin, Tmax, nT)
    M_list = []
    Chi_list = []
    spin_snapshots = []

    for T in T_list:
        beta = 1.0 / T
        p = 1 - np.exp(-2 * beta)
        S = np.random.choice([-1, 1], size=N)  # 初始自旋状态

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

            # 翻转自旋
            for site in cluster:
                S[site] *= -1

            magnetizations.append(np.sum(S))

        M_avg = np.mean(np.abs(magnetizations)) / N
        M2_avg = np.mean(np.array(magnetizations) ** 2) / (N ** 2)
        Chi = (N / T) * (M2_avg - M_avg ** 2)

        M_list.append(M_avg)
        Chi_list.append(Chi)

        # 存储二维自旋快照
        S2D = S.reshape((L, L))
        spin_snapshots.append(S2D)

        # 保存快照到文件
        np.savetxt(f"{folder}/snapshot_T{T:.3f}.dat", S2D, fmt='%d')

    return T_list, spin_snapshots, M_list, Chi_list

# 获取某个点的邻居
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

# 运行磁滞回线模拟
def run_hysteresis_simulation(L, Tmin, Tmax, nT, Ntrial, folder):
    """
    模拟磁滞回线
    """
    N = L * L
    T_list = np.linspace(Tmin, Tmax, nT)
    M_list_up = []
    M_list_down = []
    spin_snapshots_up = []
    spin_snapshots_down = []

    for T in T_list:
        beta = 1.0 / T
        p = 1 - np.exp(-2 * beta)
        S = np.random.choice([-1, 1], size=N)  # 初始自旋状态

        magnetizations_up = []
        magnetizations_down = []

        # 向上过程
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

            # 翻转自旋
            for site in cluster:
                S[site] *= -1

            magnetizations_up.append(np.sum(S))

        M_avg_up = np.mean(np.abs(magnetizations_up)) / N
        M_list_up.append(M_avg_up)

        # 向下过程（磁场反转）
        S = np.random.choice([-1, 1], size=N)  # 重置初始自旋状态
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

            # 翻转自旋
            for site in cluster:
                S[site] *= -1

            magnetizations_down.append(np.sum(S))

        M_avg_down = np.mean(np.abs(magnetizations_down)) / N
        M_list_down.append(M_avg_down)

        # 存储快照
        S2D_up = S.reshape((L, L))
        spin_snapshots_up.append(S2D_up)
        np.savetxt(f"{folder}/snapshot_up_T{T:.3f}.dat", S2D_up, fmt='%d')

        S2D_down = -S.reshape((L, L))
        spin_snapshots_down.append(S2D_down)
        np.savetxt(f"{folder}/snapshot_down_T{T:.3f}.dat", S2D_down, fmt='%d')

    return T_list, spin_snapshots_up, M_list_up, spin_snapshots_down, M_list_down
