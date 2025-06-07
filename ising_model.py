import numpy as np
from collections import deque

def initialize_lattice(L, lattice_type='Square'):
    """初始化自旋晶格，随机±1"""
    lattice = np.random.choice([-1, 1], size=(L, L))
    return lattice

def get_neighbors(lattice, i, j, lattice_type='Square'):
    """返回指定点邻居自旋"""
    L = lattice.shape[0]
    neighbors = []
    if lattice_type == 'Square':
        neighbors.extend([
            lattice[(i+1)%L, j],
            lattice[(i-1)%L, j],
            lattice[i, (j+1)%L],
            lattice[i, (j-1)%L]
        ])
    elif lattice_type == 'Triangular':
        neighbors.extend([
            lattice[(i+1)%L, j],
            lattice[(i-1)%L, j],
            lattice[i, (j+1)%L],
            lattice[i, (j-1)%L],
            lattice[(i+1)%L, (j-1)%L],
            lattice[(i-1)%L, (j+1)%L]
        ])
    return neighbors

def delta_energy(lattice, i, j, H=0.0, lattice_type='Square', J=1):
    """计算翻转某点自旋引起的能量变化"""
    spin = lattice[i, j]
    neighbors = get_neighbors(lattice, i, j, lattice_type)
    dE = 2 * spin * (J * sum(neighbors) + H)
    return dE

def monte_carlo_step(lattice, T, H=0.0, lattice_type='Square', J=1):
    """单步蒙特卡洛 Metropolis 算法"""
    L = lattice.shape[0]
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = delta_energy(lattice, i, j, H, lattice_type, J)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1
    return lattice

def simulate_ising(L, T, n_steps, H=0.0, lattice_type='Square', J=1):
    """完整模拟，返回磁化率、磁化强度、Binder比率及每步磁化数据"""
    lattice = initialize_lattice(L, lattice_type)
    M_list = []
    for _ in range(n_steps):
        lattice = monte_carlo_step(lattice, T, H, lattice_type, J)
        M_list.append(np.sum(lattice))
    M_array = np.array(M_list)
    norm_factor = L*L

    M_avg = np.mean(np.abs(M_array)) / norm_factor
    susceptibility = (np.var(M_array) / (T * norm_factor)) if T > 0 else 0
    binder_cumulant = 1 - np.mean(M_array**4) / (3 * (np.mean(M_array**2)**2)) if np.mean(M_array**2) != 0 else 0

    return lattice, M_avg, susceptibility, binder_cumulant, M_array / norm_factor

def cluster_sizes(lattice, lattice_type='Square'):
    """计算晶格中自旋簇大小分布"""
    L = lattice.shape[0]
    visited = np.zeros_like(lattice, dtype=bool)
    sizes = []

    def neighbors_coords(i, j):
        if lattice_type == 'Square':
            return [((i+1)%L, j), ((i-1)%L, j), (i, (j+1)%L), (i, (j-1)%L)]
        elif lattice_type == 'Triangular':
            return [((i+1)%L, j), ((i-1)%L, j), (i, (j+1)%L), (i, (j-1)%L),
                    ((i+1)%L, (j-1)%L), ((i-1)%L, (j+1)%L)]
        else:
            return []

    for i in range(L):
        for j in range(L):
            if not visited[i, j]:
                spin_val = lattice[i, j]
                size = 0
                queue = deque()
                queue.append((i, j))
                visited[i, j] = True
                while queue:
                    x, y = queue.popleft()
                    size += 1
                    for nx, ny in neighbors_coords(x, y):
                        if not visited[nx, ny] and lattice[nx, ny] == spin_val:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                sizes.append(size)
    return sizes

def hysteresis_loop(L, T, H_values, n_eq_steps, n_meas_steps, lattice_type='Square', J=1):
    """
    计算磁滞回线，返回磁化强度数组，矫顽力，剩余磁化强度及最终晶格状态
    """
    lattice = initialize_lattice(L, lattice_type)
    magnetizations = []
    for H in H_values:
        # 平衡
        for _ in range(n_eq_steps):
            lattice = monte_carlo_step(lattice, T, H, lattice_type, J)
        # 测量
        M_meas = []
        for _ in range(n_meas_steps):
            lattice = monte_carlo_step(lattice, T, H, lattice_type, J)
            M_meas.append(np.sum(lattice)/(L*L))
        magnetizations.append(np.mean(M_meas))
    magnetizations = np.array(magnetizations)

    # 矫顽力(Hc)估算: 找磁化强度过零点的外场值
    cross_indices = np.where(np.diff(np.sign(magnetizations)))[0]
    Hc_vals = []
    for idx in cross_indices:
        h1, h2 = H_values[idx], H_values[idx+1]
        m1, m2 = magnetizations[idx], magnetizations[idx+1]
        # 线性插值
        h_cross = h1 - m1*(h2 - h1)/(m2 - m1)
        Hc_vals.append(h_cross)
    coercivity = Hc_vals[0] if Hc_vals else 0.0

    # 剩余磁化强度 Mr
    remanence = np.interp(0, H_values, magnetizations)

    return magnetizations, coercivity, remanence, lattice
