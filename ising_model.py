import numpy as np
import random
from scipy.signal import convolve2d

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
                    index(i, j+1), index(i+1, j),
                    index(i, j-1), index(i-1, j)
                ]
            elif lattice_type == "triangular":
                nbrs = [
                    index(i, j+1), index(i+1, j),
                    index(i, j-1), index(i-1, j),
                    index(i+1, j-1), index(i-1, j+1)
                ]
            neighbors[idx] = nbrs
    return neighbors

def wolff_update(spins, neighbors, beta, H=0.0):
    N = len(spins)
    visited = np.zeros(N, dtype=bool)
    seed = random.randint(0, N-1)
    cluster_spin = spins[seed]
    
    p_add = 1 - np.exp(-2 * beta * (1 + H * cluster_spin))
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
    neighbors = build_neighbors(L, lattice_type)
    T_list = np.linspace(Tmin, Tmax, nT)
    results = []
    N = L * L

    for T in T_list:
        beta = 1.0 / T
        spins = np.random.choice([-1, 1], size=N)
        mags = np.zeros(Ntrial)
        clusters = np.zeros(Ntrial)
        M2_samples = np.zeros(Ntrial)
        M4_samples = np.zeros(Ntrial)

        for i in range(Ntrial):
            spins, csize, mag = wolff_update(spins, neighbors, beta)
            mags[i] = mag
            clusters[i] = csize
            M2_samples[i] = mag**2
            M4_samples[i] = mag**4

        M = np.mean(np.abs(mags)) / N
        M2 = np.mean(M2_samples) / N**2
        M4 = np.mean(M4_samples) / N**4
        Mvar = np.var(mags / N)
        chi = (N / T) * (M2 - M**2)
        
        ### Binder 比率计算及其方差
        U4_samples = 1 - (M4_samples / N**4) / (3 * (M2_samples / N**2)**2)
        U4_mean = np.mean(U4_samples)
        U4_var = np.var(U4_samples)

        up_ratio = np.sum(spins > 0) / N
        local_order = convolve2d(spins.reshape(L,L), np.ones((3,3)), 'same') / 9

        results.append({
            'T': T, 'M': M, 'M2': M2, 'M4': M4, 'Mvar': Mvar,
            'chi': chi, 'N': N, 
            'Var_M2': np.var(M2_samples/N**2),
            'Var_M4': np.var(M4_samples/N**4),
            'Binder_U4': U4_mean,
            'Var_U4': U4_var,
            'up_ratio': up_ratio,
            'down_ratio': 1-up_ratio,
            'spins_snapshot': spins.reshape(L,L),
            'mean_cluster_size': np.mean(clusters),
            'local_order': local_order
        })
    return results

def calculate_hysteresis_features(H_vals, M_vals):
    idx_H0 = np.argmin(np.abs(H_vals))
    M_r = M_vals[idx_H0]
    A_hyst = np.trapz(np.abs(M_vals), H_vals)
    
    idx_cross = np.where(np.diff(np.sign(M_vals)))[0]
    H_c = np.nan
    if len(idx_cross) > 0:
        i1 = idx_cross[0]
        H1, H2 = H_vals[i1], H_vals[i1+1]
        M1, M2 = M_vals[i1], M_vals[i1+1]
        H_c = np.abs(H1 - M1*(H2-H1)/(M2-M1))
    
    return {'M_r': M_r, 'A_hyst': A_hyst, 'H_c': H_c}

def run_hysteresis(L, lattice_type, T_list, Ntrial=100, n_repeat=5):
    neighbors = build_neighbors(L, lattice_type)
    N = L * L
    hyst_data = []

    for T in T_list:
        beta = 1 / T
        if abs(T - max(T_list)) < 1e-8:
            H_vals = list(np.arange(-1, 1.05, 0.05)) + list(np.arange(0.95, -1.05, -0.05))
        else:
            H_vals = list(np.arange(-1, 1.2, 0.2)) + list(np.arange(0.8, -1.2, -0.2))

        spins = np.ones(N)
        M_H = []
        frames = []

        for H in H_vals:
            for _ in range(Ntrial):
                spins, _, _ = wolff_update(spins, neighbors, beta, H)
            M_H.append(np.sum(spins) / N)
            
            if abs(T - max(T_list)) < 1e-8:
                frames.append(spins.copy().reshape(L,L))

        ### 计算矫顽力多次平均（Hc_mean和Hc_var）
        Hc_samples = []
        for _ in range(n_repeat):
            features_tmp = calculate_hysteresis_features(H_vals, M_H)
            if not np.isnan(features_tmp['H_c']):
                Hc_samples.append(features_tmp['H_c'])
        Hc_mean = np.mean(Hc_samples) if Hc_samples else np.nan
        Hc_var = np.var(Hc_samples) if Hc_samples else np.nan

        features = calculate_hysteresis_features(H_vals, M_H)
        hyst_data.append({
            'T': T, 'H_vals': H_vals, 'M_vals': M_H,
            'loop_area': features['A_hyst'],
            'M_r': features['M_r'],
            'H_c': Hc_mean,
            'Var_Hc': Hc_var,
            'final_frames': frames if frames else None
        })
    return hyst_data
