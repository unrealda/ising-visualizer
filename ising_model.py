import numpy as np
import pandas as pd
from collections import deque

def square_neighbors(L):
    neighbors = {}
    for i in range(L):
        for j in range(L):
            site = i * L + j
            neighbors[site] = []
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = (i+dx)%L, (j+dy)%L
                neighbor_site = ni * L + nj
                neighbors[site].append(neighbor_site)
    return neighbors

def triangular_neighbors(L):
    neighbors = {}
    for i in range(L):
        for j in range(L):
            site = i * L + j
            neighbors[site] = []
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]:
                ni, nj = (i+dx)%L, (j+dy)%L
                neighbor_site = ni * L + nj
                neighbors[site].append(neighbor_site)
    return neighbors

def wolff_algorithm(L, neighbors, T, Ntrial):
    N = L * L
    beta = 1.0 / T
    p = 1 - np.exp(-2 * beta)
    S = np.random.choice([-1, 1], size=N)
    M_list = []
    cluster_sizes = []

    for _ in range(Ntrial):
        seed = np.random.randint(N)
        cluster = set([seed])
        pocket = deque([seed])

        while pocket:
            site = pocket.pop()
            for neigh in neighbors[site]:
                if S[neigh] == S[site] and neigh not in cluster and np.random.rand() < p:
                    cluster.add(neigh)
                    pocket.append(neigh)
        S[list(cluster)] *= -1
        M = np.sum(S) / N
        M_list.append(M)
        cluster_sizes.append(len(cluster))

    return M_list, cluster_sizes

def simulate(L, lattice, Tmin, Tmax, nT, Ntrial):
    if lattice.lower() == 'square':
        neighbors = square_neighbors(L)
    elif lattice.lower() == 'triangular':
        neighbors = triangular_neighbors(L)
    else:
        raise ValueError("Unknown lattice type")

    T_list = np.linspace(Tmin, Tmax, nT)
    M_mean = []
    M_var = []
    Chi = []
    Binder = []
    all_cluster_sizes = []

    for T in T_list:
        M_list, cluster_sizes = wolff_algorithm(L, neighbors, T, Ntrial)
        M_arr = np.array(M_list)
        M_abs = np.abs(M_arr)
        M_mean.append(np.mean(M_abs))
        M_var.append(np.var(M_arr))
        Chi.append(L*L * (np.mean(M_arr**2) - np.mean(M_abs)**2) / T)
        M2 = np.mean(M_arr**2)
        M4 = np.mean(M_arr**4)
        Binder.append(1 - M4 / (3 * M2**2))
        all_cluster_sizes.extend(cluster_sizes)

    result_df = pd.DataFrame({
        "Temperature": T_list,
        "Magnetization": M_mean,
        "Magnetization_Var": M_var,
        "Susceptibility": Chi,
        "Binder_Ratio": Binder
    })

    return result_df, all_cluster_sizes

def hysteresis_simulation(L, lattice, T_list, Ntrial, gifname, final_gifname, save_path=None):
    from PIL import Image, ImageDraw

    if lattice.lower() == 'square':
        neighbors = square_neighbors(L)
    elif lattice.lower() == 'triangular':
        neighbors = triangular_neighbors(L)
    else:
        raise ValueError("Unknown lattice type")

    N = L * L
    beta_list = 1.0 / np.array(T_list)
    H_vals_dict = {}

    A_hyst_all = []
    M_r_all = []
    H_c_all = []

    all_gif_frames = []
    final_gif_frames = []

    for idx_T, (T, beta) in enumerate(zip(T_list, beta_list)):
        if abs(T - T_list[np.argmax(beta_list)]) < 0.3:
            H_vals = np.concatenate((np.arange(-1,1.05,0.05), np.arange(0.95,-1.05,-0.05)))
            Ntrial_hyst = 200
        else:
            H_vals = np.concatenate((np.arange(-1,1.2,0.2), np.arange(0.8,-1.2,-0.2)))
            Ntrial_hyst = 100

        S = np.ones(N)
        M_H = []

        for h in H_vals:
            for _ in range(Ntrial_hyst):
                seed = np.random.randint(N)
                cluster = set([seed])
                pocket = deque([seed])
                while pocket:
                    site = pocket.pop()
                    for neigh in neighbors[site]:
                        if S[neigh] == S[site] and neigh not in cluster:
                            prob = 1 - np.exp(-2 * beta * (1 + h * S[site]))
                            if np.random.rand() < prob:
                                cluster.add(neigh)
                                pocket.append(neigh)
                S[list(cluster)] *= -1
            M_H.append(np.sum(S)/N)

            if idx_T == len(T_list)-1:
                fig = draw_hysteresis_frame(H_vals[:len(M_H)], M_H, T, h)
                final_gif_frames.append(fig)

        M_H = np.array(M_H)

        idx_H0 = np.argmin(np.abs(H_vals))
        M_r_all.append(M_H[idx_H0])
        A_hyst_all.append(np.trapz(np.abs(M_H), H_vals))

        idx_cross = np.where(np.diff(np.sign(M_H)))[0]
        if len(idx_cross) > 0:
            i1 = idx_cross[0]
            H1, H2 = H_vals[i1], H_vals[i1+1]
            M1, M2 = M_H[i1], M_H[i1+1]
            H_c = np.abs(H1 - M1 * (H2 - H1) / (M2 - M1))
            H_c_all.append(H_c)
        else:
            H_c_all.append(np.nan)

        fig = draw_hysteresis_frame(H_vals, M_H, T)
        all_gif_frames.append(fig)

        if save_path:
            fig.save(f"{save_path}/Hysteresis_T={T:.2f}.png")

    all_gif_frames[0].save(gifname, save_all=True, append_images=all_gif_frames[1:], loop=0, duration=800)
    final_gif_frames[0].save(final_gifname, save_all=True, append_images=final_gif_frames[1:], loop=0, duration=400)

    df = pd.DataFrame({
        "Temperature": T_list,
        "Area": A_hyst_all,
        "Remanence": M_r_all,
        "Coercivity": H_c_all
    })

    return df

def draw_hysteresis_frame(H_vals, M_H, T, H=None):
    import matplotlib.pyplot as plt
    from PIL import Image
    from io import BytesIO

    fig, ax = plt.subplots()
    ax.plot(H_vals, M_H, 'o-', linewidth=1.5)
    title = f"Hysteresis Loop at T={T:.2f}"
    if H is not None:
        title += f" | H={H:.2f}"
    ax.set_title(title)
    ax.set_xlabel("External Field H")
    ax.set_ylabel("Magnetization")
    ax.axis([-1.1,1.1,-1.1,1.1])
    ax.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
