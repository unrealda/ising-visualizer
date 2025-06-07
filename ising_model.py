import numpy as np

def run_ising_simulation(L, Tmin, Tmax, nT, Ntrial, folder):
    N = L * L
    T_list = np.linspace(Tmin, Tmax, nT)
    M_list = []
    Chi_list = []
    spin_snapshots = []

    for T in T_list:
        beta = 1.0 / T
        p = 1 - np.exp(-2 * beta)
        S = np.random.choice([-1, 1], size=N)  # initial spin state

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

        # Store snapshot in 2D
        S2D = S.reshape((L, L))
        spin_snapshots.append(S2D)

        # Save raw snapshot
        np.savetxt(f"{folder}/snapshot_T{T:.3f}.dat", S2D, fmt='%d')

    return T_list, spin_snapshots, M_list, Chi_list

def get_neighbors(index, L):
    row = index // L
    col = index % L
    neighbors = [
        ((row - 1) % L) * L + col,  # up
        ((row + 1) % L) * L + col,  # down
        row * L + (col - 1) % L,    # left
        row * L + (col + 1) % L     # right
    ]
    return neighbors
