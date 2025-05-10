import matplotlib.pyplot as plt
import numpy as np

def generate_arrow_plot(S2D, temperature):
    Lx, Ly = S2D.shape
    X, Y = np.meshgrid(np.arange(Ly), np.arange(Lx))
    U = np.zeros_like(S2D)
    V = S2D
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, U, V, pivot='middle', scale=1, scale_units='xy', angles='xy', color='black')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, Ly - 0.5)
    ax.set_ylim(-0.5, Lx - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Spin Configuration at T = {temperature:.3f}')
    return fig

def generate_hysteresis_plot(T_list, M_list, T_c):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T_list, M_list, 'o-', label='Magnetization')
    ax.axvline(x=T_c, color='red', linestyle='--', label=f'Tc = {T_c:.3f}')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Magnetization (M)')
    ax.set_title('Hysteresis Plot')
    ax.legend()
    return fig

def generate_cluster_size_plot(S2D, temperature):
    clusters = find_clusters(S2D)
    cluster_sizes = [len(cluster) for cluster in clusters]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cluster_sizes, bins=np.arange(1, max(cluster_sizes) + 1), color='blue', edgecolor='black')
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Cluster Size Distribution at T = {temperature:.3f}')
    return fig

def find_clusters(S2D):
    clusters = []
    visited = np.zeros_like(S2D, dtype=bool)
    
    def dfs(x, y):
        stack = [(x, y)]
        cluster = []
        while stack:
            cx, cy = stack.pop()
            if not visited[cx, cy]:
                visited[cx, cy] = True
                cluster.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = (cx + dx) % S2D.shape[0], (cy + dy) % S2D.shape[1]
                    if not visited[nx, ny] and S2D[nx, ny] == S2D[cx, cy]:
                        stack.append((nx, ny))
        return cluster
    
    for i in range(S2D.shape[0]):
        for j in range(S2D.shape[1]):
            if not visited[i, j]:
                cluster = dfs(i, j)
                clusters.append(cluster)
    
    return clusters
