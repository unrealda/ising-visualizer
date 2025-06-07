import matplotlib.pyplot as plt
import numpy as np

def generate_arrow_plot(S2D, temperature):
    Lx, Ly = S2D.shape
    X, Y = np.meshgrid(np.arange(Ly), np.arange(Lx))
    U = np.zeros_like(S2D)
    V = S2D  # spin directions: +1 up, -1 down

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, U, V, pivot='middle', scale=1, scale_units='xy', angles='xy', color='black')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, Ly - 0.5)
    ax.set_ylim(-0.5, Lx - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Spin Configuration at T = {temperature:.3f}')
    return fig
