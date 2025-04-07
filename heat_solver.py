import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

def iterate(space, F, timestep_for_print=1):
    for i in range(1, space.shape[0]):
        for j in range(1, space.shape[1] - 1):
            space[i,j] = (1-2*F) * space[i-1,j] + F * space[i-1,j-1] + F * space[i-1,j+1]
            if ((i+ 1,j + 1) == space.shape):
                print(f'{(1-2*F) * space[i-1,j]} + {F * space[i-1,j-1]} + {F * space[i-1,j+1]}')
                print(f'{i}, {j}: {space[i,j]}')
                print(space[i-10:i+1, :])
                print("============")

def iterate_spherical(space, alpha, delta_r, delta_t, timestep_for_print=1):
    print(f'F is: {alpha / delta_r**2 * delta_t}')
    for k in range(1, space.shape[0]):  # Index at 0 since we don't touch ICs
        # Center of egg (r = 0)
        space[k, 0] = space[k-1, 0] + alpha*delta_t / delta_r**2 * (2*space[k-1, 1] - 2*space[k-1, 0])

        # Remaining space inside egg
        for i in range(1, space.shape[1] - 1):
            r_i = delta_r * i
            space[k, i] = space[k-1, i] + alpha*delta_t * (
                (space[k-1, i+1] - 2 * space[k-1, i] + space[k-1, i-1]) / delta_r**2
                + 2 / r_i * (space[k-1, i+1] - space[k-1, i]) / delta_r
            )

            print(2 / r_i * (space[k-1, i+1] - space[k-1, i]) / delta_r)
        
        print(space[k])

    return -1

def plot_temperature(space: np.ndarray):
    """
    Plots a temperature grid with multiple views: 
    - One 2D heatmap
    - Three 3D surface plots from different angles
    
    :param space: np.ndarray of shape (n, m) containing temperature values.
    """
    n, m = space.shape
    x = np.arange(n)
    y = np.arange(m)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(12, 10))
    
    # 2D Heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    c = ax1.imshow(space.T, cmap='coolwarm', origin='lower', aspect='auto')
    fig.colorbar(c, ax=ax1, label='Temperature (K)')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("X")
    ax1.set_title("2D Temperature Distribution")
    
    # 3D Surface Plot - Default View
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, space.T, cmap='coolwarm', edgecolor='k')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("X")
    ax2.set_zlabel("Temperature (K)")
    ax2.set_title("3D View 1")
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, label='Temperature (K)')
    
    # 3D Surface Plot - Rotated View
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax3.plot_surface(X, Y, space.T, cmap='coolwarm', edgecolor='k')
    ax3.view_init(elev=30, azim=135)  # Rotate view
    ax3.set_xlabel("Time")
    ax3.set_ylabel("X")
    ax3.set_zlabel("Temperature (K)")
    ax3.set_title("3D View 2")
    
    # 3D Surface Plot - Top-Down View
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf = ax4.plot_surface(X, Y, space.T, cmap='coolwarm', edgecolor='k')
    ax4.view_init(elev=90, azim=0)  # Top-down view
    ax4.set_xlabel("Time")
    ax4.set_ylabel("X")
    ax4.set_zlabel("Temperature (K)")
    ax4.set_title("3D View 3")
    
    plt.tight_layout()
    plt.show()

def analytical_solver(times: List[float], order: int = 100, num_points: int = 100):
    def C_n(n : int):
        if n == 1:
            return -4 / np.pi
        else:
            return (1 - np.cos((n+1)*np.pi))/ (n+1) / np.pi - (1-np.cos((1-n)*np.pi)) / (n-1)/np.pi + 4*np.cos(np.pi*n) / n / np.pi

    def f_x(x : np.array, t : np.array, order : int) -> float:
        if order < 1:
            raise Exception("Order must be at least 2")
        # Shape: (len(t), len(x))
        values =  2 * np.tile(x, (t.shape[0], 1))

        for n in range(1, order + 1):
            coeff = C_n(n)
            sin_term = np.sin(np.pi * n * x)  # shape: (len(x),)
            decay = np.exp(-2 * (np.pi * n)**2 * t[:, None])  # shape: (len(t), 1)
            values += coeff * sin_term[None, :] * decay  # broadcasted multiplication
        return values  # shape: (len(t), len(x))

    times = np.array(times)
    x_points = np.linspace(0, 1, num_points)

    values = f_x(x_points, times, order)

    plt.figure(figsize=(10, 6))

    for i, t in enumerate(times):
        plt.plot(x_points, values[i, :], label=f"t = {t}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Analytical Solution of the Heat Equation(N={order})")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig("plots/2a_analytical.png", dpi=300)
    plt.show()

def plot_times(times, time_step, space, x_grid):
    """
    Plots a temperature vs position at times: 
    
    :param space: np.ndarray of shape (n, m) containing temperature values.
    """     
    plt.figure(figsize=(10, 6))

    for time in times:
        plt.plot(x_grid,space[int(time / time_step),:], label = f"t={time}")

    plt.xlabel("x (m)")
    plt.ylabel("Temperature (K)")
    plt.grid(True)
    plt.title(f"Temperature vs Position at Varied Times")
    plt.legend(loc="upper left")
    plt.savefig("plots/2a_numerical.png", dpi=300)
    plt.show()

