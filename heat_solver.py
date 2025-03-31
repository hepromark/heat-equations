import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def iterate(space, F):
    for i in range(1, space.shape[0]):
        for j in range(1, space.shape[1] - 1):
            space[i,j] = (1-2*F) * space[i-1,j] + F * space[i-1,j-1] + F * space[i-1,j+1]
            if space[i][int(space.shape[1] / 2)] > 80 + 273.15:
                print(f"Cooked at time timestep {i}")
                return i
            if ((i+ 1,j + 1) == space.shape):
                print(f'{(1-2*F) * space[i-1,j]} + {F * space[i-1,j-1]} + {F * space[i-1,j+1]}')
                print(f'{i}, {j}: {space[i,j]}')
                print(space[i-10:i+1, :])
                print("============")
    
    print(space)
    print(f"Center temp is {space[space.shape[0] // 2][space.shape[1] // 2]}")
    print(f"Not fully cooked")

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


def plot_times(times, time_step, space, x_grid):
    """
    Plots a temperature vs position at times: 
    
    :param space: np.ndarray of shape (n, m) containing temperature values.
    """   
    fig = plt.figure(figsize=(12, 10))
    
    for time in times:
        plt.plot(x_grid,space[int(time / time_step),:], label = f"t={time}")

    plt.xlabel("x (m)")
    plt.ylabel("Temperature (K)")
    plt.title(f"Temperature vs Position at Varied Times")
    plt.legend(loc="upper left")
    plt.show()

