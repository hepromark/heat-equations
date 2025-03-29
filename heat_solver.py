import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def iterate(space, F):
    for i in range(1, space.shape[0]):
        for j in range(1, space.shape[1] - 1):
            space[i,j] = (1-2*F) * space[i-1,j] + F * space[i-1,j-1] + F * space[i-1,j+1]
            # if (i == 40 and j == 1):
            #     print(f'{(1-2*F) * space[i-1,j]} + {F * space[i-1,j-1]} + {F * space[i-1,j+1]}')
            #     print(f'{i}, {j}: {space[i,j]}')
            #     print(space[i-10:i+1, :])
            #     print("============")
    
    print(space)
    print(f"Center temp is {space[space.shape[0] // 2][space.shape[1] // 2]}")

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

if __name__ == "__main__":
    t_range = (0, 1000)
    t_grid_size = 3
    x_range = (0, 0.05)
    x_grid_size = 0.001

    t_grid_count = int((t_range[1] - t_range[0]) / t_grid_size) + 1
    x_grid_count = int((x_range[1] - x_range[0]) / x_grid_size) + 1

    t_grid = np.linspace(t_range[0], t_range[1], t_grid_count)
    x_grid = np.linspace(x_range[0], x_range[1], x_grid_count)

    k = 0.479 # thermal conductivity [W/mk]
    p = 1176  # density [kg/m^3]
    c_p = 2893 # specific heat  [j/Kg*k]
    alpha = k / p / c_p

    F = t_grid_size / (x_grid_size**2) * alpha
    print(F)
    print(1-2*F, " should be bigger than 0")

    space = np.full((t_grid_count, x_grid_count), 275)  # 275 K intial temp
    space[0,:] = 275
    space[:,0] = 373.15
    space[:,-1] = 373.15
    print(space.shape)
    print("=============")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    iterate(space, F)
    plot_temperature(space)

