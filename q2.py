import matplotlib.pyplot as plt
import numpy as np

from heat_solver import iterate, plot_times, analytical_solver


if __name__ == "__main__":
    t_range = (0, 1000)
    t_grid_size = 0.001
    x_range = (0, 1)
    x_grid_size = 0.1

    t_grid_count = int((t_range[1] - t_range[0]) / t_grid_size) + 1
    x_grid_count = int((x_range[1] - x_range[0]) / x_grid_size) + 1

    t_grid = np.linspace(t_range[0], t_range[1], t_grid_count)
    x_grid = np.linspace(x_range[0], x_range[1], x_grid_count)

    alpha = 2

    F = t_grid_size / (x_grid_size**2) * alpha
    print(F)
    print(1-2*F, " should be bigger than 0")

    # Initial Conditions
    space = np.zeros((t_grid_count, x_grid_count), dtype=np.float64)  
    space[0,:] = np.cos(np.pi * x_grid) # u(t=0) = cos(pix)
    space[:,0] = 0
    space[:,-1] = 2
    print(space.shape, x_grid.shape)

    print(space.shape)
    print("=============")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    times = [0, 0.001,0.01,0.1,10]
    
    # Iterative approach
    iterate(space, F)
    plot_times(times, t_grid_size, space, x_grid)
    
    # Analytical approach
    analytical_solver(times, order=100, num_points=10)