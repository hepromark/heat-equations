import matplotlib.pyplot as plt
import numpy as np

from heat_solver import iterate, plot_temperature


if __name__ == "__main__":
    t_range = (0, 10000)
    t_grid_size = 10
    x_range = (0, 0.048)
    x_grid_size = 0.001

    t_grid_count = int((t_range[1] - t_range[0]) / t_grid_size) + 1
    x_grid_count = int((x_range[1] - x_range[0]) / x_grid_size) + 1

    t_grid = np.linspace(t_range[0], t_range[1], t_grid_count)
    x_grid = np.linspace(x_range[0], x_range[1], x_grid_count)

    k = 0.479 # thermal conductivity [W/mK]
    p = 1176  # density [kg/m^3]
    c_p = 2893 # specific heat  [j/Kg*K]
    alpha = k / p / c_p

    F = t_grid_size / (x_grid_size**2) * alpha
    print(F)
    if abs(F) > .5:
        raise Exception(f"F is {F}, should be smaller than 0.5")

    space = np.full((t_grid_count, x_grid_count), 275, dtype=np.float64)  # 275 K intial temp
    space[0,:] = 275
    space[:,0] = 373.15
    space[:,-1] = 373.15

    print(space.shape)
    print("=============")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    cooked_t_idx = iterate(space, F)

    space = space[:cooked_t_idx, :]
    plot_temperature(space)