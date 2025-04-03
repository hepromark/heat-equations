import matplotlib.pyplot as plt
import numpy as np

from heat_solver import iterate, plot_temperature, iterate_spherical


if __name__ == "__main__":
    t_range = (0, 500)
    x_range = (0, 0.024)
    x_grid_size = 0.005

    x_grid_count = int((x_range[1] - x_range[0]) / x_grid_size) + 1

    x_grid = np.linspace(x_range[0], x_range[1], x_grid_count)

    k = 0.3370 # thermal conductivity [W/mK]
    p = 1036  # density [kg/m^3]
    c_p = 2093 # specific heat  [j/Kg*K]
    alpha = k / p / c_p

    # set a target F and back-calculate the t_grid_size
    F = 0.40
    t_grid_size = F / alpha * x_grid_size**2
    t_grid_count = int((t_range[1] - t_range[0]) / t_grid_size) + 1
    t_grid = np.linspace(t_range[0], t_range[1], t_grid_count)

    print(f'Alpha: {alpha}')
    print(f'F: {F}')
    print(f't_grid_size: {t_grid_size}')
    print(f'x_grid_size: {x_grid_size}')

    if abs(F) > .5:
        raise Exception(f"F is {F}, should be smaller than 0.5")

    space = np.full((t_grid_count, x_grid_count), 275, dtype=np.float64)  # 275 K intial temp
    space[0,:] = 275.0
    space[:,-1] = 373.15

    plot_temperature(space)

    print(space.shape)
    print("=============")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    cooked_t_idx = iterate_spherical(space, alpha=alpha, timestep_for_print=t_grid_size, delta_r=x_grid_size, delta_t=t_grid_size)

    space = space[:cooked_t_idx, :]
    plot_temperature(space)