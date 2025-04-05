import numpy as np
import matplotlib.pyplot as plt
from egg_equation import get_sphere_radius_of_egg 

# Parameters
t_chick = 900  # s total simulation time
t_os = 7000
t_quail = 300
r_points = 10
dt = 0.1  # s
k = 0.499  # thermal conductivity W/mK
p = 1175.04  # density [kg/m^3]
c_p = 3026.05  # specific heat [j/Kg*K]
alpha = k / p / c_p

t_water = 100  # °C
t_init = 2  # °C

# Egg dimens (cm)
quail_egg_dimens = [3.5, 2.7]
chicken_egg_dimens = [6.7, 4.5]
ostrich_egg_dimens = [15, 13]

rad_quail = get_sphere_radius_of_egg(quail_egg_dimens)
rad_chicken = get_sphere_radius_of_egg(chicken_egg_dimens)
rad_ostrich = get_sphere_radius_of_egg(ostrich_egg_dimens)

#Egg Function Temperature


def egg_temperature(r, t, r_points, dt, alpha, t_water, t_init):
    r = r / 100
    dr = r / r_points
    t_points = int(t / dt)
    
    temp = np.zeros((r_points, t_points))
    
    temp[-1, :] = t_water  # Python uses -1 for last index
    temp[:, 0] = t_init
    
    done_counter = 0
    end_index = 0
    
    for k in range(t_points - 1):
        temp[0, k+1] = temp[0, k] + alpha * dt * ((2 * temp[1, k] - 2 * temp[0, k]) / (dr**2))
        
        for i in range(1, r_points - 1):
            temp[i, k+1] = temp[i, k] + alpha * dt * (
                (temp[i+1, k] - 2 * temp[i, k] + temp[i-1, k]) / (dr**2) + 
                (2 / (i * dr)) * ((temp[i+1, k] - temp[i, k]) / dr)
            )
        
        

        temp[-1, k+1] = t_water
        
        

        # 80°C Check
        if np.all(temp[:, k+1] >= 80):
            done_counter += 1
        else:
            done_counter = 0
        
        if done_counter >= 100:
            end_index = k
            break
    
    return temp, end_index

###CHICKEN
# Run simulation
temp_chick, end_index_chick = egg_temperature(rad_chicken, t_chick, r_points, dt, alpha, t_water, t_init)

# Extract temperature at the center (r = 0)
temp_center_chick = temp_chick[0, :]

time = np.linspace(0, t_chick, len(temp_center_chick))

final_time_chick = (end_index_chick) * dt if end_index_chick > 0 else t_chick

print(f'The chicken egg is fully cooked at t = {final_time_chick:.1f} seconds or or t = {final_time_chick/60:.1f} minutes')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, temp_center_chick, linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Center Temperature [°C]')
plt.title('Temperature at the Center of the Chicken Egg vs Time')
plt.grid(True)
plt.show()


###QUAIL
# Run simulation
temp_quail, end_index_quail = egg_temperature(rad_quail, t_quail, r_points, dt, alpha, t_water, t_init)

# Extract temperature at the center (r = 0)
temp_center_quail = temp_quail[0, :]

time = np.linspace(0, t_quail, len(temp_center_quail))

final_time_quail = (end_index_quail) * dt if end_index_chick > 0 else t_quail

print(f'The quail egg is fully cooked at t = {final_time_quail:.1f} seconds or or t = {final_time_quail/60:.1f} minutes')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, temp_center_quail, linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Center Temperature [°C]')
plt.title('Temperature at the Center of the Quail Egg vs Time')
plt.grid(True)
plt.show()


###OSTRICH
# Run simulation
temp_os, end_index_os = egg_temperature(rad_ostrich, t_os, r_points, dt, alpha, t_water, t_init)

# Extract temperature at the center (r = 0)
temp_center_os = temp_os[0, :]

time = np.linspace(0, t_os, len(temp_center_os))

final_time_os = (end_index_os) * dt if end_index_os > 0 else t_os

print(f'The ostrich egg is fully cooked at t = {final_time_os:.1f} seconds or or t = {final_time_os/60:.1f} minutes')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, temp_center_os, linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Center Temperature [°C]')
plt.title('Temperature at the Center of the Ostrich Egg vs Time')
plt.grid(True)
plt.show()