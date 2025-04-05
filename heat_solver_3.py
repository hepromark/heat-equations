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

r = rad_quail
#Egg Function Temperature


def spherical_heat_solver(r, t, r_points, dt, alpha, t_water, t_init):
    r = r / 100
    dr = r / r_points
    t_points = int(t / dt)

    
    temp = np.zeros((t_points, r_points))
    
    # Set boundary conditions
    temp[:, -1] = t_water  # Python uses -1 for last index
    temp[0,:] = t_init
    
    # Calculate Fourier number for stability check 
    F = alpha * dt / (dr**2)
    if F > 0.5:
        raise ValueError("Stability condition violated: F > 0.5")
    

    done_counter = 0
    end_index = 0

    # Main simulation loop
    
    for k in range(1, t_points): # Temp steps
        temp[k, 0] = temp[k-1, 0] + alpha * dt / (dr**2) * (2 * temp[k-1, 1] - 2* temp [k-1, 0])
        
        # Interior Points
        for i in range(1, r_points - 1): # Radial positions
            r_i = i*dr # radial position
            temp[k, i] = temp[k-1, i] + alpha * dt * (
                (temp[k-1, i+1] - 2 * temp[k-1, i] + temp[k-1, i-1]) / (dr**2) + 
                (2 / (r_i)) * ((temp[k-1, i+1] - temp[k-1, i]) / dr)
            )
        # 80°C Check

        if temp[k,0] >= 80:
            print(f"Cooked at time(s): {k*dt:.1f}")
            return temp[:k+1, :], k*dt
    
    return temp, t
# Run simulation

# Display maximum temperature
temp_chick, end_index_chick = spherical_heat_solver(rad_chicken, t_chick, r_points, dt, alpha, t_water, t_init)

# Extract temperature at the center (r = 0)
temp_center_chick = temp_chick[0, :]

time = np.linspace(0, t_chick, len(temp_center_chick))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, temp_center_chick, linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Center Temperature [°C]')
plt.title('Temperature at the Center of the Chicken Egg vs Time')
plt.grid(True)

# Temperature at center over time
plt.subplot(1, 2, 1)
plt.plot(np.arange(temp_chick.shape[0]), temp_chick[:, 0], 'r-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Center Temperature')
final_time_chick = (end_index_chick) if end_index_chick > 0 else t_chick

# Temperature profile at last time step
plt.subplot(1, 2, 2)
radial_positions = np.linspace(0, r, r_points)
plt.plot(radial_positions, temp_chick[-1, :], 'b-', linewidth=2)
plt.xlabel('Radial Position (mm)')
plt.ylabel('Temperature (°C)')
plt.title('Final Temperature Profile')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final center temperature: {temp_chick[-1, 0]:.2f}°C")
print(f'The chicken egg is fully cooked at t = {final_time_chick:.1f} seconds or or t = {final_time_chick/60:.1f} minutes')

###QUAIL
# Run simulation
temp_quail, end_index_quail = spherical_heat_solver(rad_quail, t_quail, r_points, dt, alpha, t_water, t_init)

# Extract temperature at the center (r = 0)
temp_center_quail = temp_quail[0, :]

time = np.linspace(0, t_quail, len(temp_center_quail))

final_time_quail = (end_index_quail) if end_index_quail > 0 else t_quail


# Temperature at center over time
plt.subplot(1, 2, 1)
plt.plot(np.arange(temp_quail.shape[0]), temp_quail[:, 0], 'r-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Center Temperature')
final_time_chick = (end_index_chick) if end_index_chick > 0 else t_chick

# Temperature profile at last time step
plt.subplot(1, 2, 2)
radial_positions = np.linspace(0, rad_quail, r_points)
plt.plot(radial_positions, temp_quail[-1, :], 'b-', linewidth=2)
plt.xlabel('Radial Position (mm)')
plt.ylabel('Temperature (°C)')
plt.title('Final Temperature Profile')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final center temperature: {temp_quail[-1, 0]:.2f}°C")
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
temp_os, end_index_os = spherical_heat_solver(rad_ostrich, t_os, r_points, dt, alpha, t_water, t_init)

# Extract temperature at the center (r = 0)
temp_center_os = temp_os[0, :]

time = np.linspace(0, t_os, len(temp_center_os))

final_time_os = (end_index_os) if end_index_os > 0 else t_os

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, temp_center_os, linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Center Temperature [°C]')
plt.title('Temperature at the Center of the Ostrich Egg vs Time')
plt.grid(True)
plt.show()


# Temperature at center over time
plt.subplot(1, 2, 1)
plt.plot(np.arange(temp_os.shape[0]), temp_os[:, 0], 'r-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Center Temperature')

# Temperature profile at last time step
plt.subplot(1, 2, 2)
radial_positions = np.linspace(0, rad_ostrich, r_points)
plt.plot(radial_positions, temp_os[-1, :], 'b-', linewidth=2)
plt.xlabel('Radial Position (mm)')
plt.ylabel('Temperature (°C)')
plt.title('Final Temperature Profile')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final center temperature: {temp_os[-1, 0]:.2f}°C")
print(f'The ostrich egg is fully cooked at t = {final_time_os:.1f} seconds or or t = {final_time_os/60:.1f} minutes')
