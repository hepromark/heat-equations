import numpy as np
from scipy.optimize import fsolve

class eggSystem:
    def __init__(self, major_length, minor_length, x_guess=2, b_guess=1):
        self.a = major_length  # Major axis length
        self.target_value = minor_length / 2  # Half of the minor axis length
        self.x_guess = x_guess  # Initial guess for x
        self.b_guess = b_guess  # Initial guess for b
        self.x = None  # Solution for x
        self.b = None  # Solution for b
        self.volume = None  # Computed volume

    def egg_function(self, x, a, b):
        return np.sqrt((a - b) - 2*x + np.sqrt(4*b*x + (a - b)**2)) * np.sqrt(x) / np.sqrt(2)

    def egg_func_derivative(self, b, a, x):
        A = a - b  
        B = 4 * b 
        C = A ** 2 

        term1 = np.sqrt(A - 2 * x + np.sqrt(B * x + C))
        term2 = np.sqrt(x) / np.sqrt(2)

        # Derivative of term 1
        f_prime = (1 / (2 * np.sqrt(A - 2 * x + np.sqrt(B * x + C)))) * (-2 + (B / (2 * np.sqrt(B * x + C))))

        # Derivative of term 2
        g_prime = 1 / (2 * np.sqrt(2 * x))

        # Product rule
        dy_dx = f_prime * term2 + term1 * g_prime
        return dy_dx

    def egg_system(self, vars, a, target_value):
        x, b = vars  
        eq1 = self.egg_function(x, a, b) - target_value  # Function should be target_value
        eq2 = self.egg_func_derivative(b, a, x)  # Derivative should be 0
        return [eq1, eq2]

    def solve_egg_system(self):
        solution = fsolve(self.egg_system, [self.x_guess, self.b_guess], args=(self.a, self.target_value))
        self.x, self.b = solution
        return self.x, self.b

    def egg_volume(self):
        if self.b is None:
            raise ValueError("Solve for x and b first using solve_egg_system().")

        a = self.a
        b = self.b
        if b < 1e-5:
            self.volume = (np.pi * a**3) / 6
        else:
            self.volume = (np.pi / 2) * (((a + b) ** 3 * a / (6 * b)) - (a**3 / 6) - (a**2 * b / 2) - (((a + b) ** 5 - (a - b) ** 5) / (60 * b**2)))
        return self.volume

    def find_spherical_radius(self):
        if self.volume is None:
            raise ValueError("Compute volume first using egg_volume().")
        return np.cbrt(self.volume * 6 / np.pi) / 2

def main():
    quail_lengths = [3.5, 2.7]
    
    egg = eggSystem(major_length=quail_lengths[0], minor_length=quail_lengths[1])
    
    x_sol, b_sol = egg.solve_egg_system()
    print(f"Solution: x = {x_sol}, b = {b_sol}")
    
    volume = egg.egg_volume()
    print(f"Volume: {volume}")

    radius = egg.find_spherical_radius()
    print(f"Equivalent sphere radius: {radius}")

if __name__ == "__main__":
    main()


