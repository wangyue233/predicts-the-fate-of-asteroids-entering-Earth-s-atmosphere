from . import solver
import math
import matplotlib.pyplot as plt
import numpy as np

# Input constants
Cd = 1
rho0 = 1.2
radius = 1
angle = 30
density = 3000
velocity = 1.0e5
H = 8000
init_altitude = 100e3
A = np.pi*radius**2
m = 4/3*np.pi*radius**3*density
K = Cd*A*rho0/(2*m)

# Set the condition for analytical solution
another_planet = solver.Planet(Cd=1., Ch=0, Q=1e7, Cl=0, alpha=0.3, Rp=math.inf,
                 g=0, H=8000., rho0=1.2)

# Set timestep series
start_time = 0.005
end_time = 0.2
step = 0.005
timestep = np.arange(start_time, end_time, step)

# Calculate RMS error for each timestep size
rms_list = []
for tt in timestep:
    another_result = another_planet.solve_atmospheric_entry(
    radius=1, angle=30, strength=1e32, velocity=1.0e5, density=3000,init_altitude=100e3, dt=tt)
    test_velocity = velocity * np.exp(H * K/np.sin(angle * math.pi/180) * (np.exp(-init_altitude/H)\
                    -np.exp(-another_result['altitude']/H)))
    rms = np.mean((test_velocity-another_result['velocity'])**2)
    rms = math.sqrt(rms)
    rms_list.append(rms)

plt.plot(timestep,rms_list)
plt.show()