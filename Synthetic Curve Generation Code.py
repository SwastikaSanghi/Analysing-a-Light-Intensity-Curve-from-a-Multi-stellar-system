import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters for the system
t_max =1000  # total observation time in days
dt = 0.02       # time step in days
time_uniform = np.arange(0, t_max, dt)

# Introducing small variation in observed time 
time_jitter = np.random.normal(0, 0.002, size=len(time_uniform))
time = time_uniform + time_jitter                              

# Inner binary parameters
P_inner = 8.23     # inner binary period in days
depth_primary = 0.033
depth_secondary = 0.035
duration_inner = 0.2  # duration of eclipses in days

# Tertiary parameters
P_outer = 188   # tertiary star period
depth_outer = 0.75
duration_outer = 1.7

# Generate inner binary eclipses
mag_inner = np.zeros_like(time)
for t0 in np.arange(0, t_max, P_inner):
    # primary eclipse
    mag_inner += -depth_primary * np.exp(-0.5*((time-t0)/duration_inner)**2)
    # secondary eclipse at half period
    mag_inner += -depth_secondary * np.exp(-0.5*((time-(t0+P_inner/2))/duration_inner)**2)

# Generate tertiary eclipses
mag_outer = np.zeros_like(time)
for t0 in np.arange(0, t_max, P_outer):
    mag_outer += -depth_outer * np.exp(-0.5*((time-t0)/duration_outer)**2)

# Total light curve with noise
noise = np.random.normal(0, 0.001, size=len(time))
mag_total = mag_inner + mag_outer + noise

# Save to CSV
df = pd.DataFrame({'time': time, 'magnitude': mag_total})
df.to_csv('synthetic_triple_star.csv', index=False)
print("Synthetic triple-star light curve saved as 'synthetic_triple_star.csv'")

# Optional: plot the light curve
plt.figure(figsize=(12,4))
plt.plot(time, mag_total, 'k')
plt.xlabel("Time [days]")
plt.ylabel("Magnitude")
plt.title("Triple Eclipsing Star Light Curve")
plt.show()
