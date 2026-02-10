import matplotlib.pyplot as plt
import numpy as np

# Generate random data for stride length and stride velocity
np.random.seed(42)
stride_length = np.random.uniform(50, 120, 50)  # Simulating stride lengths in cm
stride_velocity = np.random.uniform(20, 100, 50)  # Simulating stride velocities in cm/s

# Adding outliers for stride velocity
stride_velocity_outliers = np.append(stride_velocity, [200, 250, 300])  # Extreme outliers
stride_length_outliers = np.append(stride_length, [60, 70, 80])  # Random lengths for outliers

# Creating the scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(stride_length_outliers, stride_velocity_outliers, color='blue', label='Data Points')
plt.axhline(y=150, color='red', linestyle='--', label='Outlier Threshold (Velocity > 150 cm/s)')
plt.title('Scatterplot of Stride Length vs. Stride Velocity with Outliers', fontsize=14)
plt.xlabel('Stride Length (cm)', fontsize=12)
plt.ylabel('Stride Velocity (cm/s)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
