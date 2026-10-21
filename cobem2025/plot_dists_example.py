# This code will create 2 plots 
# The linear distribution will be a simple uniform distribution
# The logarithmic distribution will be a log-uniform distribution


import matplotlib.pyplot as plt
import numpy as np

# Define the bounds
lb = 1e-1
ub = 1e3

# Define the minimum and maximum values
min_x = -1
max_x = 1

# Define the number of points
n_points = 200

# Init x vector
x_upper = np.linspace(0, max_x, n_points)
x_lower = np.linspace(-max_x, 0, n_points)
x = np.concatenate([x_lower, x_upper])

# Create the linear distribution
linear_distribution_upper = lb + x_upper * (ub - lb)
linear_distribution_lower = -(-lb + x_lower * (-ub + lb))
linear_distribution = np.concatenate([linear_distribution_lower, linear_distribution_upper])

# Create the logarithmic distribution
slud_distribution = np.sign(x) * lb * np.power(ub/lb, np.abs(x))


# Create the first plot (linear scale) as a square figure for better side-by-side comparison in a paper
plt.figure(figsize=(7, 7))  # Square aspect ratio
plt.plot(x, linear_distribution, label='Linear Distribution', linewidth=2, color='blue', marker='x', markersize=2)
plt.plot(x, slud_distribution, label='SLUD Distribution', linewidth=2, color='red', marker='o', markersize=2)
plt.xlabel('Normalized Variable x')
plt.ylabel('Physical Value')
plt.title('Distribution Comparison: Linear vs SLUD for 200 points')
plt.legend()
plt.grid(True, alpha=0.3)

# Add zoomed inset in bottom right corner
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

axins = inset_axes(
    plt.gca(),
    width="30%",   # slightly larger for clarity in a square plot
    height="30%",
    loc='lower right',
    borderpad=1.6
)

# Define zoom region
zoom_x_min, zoom_x_max = -0.02, 0.02
zoom_mask = (x >= zoom_x_min) & (x <= zoom_x_max)

# Plot zoomed region
axins.plot(x[zoom_mask], linear_distribution[zoom_mask], linewidth=1.5, color='blue', marker='x', markersize=4)
axins.plot(x[zoom_mask], slud_distribution[zoom_mask], linewidth=1.5, color='red', marker='o', markersize=2)
axins.set_xlim(zoom_x_min, zoom_x_max)
axins.grid(True, alpha=0.3)
axins.set_title('Zoom: x ∈ [-0.02, 0.02]', fontsize=8)

plt.tight_layout()
plt.savefig('dists_example.png', dpi=300, bbox_inches='tight')
# plt.show()

# Create the second plot (log scale) - only positive half, also as a square figure
plt.figure(figsize=(7, 7))  # Square aspect ratio
# Filter for positive values only
positive_mask = x > 0
x_positive = x[positive_mask]
linear_positive = linear_distribution[positive_mask]
slud_positive = slud_distribution[positive_mask]

plt.semilogy(x_positive, linear_positive, label='Linear Distribution (Positive Half)', marker='x', linewidth=2, color='blue')
plt.semilogy(x_positive, slud_positive, label='SLUD Distribution (Positive Half)', marker='o', linewidth=2, color='red')
plt.xlabel('Normalized Variable x (Positive Half)')
plt.ylabel('Physical Value (Log Scale)')
plt.title('Distribution Comparison: Linear vs SLUD (Positive Half Only) for 100 points')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dists_example_semilogy.png', dpi=300, bbox_inches='tight')
# plt.show()