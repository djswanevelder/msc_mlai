import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from typing import List

# --- START: Mock Data Section ---

# all_results: List[List[float]] = [
#     [0.887, 0.821, 0.914],
#     [0.912, 0.850, 0.923],
#     [0.855, 0.799, 0.880],
#     [0.941, 0.888, 0.955],
#     [0.867, 0.803, 0.911],
#     [0.925, 0.860, 0.930],
#     [0.890, 0.830, 0.900],
#     [0.950, 0.895, 0.960]
# ]
# --- END: Mock Data Section ---
CSV_FILE_PATH = 'encoding_comparison_stats.csv'

all_results: List[List[float]] = np.loadtxt(
    CSV_FILE_PATH, 
    delimiter=',', 
    skiprows=1, 
    dtype=float
).tolist()

data_array = np.array(all_results)
X = data_array[:, 0]
Y = data_array[:, 1]


slope, intercept, r_value, _, _ = linregress(X, Y)
r_squared = r_value**2
r_squared_str = f"R$^2$ = {r_squared:.4f}"

plt.figure(figsize=(8, 6))

plt.scatter(X, Y, color='#1f77b4', label="Data Points", alpha=0.8, edgecolors='w', s=60)

regression_line = slope * X + intercept
plt.plot(X, regression_line, color='red', 
         label=f'Linear Fit', 
         linestyle='--', linewidth=4)

# plt.title(f"Original vs. Reconstructed Accuracy\n($\mathbf{{N={len(X)}}}$, {r_squared_str})", fontsize=14, fontweight='bold')
plt.xlabel("Original Accuracy", fontsize=18)
plt.ylabel("Reconstructed Accuracy (Y)", fontsize=18)

plt.text(min(X), max(Y) - 0.02, r_squared_str, fontsize=16)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
