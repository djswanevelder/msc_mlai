import wandb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm 

# --- Configuration ---

# Dictionary of all runs to plot (excluding 'BEST')
RUNS_TO_PLOT = {
    "[2048,1024,512]": "/25205269-stellenbosch-university/weight-space-ae-pca/runs/mv8k1ot6", # [2048, 1024, 512] -> 2048
    "[1024,512]": "/25205269-stellenbosch-university/weight-space-ae-pca/runs/wy7nhl5s", # [1024, 512] -> 1024
    "[512]": "/25205269-stellenbosch-university/weight-space-ae-pca/runs/fdtqcd8g", # [512] -> 512
    "[]": "/25205269-stellenbosch-university/weight-space-ae-pca/runs/92o3ejwx", # [] -> 256
}

VAL_LOSS_KEY = 'val_loss'
X_AXIS_KEY = 'epoch'
OUTPUT_FILENAME = "hidden_dim_loss.png"

# --- NEW CONFIGURATION FOR TRUNCATION AND VISUALS ---
MAX_EPOCH_TO_PLOT = 35 # Set the maximum epoch to display
LINE_THICKNESS = 5     # Line thickness
TICK_FONTSIZE = 18     # Font size for axis tick labels
LEGEND_FONTSIZE = 18   # Font size for the legend
LABEL_FONTSIZE = 18    # Font size for axis labels

# Matplotlib Colormaps for 4 lines (using a qualitative palette for distinction)
COLORS = plt.cm.get_cmap('Dark2', len(RUNS_TO_PLOT))


print("--- Accessing W&B Run Data and Plotting Validation Loss ---")
api = wandb.Api()
plt.style.use('default')

# Create a single axes (ax1) for plotting, which will be the left Y-axis
fig, ax1 = plt.subplots(figsize=(12, 8)) 

all_lines = []
all_labels = []

# --- PLOT LOOP ---

for idx, (run_label, run_path) in enumerate(RUNS_TO_PLOT.items()):
    try:
        run = api.run(run_path)
        hist = run.history()

        # 1. Create a clean DataFrame for Validation Loss
        val_data = hist[[X_AXIS_KEY, VAL_LOSS_KEY]].dropna(subset=[VAL_LOSS_KEY])

        if val_data.empty:
            print(f"Skipping {run_label}: Validation Loss data is missing.")
            continue

        # --- TRUNCATE DATA ---
        val_data = val_data[val_data[X_AXIS_KEY] <= MAX_EPOCH_TO_PLOT]

        # Get a unique color for the run
        color = COLORS(idx)
        
        # --- Plot Validation Loss on the Primary (Left) Axis (ax1) ---
        val_line, = ax1.plot(
            val_data[X_AXIS_KEY],
            val_data[VAL_LOSS_KEY],
            label=f'{run_label}', # Only need the run label now
            color=color,
            linewidth=LINE_THICKNESS,
            linestyle='-' # Use solid line
        )
        
        # Collect lines and labels for the combined legend
        all_lines.append(val_line)
        all_labels.append(val_line.get_label())

        print(f"Successfully processed and plotted Validation Loss for: {run_label}")

    except Exception as e:
        print(f"An error occurred accessing W&B run {run_path} ({run_label}): {e}")

# --- Final Plot Customization ---

# Set Axis Labels and Ticks for the LEFT Y-axis (ax1)
ax1.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE)
ax1.set_ylabel('Validation Loss', fontsize=LABEL_FONTSIZE, color='black') # Set Y-label
ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE, colors='black')
ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE) 

# Set X-axis limit
ax1.set_xlim(left=0, right=MAX_EPOCH_TO_PLOT) 

# --- Legend ---
if all_lines:
    # We use ax1.legend as there is only one axes now
    ax1.legend(all_lines, all_labels, loc='upper right', ncol=1, frameon=True, fontsize=LEGEND_FONTSIZE)

# --- Save Plot ---
plt.tight_layout()
plt.savefig(OUTPUT_FILENAME, dpi=300)

print(f"\n--- Plot saved successfully as: {OUTPUT_FILENAME} ---")