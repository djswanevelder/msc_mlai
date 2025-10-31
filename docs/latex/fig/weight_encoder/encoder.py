import wandb
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
WANDB_RUN_PATH = "/25205269-stellenbosch-university/weight-space-ae-pca/runs/e5u0alh7"
TRAIN_LOSS_KEY = 'train_loss_epoch'
VAL_LOSS_KEY = 'val_loss' # Key for validation loss
X_AXIS_KEY = 'epoch'
OUTPUT_FILENAME = "combined_loss_plot.png"

# --- NEW CONFIGURATION FOR TRUNCATION AND VISUALS ---
MAX_EPOCH_TO_PLOT = 80 # Set the maximum epoch to display (e.g., 25 for a plateau at 20)
LINE_THICKNESS = 6     # Increased line thickness for better visibility
TICK_FONTSIZE = 18     # New: Increased font size for axis tick labels
LEGEND_FONTSIZE = 24   # New: Increased font size for the legend

print("--- Accessing W&B Run Data and Plotting ---")

# Access the data
api = wandb.Api()
run = api.run(WANDB_RUN_PATH)
hist = run.history()

# --- Data Preparation (Robust Workflow) ---

# 1. Create a clean DataFrame for Training Loss
train_data = hist[[X_AXIS_KEY, TRAIN_LOSS_KEY]].dropna(subset=[TRAIN_LOSS_KEY])

# 2. Create a clean DataFrame for Validation Loss
val_data = hist[[X_AXIS_KEY, VAL_LOSS_KEY]].dropna(subset=[VAL_LOSS_KEY])


# Check for data presence
if train_data.empty or val_data.empty:
    print(f"Error: Either Training Loss ({TRAIN_LOSS_KEY}) or Validation Loss ({VAL_LOSS_KEY}) data is missing or entirely null. Cannot plot.")
    # In a real script, you'd exit() here
    # exit() 
else:

    # --- TRUNCATE DATA BASED ON MAX_EPOCH_TO_PLOT ---
    train_data = train_data[train_data[X_AXIS_KEY] <= MAX_EPOCH_TO_PLOT]
    val_data = val_data[val_data[X_AXIS_KEY] <= MAX_EPOCH_TO_PLOT]

    print(f"Plotting data truncated to Epoch {MAX_EPOCH_TO_PLOT} to focus on learning dynamics.")


    # Setup Matplotlib Figure with a single axis and minimal style
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot Training Loss (Primary / Left Axis) ---
    train_color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=LEGEND_FONTSIZE)
    ax1.set_ylabel('Training Loss', fontsize=LEGEND_FONTSIZE)
    train_line, = ax1.plot(
        train_data[X_AXIS_KEY],
        train_data[TRAIN_LOSS_KEY],
        label='Training Loss',
        color=train_color,
        linewidth=LINE_THICKNESS,
    )
    # New: Apply TICK_FONTSIZE to the y-axis ticks
    ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    # New: Apply TICK_FONTSIZE to the x-axis ticks
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE) 
    ax1.set_xlim(left=0, right=MAX_EPOCH_TO_PLOT) 

    # --- Plot Validation Loss (Secondary / Right Axis) ---
    ax2 = ax1.twinx() # Creates a second axes that shares the same x-axis
    val_color = 'tab:orange'
    ax2.set_ylabel('Validation Loss', fontsize=LEGEND_FONTSIZE)
    val_line, = ax2.plot(
        val_data[X_AXIS_KEY],
        val_data[VAL_LOSS_KEY],
        label='Validation Loss',
        color=val_color,
        linewidth=LINE_THICKNESS,
        linestyle='--'
    )
    # New: Apply TICK_FONTSIZE to the secondary y-axis ticks
    ax2.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    # --- Combine Legends ---
    # Combine handles and labels from both axes for a single legend box
    lines = [train_line, val_line]
    labels = [l.get_label() for l in lines]
    # New: Apply LEGEND_FONTSIZE
    ax1.legend(lines, labels, loc='upper right', frameon=False, fontsize=LEGEND_FONTSIZE)

    # --- Final Touches ---
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=300)

    print(f"\n--- Plot saved successfully as: {OUTPUT_FILENAME} ---")