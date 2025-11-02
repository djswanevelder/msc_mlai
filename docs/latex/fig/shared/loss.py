import wandb
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
# New W&B Run Path for the latent encoder tuning project
WANDB_RUN_PATH = "25205269-stellenbosch-university/weight-latent-encoder/runs/cphggrhz"
# X_AXIS_KEY is removed as the index will be used for epoch
MAX_EPOCH_TO_PLOT = 300 # Set the maximum epoch to display
LINE_THICKNESS = 6     # Increased line thickness for better visibility
TICK_FONTSIZE = 18     # Increased font size for axis tick labels
LEGEND_FONTSIZE = 24   # Increased font size for the legend
DPI_SETTING = 300      # High resolution for saving

print("--- Accessing W&B Run Data ---")

# Access the data
try:
    api = wandb.Api()
    run = api.run(WANDB_RUN_PATH)
    hist = run.history()
except Exception as e:
    print(f"Error accessing W&B data for run {WANDB_RUN_PATH}. Ensure you are logged in or the path is correct.")
    print(f"Details: {e}")
    # Exit if we cannot fetch data
    exit()

def plot_wandb_metrics(history_df, train_key, val_key, title, output_filename, max_epoch, line_thickness, tick_fontsize, legend_fontsize):
    """
    Plots training and validation metrics using a SHARED Y-axis and the DataFrame index as epoch
    for the x-axis, adhering to strict style guidelines.
    """
    print(f"\n--- Processing plot for {title} (File: {output_filename}) ---")

    # 1. Create clean DataFrames, dropping rows where the metric is NaN
    train_data = history_df[[train_key]].dropna()
    val_data = history_df[[val_key]].dropna()

    # Get the index (epoch) source.
    train_epochs = train_data.index.to_series(name='epoch_idx')
    val_epochs = val_data.index.to_series(name='epoch_idx')

    # Check for data presence
    if train_data.empty or val_data.empty:
        print(f"Warning: Data for '{train_key}' or '{val_key}' is missing or incomplete. Skipping plot.")
        return

    # --- TRUNCATE DATA ---
    # Filter the data using the index values (epochs)
    train_data = train_data[train_epochs <= max_epoch]
    train_epochs = train_epochs[train_epochs <= max_epoch]
    
    val_data = val_data[val_epochs <= max_epoch]
    val_epochs = val_epochs[val_epochs <= max_epoch]

    print(f"Plotting data truncated to Epoch {max_epoch}.")

    # Setup Matplotlib Figure
    plt.style.use('default')
    # Use only one axes object, ax1
    fig, ax1 = plt.subplots(figsize=(12, 7)) 

    # --- Setup Shared Axes Labels and Limits ---
    ax1.set_xlabel('Epoch', fontsize=legend_fontsize)
    # Set a combined Y-axis label
    ax1.set_ylabel(f'{title} Value', fontsize=legend_fontsize) 
    
    # Apply X-axis TICK_FONTSIZE and limits
    ax1.tick_params(axis='x', labelsize=tick_fontsize) 
    ax1.set_xlim(left=0, right=max_epoch) 
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Plot Training Data ---
    train_color = 'tab:blue'
    
    # Plot the line on ax1
    train_line, = ax1.plot(
        train_epochs, 
        train_data[train_key],
        label=f'Training {title}',
        color=train_color,
        linewidth=line_thickness,
    )
    
    # Apply Y-axis TICK_FONTSIZE 
    ax1.tick_params(axis='y', labelsize=tick_fontsize)

    # --- Plot Validation Data ---
    val_color = 'tab:orange'
    
    # Plot the line on ax1 (NO ax2 needed)
    val_line, = ax1.plot(
        val_epochs, 
        val_data[val_key],
        label=f'Validation {title}',
        color=val_color,
        linewidth=line_thickness,
        linestyle='--'
    )
    
    # --- Combine Legends ---
    lines = [train_line, val_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, fontsize=legend_fontsize, facecolor='white', edgecolor='gray')

    # --- Final Touches and Save ---
    plt.tight_layout()
    plt.savefig(output_filename, dpi=DPI_SETTING)
    plt.close(fig) # Close the figure to free memory

    print(f"Plot saved successfully as: {output_filename}")


# --- Plot NT-Xent Loss (Using the function with shared axis) ---
plot_wandb_metrics(
    history_df=hist,
    train_key='train/avg_nt_xent_loss',
    val_key='val/avg_nt_xent_loss',
    title='NT-Xent Contrastive Loss',
    output_filename='nt_xent_loss_plot.png', # Changed filename to reflect change
    max_epoch=MAX_EPOCH_TO_PLOT,
    line_thickness=LINE_THICKNESS,
    tick_fontsize=TICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE,
)

# You can add the MSE plot back if you need it, using the same function:
# plot_wandb_metrics(
#     history_df=hist,
#     train_key='train/mse_loss',
#     val_key='val/mse_loss',
#     title='MSE Reconstruction Loss',
#     output_filename='mse_loss_plot_shared_axis.png',
#     max_epoch=MAX_EPOCH_TO_PLOT,
#     line_thickness=LINE_THICKNESS,
#     tick_fontsize=TICK_FONTSIZE,
#     legend_fontsize=LEGEND_FONTSIZE,
# )

print("\n--- All plots saved successfully. ---")