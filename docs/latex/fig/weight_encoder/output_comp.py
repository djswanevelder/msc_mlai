import wandb
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration (Copied and adapted from your first example) ---
WANDB_RUN_PATH = "/25205269-stellenbosch-university/weight-space-ae-pca/runs/e5u0alh7"
# Keys for the three metrics requested
CORR_KEY = 'func_test_output_correlation'
COS_KEY = 'func_test_output_cosine_similarity'
AGREE_KEY = 'func_test_prediction_agreement'
X_AXIS_KEY = 'epoch'
OUTPUT_FILENAME = "output_comparison.png" # Changed filename

# --- NEW CONFIGURATION FOR TRUNCATION AND VISUALS ---
MAX_EPOCH_TO_PLOT = 80 # Set the maximum epoch to display
LINE_THICKNESS = 6     # Increased line thickness for better visibility
TICK_FONTSIZE = 18     # Increased font size for axis tick labels
LEGEND_FONTSIZE = 24   # Increased font size for the legend
LABEL_FONTSIZE = 24    # New: Setting font size for X and Y axis labels


# Define colors for the three lines
COLOR_CORR = 'tab:blue'
COLOR_COS = 'tab:green'
COLOR_AGREE = 'tab:orange'

print("--- Accessing W&B Run Data and Plotting ---")

# Access the data
api = wandb.Api()
run = api.run(WANDB_RUN_PATH)
hist = run.history()

# --- Data Preparation (Robust Workflow for three metrics) ---

# Create a clean DataFrame for each metric by dropping rows where the metric is NaN
corr_data = hist[[X_AXIS_KEY, CORR_KEY]].dropna(subset=[CORR_KEY])
cos_data = hist[[X_AXIS_KEY, COS_KEY]].dropna(subset=[COS_KEY])
agree_data = hist[[X_AXIS_KEY, AGREE_KEY]].dropna(subset=[AGREE_KEY])


# Check for data presence
if corr_data.empty and cos_data.empty and agree_data.empty:
    print("Error: No data found for any of the requested evaluation metrics. Cannot plot.")
    # In a real script, you'd exit() here
    # exit()
else:
    
    # --- TRUNCATE DATA BASED ON MAX_EPOCH_TO_PLOT ---
    corr_data = corr_data[corr_data[X_AXIS_KEY] <= MAX_EPOCH_TO_PLOT]
    cos_data = cos_data[cos_data[X_AXIS_KEY] <= MAX_EPOCH_TO_PLOT]
    agree_data = agree_data[agree_data[X_AXIS_KEY] <= MAX_EPOCH_TO_PLOT]
    print(f"Plotting data truncated to Epoch {MAX_EPOCH_TO_PLOT} to focus on learning dynamics.")


    # Setup Matplotlib Figure with a single axis and minimal style
    plt.style.use('default') 
    fig, ax1 = plt.subplots(figsize=(12, 8)) # Increased figure size slightly for larger fonts

    # --- Axis 1: All Metrics (Primary Left Axis) ---
    # Apply new LABEL_FONTSIZE
    ax1.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE) 
    ax1.set_ylabel('Evaluation Score', fontsize=LABEL_FONTSIZE) 

    # Plot Correlation
    corr_line, = ax1.plot(
        corr_data[X_AXIS_KEY], 
        corr_data[CORR_KEY], 
        label='Correlation', 
        color=COLOR_CORR, 
        linewidth=LINE_THICKNESS, # Apply LINE_THICKNESS
        linestyle='-' 
    )
    # Plot Cosine Similarity
    cos_line, = ax1.plot(
        cos_data[X_AXIS_KEY], 
        cos_data[COS_KEY], 
        label='Cosine Similarity', 
        color=COLOR_COS, 
        linewidth=LINE_THICKNESS, # Apply LINE_THICKNESS
        linestyle='--' 
    )
    # Plot Prediction Agreement
    agree_line, = ax1.plot(
        agree_data[X_AXIS_KEY], 
        agree_data[AGREE_KEY], 
        label='Prediction Agreement', 
        color=COLOR_AGREE, 
        linewidth=LINE_THICKNESS, # Apply LINE_THICKNESS
        linestyle=':' 
    )

    # Clean up axes appearance for minimal style
    # Apply TICK_FONTSIZE to both X and Y axis ticks
    ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    
    # Set X-axis limit based on MAX_EPOCH_TO_PLOT
    ax1.set_xlim(left=0, right=MAX_EPOCH_TO_PLOT) 
    
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # --- Combine Legends ---
    lines = [corr_line, cos_line, agree_line]
    labels = [l.get_label() for l in lines]
    # Apply LEGEND_FONTSIZE
    ax1.legend(lines, labels, 
               frameon=False, 
               fontsize=LEGEND_FONTSIZE) 

    # --- Final Touches ---
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=300) 

    print(f"\n--- Enhanced single-axis plot saved successfully as: {OUTPUT_FILENAME} ---")