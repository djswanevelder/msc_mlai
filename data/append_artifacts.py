import pandas as pd
import wandb
import re
from tqdm import tqdm

# --- Configuration ---
ENTITY = "25205269-stellenbosch-university" 
PROJECT_NAME = "MSc_MLAI"
WANDB_PROJECT = f"{ENTITY}/{PROJECT_NAME}"

INPUT_CSV_FILE = 'data/wandb_export.csv'
OUTPUT_CSV_FILE = 'data/wandb_w_artifacts.csv'
# ---------------------

# Initialize the WandB API
api = wandb.Api()

print(f"Attempting to read data from: {INPUT_CSV_FILE}")

try:
    # 1. READ THE INPUT CSV FILE
    df = pd.read_csv(INPUT_CSV_FILE)
    
    if 'ID' not in df.columns:
        print("❌ Error: The required 'ID' column (WandB Run ID) was not found in the CSV.")
        exit()

    print(f"✅ Successfully loaded {len(df)} runs.")
    
except FileNotFoundError:
    print(f"❌ Error: The input file '{INPUT_CSV_FILE}' was not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# --- NEW STEP 1: CLEAN THE 'data.classes' COLUMN ---
if 'data.classes' in df.columns:
    print("Cleaning 'data.classes' column format...")
    # Use str.replace to remove quotes and double quotes, keeping only brackets and commas
    df['data.classes'] = df['data.classes'].astype(str).str.replace(r'\[\"', '[', regex=True)
    df['data.classes'] = df['data.classes'].str.replace(r'\"\]', ']', regex=True)
    df['data.classes'] = df['data.classes'].str.replace(r'\"\,\"', ',', regex=True)
    # Final check for simple string quotes left by export (e.g., "['item',...]")
    df['data.classes'] = df['data.classes'].str.replace(r'\"', '', regex=False)
else:
    print("Warning: 'data.classes' column not found for cleaning.")

# List to store formatted artifact data
artifacts_list = []

print(f"Starting artifact retrieval and filtering for runs in project: {WANDB_PROJECT}...")

# 2. ITERATE, FETCH, FILTER, AND FORMAT ARTIFACTS
for index, row in tqdm(df.iterrows(), total=len(df), desc="Fetching Artifacts"):
    run_id = row['ID']
    
    formatted_artifacts = []
    
    try:
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        artifacts = run.logged_artifacts()
        
        for a in artifacts:
            # CHECK 1: Filter to only include 'model-weights' type
            if a.type == 'model-weights':
                # Remove the version tag ':v0' (or any version)
                name_no_version = re.sub(r':v\d+$', '', a.name)
                
                # CHECK 2: Ensure the artifact name doesn't already end with .pth before appending
                if not name_no_version.lower().endswith('.pth'):
                    final_name = name_no_version + '.pth'
                else:
                    final_name = name_no_version
                
                formatted_artifacts.append(final_name)
                
        artifacts_list.append(formatted_artifacts)
    
    except Exception as e:
        print(f"\nWarning: Could not fetch artifacts for run ID '{run_id}'. Error: {e}")
        artifacts_list.append([f"Error: Could not retrieve artifacts for run ID {run_id}"])


# 3. APPEND THE NEW COLUMN TO THE DATAFRAME
df['artifacts'] = artifacts_list

# 4. SAVE THE RESULTING DATAFRAME TO A NEW CSV FILE
df.to_csv(OUTPUT_CSV_FILE, index=False)

print("\n--- Script Complete 🚀 ---")
print(f"Processed {len(df)} runs.")
print(f"DataFrame with cleaned data saved to {OUTPUT_CSV_FILE}")