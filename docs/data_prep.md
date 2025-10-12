# Data Preparation and Configuration Scripts (`src/data_prep/`)

This directory contains scripts essential for dataset acquisition, class subset generation, and training configuration orchestration for the base models.

***

## `download_imagenet.py`

| Field | Description |
| :--- | :--- |
| **Purpose** | To download and extract specific ImageNet class subsets required for the meta-learning task. |
| **Functionality** | <ul><li>Reads target classes from a Hydra configuration object (`cfg.classes`).</li><li>Resolves the **absolute path** for the download directory using the `dataset_path` argument.</li><li>Checks for existing files/archives to ensure **idempotency** (avoiding re-downloading or re-extracting partially completed classes).</li><li>Downloads the corresponding ImageNet TAR archive for each class via HTTP.</li><li>Extracts images and cleans up the `.tar` file.</li></ul>

***

## `generate_sweep_config.py`

| Field | Description |
| :--- | :--- |
| **Purpose** | To algorithmically generate the full set of training configurations (a "sweep") for the base ResNet-18 models. |
| **Functionality** | <ul><li>Reads all available class names from the ImageNet mapping file (e.g., `imagenet_map.txt`).</li><li>Randomly generates a specified number of **unique 3-class subsets**.</li><li>Assigns random, but controlled, **hyperparameters** (e.g., `seed`, `max_epoch`, `learning_rate`, `optimizer`) to each class subset.</li><li>Outputs the complete list of configurations as a structured file (e.g., `sweep.csv`) used by the subsequent training script.</li></ul>

***

## `resNet18.py`

| Field | Description |
| :--- | :--- |
| **Purpose** | Defines the ResNet-18 model architecture used as the base classifier in the meta-learning task. |
| **Functionality** | <ul><li>Defines the `ResNet18Classifier` PyTorch module.</li><li>Instantiates the base `torchvision.models.resnet18` architecture.</li><li>Replaces the final fully connected (FC) layer to match the required number of output classes (typically 3 for the subsets).</li><li>Manages custom weight initialization for the final layer.</li></ul>

***

## `train_models.py`

| Field | Description |
| :--- | :--- |
| **Purpose** | Orchestrates the training and validation of the base ResNet-18 models across all configurations defined in the sweep file. |
| **Functionality** | <ul><li>Reads hyperparameters and class lists for a specific training instance from the input **`sweep.csv`** file.</li><li>Updates the `status` column of the current training instance (row) within **`sweep.csv`** after each epoch to maintain a record of training progress.</li><li>Integrates PyTorch Lightning (`pl.LightningModule`) for structured training loops and logging.</li><li>**Data Preparation:** Filters the dataset based on the current configuration's classes, splits into train/validation sets, and calculates/applies normalization (mean/std).</li><li>Initializes the `ResNetClassifier` with sweep-defined hyperparameters.</li><li>Logs the entire configuration (`cfg`) to **Wandb**.</li><li>Uses a custom callback (`StateDictSaver`) to save the model's weights (`.pth` file) at a critical early epoch and at completion, and logs them as a Wandb artifact.</li><li>Executes the training and validation loop using `pl.Trainer`.</li></ul>