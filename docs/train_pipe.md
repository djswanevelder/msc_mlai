# Training Pipeline
The resNet18 model will be trained, with the final layer changed to output the number of classes trained on for a training instance



## `sweep.py`
- Find all classes used and specified in `sweep.csv` and download them
- Robustly start training instances, tracking status incase of interruption 
- Iterate over each line in `sweep.csv` and training the model as specified

## `restNet18.py`
- Train a resNet18 model with the final layer changed to only 3 classes
- `config.yaml` as input with various funcitonalities
    - Adam and SGD Optimizers
    - Classes to train on specified
    - Seed specified
    - `early_epoch`: the first epoch to locally store weights
    - `max_epoch`: the number of training epochs
    - `store_weight`: Whether or not the weight is also stored on WandB

### ResNet18 Training Descisions
- Model
    - Initialization : kaiming_normal, fan_out, relu, 0 Bias
    - Loss : Cross entropy
    - Output : Softmax `num_classes`
    - Learning Rate: Default, Adam + SGD
- Data Pre-processing
    - 244x244
    - Normalizated by Mean and Variance (Calculated)
        
## Tech Stack
- Pytorch Lightnight
- Hydra
- Weights and Biases: Logging + Weight Storage
- UV toml
