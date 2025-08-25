# Training Pipeline
The resNet18 model will be trained, with the final layer changed to output the number of classes trained on for a training instance

The `config.yaml` will specify all of the different hyperparameter chosen for a training instance, and the `resNet18.py` will take this as input. using Hydra 



## ResNet18 Training Descisions
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
- Docker
