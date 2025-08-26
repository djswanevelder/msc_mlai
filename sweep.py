import hydra
from omegaconf import DictConfig, OmegaConf
from resNet18 import train_restNet18


my_config = {
    "data": {
        "data_path": "./imagenet_subsets",
        "classes": [
            "red_fox",
            "hyena",
            "doberman"
        ],
        "batch_size": 32
    },
    "training": {
        "optimizer": "Adam",
        "max_epoch": 2,
        "early_epoch": 1,

    },
    "seed": 42,
    "wandb": {
        "project_name": "MSc_MLAI",
        "run_name": "No Weight Run",
        "store_weight": "True",
    }
}

def sample_uniform_classes(csv_filepath:str)->None:
    

@hydra.main(version_base=None, config_path=None, config_name="config")
def my_app(cfg: DictConfig) -> None:

    

if __name__ == "__main__":
    # Create the DictConfig object from the Python dictionary
    cfg = OmegaConf.create(my_config)
    
    # You can now run the app with this config
    my_app(cfg)