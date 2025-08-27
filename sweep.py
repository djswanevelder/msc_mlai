import hydra
from omegaconf import DictConfig, OmegaConf
from resNet18 import train_restNet18
import pandas as pd

def sweep_over_all() -> None:
    df = pd.read_csv('sweep.csv')
    for index, row in df.iterrows():
        my_config = {
            "data": {
                "data_path": "./imagenet_subsets",
                "classes": [
                    row['class1'],
                    row['class2'],
                    row['class3']
                ],
                "batch_size": 32
            },
            "training": {
                "optimizer": row['optimizer'],
                "max_epoch": int(row['max_epoch']),
                "early_epoch": int(row['early_epoch']),

            },
            "seed": int(row['seed']),
            "wandb": {
                "project_name": "MSc_MLAI",
                "run_name": "No Weight Run",
                "store_weight": bool(row['store_weight']),
            }
        }
        cfg = OmegaConf.create(my_config)
        train_restNet18(cfg)

def find_all_classes():
    df = pd.read_csv('sweep.csv')
    cl1 = df['class1'].unique()
    cl2 = df['class2'].unique()
    cl3 = df['class3'].unique()
    return pd.concat([pd.Series(cl1), pd.Series(cl2), pd.Series(cl3)]).unique()



if __name__ == "__main__":
    all_classes = find_all_classes()
    # print(f'{all_classes}')
    # sweep_over_all()

    my_config = {
        "classes":{all_classes}
    }
    print(f'{my_config}')