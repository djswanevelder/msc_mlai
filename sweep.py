from omegaconf import DictConfig, OmegaConf
from resNet18 import train_restNet18
import pandas as pd
from download import download
import time

def sweep_over_all(input_filename) -> None:
    df = pd.read_csv(input_filename)
    for index, row in df.iterrows():
        if df['status'][index] == 'Done':
            print(f"Skipping stef{index}")
            continue
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
                "store_weight": bool(row['store_weight']),
            }
        }
        cfg = OmegaConf.create(my_config)
        df.loc[index, "status"] = "Busy"
        df.to_csv(input_filename, index=False)
        
        # train_restNet18(cfg)
        time.sleep(5)

        df.loc[index, "status"] = "Done"
        df.to_csv(input_filename, index=False)

def download_all_classes(input_filename):
    df = pd.read_csv(input_filename)
    cl1 = df['class1'].str.lower().unique()
    cl2 = df['class2'].str.lower().unique()
    cl3 = df['class3'].str.lower().unique()
    all_classes = pd.concat([pd.Series(cl1), pd.Series(cl2), pd.Series(cl3)]).unique()
    data_config = {"classes":all_classes.tolist()}
    data_cfg = OmegaConf.create(data_config)
    download(data_cfg)

if __name__ == "__main__":
    # download_all_classes('sweep.csv')
    sweep_over_all('sweep.csv')
