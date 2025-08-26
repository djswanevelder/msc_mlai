import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import pytorch_lightning as pl
from torchvision import models, datasets, transforms
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

class StateDictSaver(pl.Callback):
    """
    A custom callback to save only the model's state_dict at the end of training.
    """
    def __init__(self, log_to_wandb: bool,file_name:str):
        """
        Args:
            log_to_wandb (bool): If True, the state_dict will be logged as a W&B artifact.
        """
        self.log_to_wandb = log_to_wandb
        self.file_name = file_name

    def on_train_end(self, trainer, pl_module):
        best_model_path = trainer.checkpoint_callback.best_model_path
        
        checkpoint = torch.load(best_model_path)
        state_dict_path = os.path.join(trainer.default_root_dir, f'weights/{self.file_name}.pth')
        
        torch.save(checkpoint["state_dict"], state_dict_path)
        if self.log_to_wandb:
            artifact = wandb.Artifact("best-model-state-dict", type="model-weights")
            artifact.add_file(state_dict_path)
            trainer.logger.experiment.log_artifact(artifact)

class ResNetClassifier(pl.LightningModule):
    """
    A ResNet-18 classifier using PyTorch Lightning.
    """
    def __init__(self, num_classes: int, optimizer_name: str):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = models.resnet18(weights=None)
        # Change the final fully-connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Initialize the weights and biases of the new classification layer
        nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    # def on_validation_epoch_end(self):
    #     # A simple example of how to log extra information at the end of a validation epoch.
    #     # This will be logged to the W&B dashboard.
    #     # self.log('some_extra_metric', 1.0)

    def configure_optimizers(self):
        """
        Configures the optimizer based on the name passed from the Hydra config.
        """
        if self.hparams.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        elif self.hparams.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

def calculate_mean_std(dataset):
    """
    Calculates the mean and standard deviation of a dataset.
    This is necessary for normalization.
    
    Args:
        dataset: A PyTorch Dataset object.
        
    Returns:
        A tuple containing the mean and standard deviation as lists.
    """
    print("Calculating dataset mean and standard deviation...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Initialize variables for mean calculation
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    # Iterate through the data loader to calculate mean
    for images, _ in tqdm(loader, desc="Calculating Mean"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    
    # Reset total_samples for std calculation
    total_samples = 0
    
    # Iterate through the data loader again to calculate std
    for images, _ in tqdm(loader, desc="Calculating Std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).sum([0, 2])
        total_samples += batch_samples
    
    std = torch.sqrt(std / (total_samples * images.size(2)))
    
    return mean.tolist(), std.tolist()

def prepare_data(data_path, classes_to_use, batch_size):
    """
    Loads data from the file system, splits it into training and validation sets,
    and creates DataLoaders.
    """
    initial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    all_data = datasets.ImageFolder(data_path, transform=initial_transform)

    filtered_samples = []
    class_to_new_label = {cls: i for i, cls in enumerate(classes_to_use)}
    
    for path, label in all_data.samples:
        class_name = all_data.classes[label]
        if class_name in classes_to_use:
            new_label = class_to_new_label[class_name]
            filtered_samples.append((path, new_label))

    filtered_data = ImageFolder(
        root=data_path,
        loader=default_loader,
        transform=initial_transform,
    )
    filtered_data.samples = filtered_samples
    filtered_data.imgs = filtered_samples
    filtered_data.classes = classes_to_use
    filtered_data.class_to_idx = class_to_new_label
    filtered_data.targets = [s[1] for s in filtered_samples]
    
    total_size = len(filtered_data)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_data, val_data = random_split(filtered_data, [train_size, val_size])

    calculated_mean, calculated_std = calculate_mean_std(train_data)
    print(f"Calculated Mean: {calculated_mean}")
    print(f"Calculated Std: {calculated_std}")
    
    final_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=calculated_mean, std=calculated_std),
    ])

    train_data.dataset.transform = final_transform
    val_data.dataset.transform = final_transform

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, len(classes_to_use), calculated_mean, calculated_std

@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    The main training loop, now driven by a Hydra config file.
    """
    weight_name = f'{cfg.data.classes}_{cfg.training.optimizer}_{cfg.training.max_epochs}'

    pl.seed_everything(cfg.seed)
    wandb_logger = WandbLogger(
        project=cfg.wandb.project_name,
        name=cfg.wandb.run_name,
        log_model=False,
    )

    state_dict_saver_callback = StateDictSaver(log_to_wandb=cfg.wandb.store_weight,file_name = weight_name)


    train_loader, val_loader, num_classes, calculated_mean, calculated_std = prepare_data(
        data_path=cfg.data.data_path,
        classes_to_use=cfg.data.classes,
        batch_size=cfg.data.batch_size
    )
    model = ResNetClassifier(
        num_classes=num_classes,
        optimizer_name=cfg.training.optimizer
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[state_dict_saver_callback],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
