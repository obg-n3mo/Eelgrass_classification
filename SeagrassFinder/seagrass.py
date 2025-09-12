import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import os
import pandas as pd
import torchvision
import torchvision.models as models
from sklearn.model_selection import train_test_split
import tensorflow as tf
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryCalibrationError
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SeagrassLightningModule(pl.LightningModule):
    def __init__(self, model, model_name, train_data, val_data=None, 
                 test_data=None, learning_rate=1e-5, batch_size=128, 
                 second_last_layer_size=1024, last_layer_size=128):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = torch.nn.BCEWithLogitsLoss()

        # Metrics
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.auc = BinaryAUROC()
        self.calibration_error = BinaryCalibrationError(n_bins=5)
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())

        preds = torch.sigmoid(logits)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.accuracy(preds, y.int()), on_epoch=True, prog_bar=True)
        self.log("train_auc", self.auc(preds, y.int()), on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())
        preds = torch.sigmoid(logits)

        self.val_acc.update(preds, y.int())
    
        self.log("val_loss", loss, on_step=False, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step = False, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())
        preds = torch.sigmoid(logits)

        self.test_acc.update(preds, y.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
	return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



def get_transforms(model_name):
    if model_name.startswith("resnet"):
        version = model_name.split("-")[1]
        return getattr(torchvision.models, f"ResNet{version}_Weights").DEFAULT.transforms()
    elif model_name.startswith("densenet"):
        version = model_name.split("-")[1]
        return getattr(torchvision.models, f"DenseNet{version}_Weights").DEFAULT.transforms()
    elif model_name == "vit":
        return ViT_L_32_Weights.DEFAULT.transforms()
    elif model_name == "inception":
        return Inception_V3_Weights.DEFAULT.transforms()

class SeagrassDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name, label = row["image_name"], row["label"]

        # Load image on-demand with PIL
        image = Image.open(image_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class SeagrassDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df=None, test_df=None, 
                 batch_size=32, model_name="densenet-201"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = multiprocessing.cpu_count()
        self.model_name = model_name
        self.transform = get_transforms(model_name)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df



    def setup(self, stage=None):
        """Prepare datasets for different stages."""
        if self.train_df is not None:
            self.train_dataset = SeagrassDataset(self.train_df, transform=self.transform)
        if self.val_df is not None:
            self.val_dataset = SeagrassDataset(self.val_df, transform=self.transform)
        if self.test_df is not None:
            self.test_dataset = SeagrassDataset(self.test_df, transform=self.transform)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers= self.num_workers>0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers= self.num_workers>0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers= self.num_workers>0)
