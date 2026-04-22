import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import os
import pandas as pd
import torchvision
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryCalibrationError
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from seagrass_modules import SeagrassLightningModule, get_transforms, SeagrassDataset, SeagrassDataModule
from models import DenseNetSeagrass, ViTSeagrass


def train_seagrass_model(model_name, batch_size, learning_rate, max_epochs, 
                        train_transects, test_transects, data_folder):

    # 1. Data Loading by Transect
    train_dfs = []
    for transect in train_transects:
        transect_folder = os.path.join(data_folder, str(transect))
        df = pd.read_csv(os.path.join(transect_folder, f"{transect}.csv"))
        df["image_name"] = df['image_name'].apply(
            lambda x: os.path.join(transect_folder, x)
        )
        train_dfs.append(df)

    train_data = pd.concat(train_dfs).sample(frac=1).reset_index(drop=True)
    
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    test_dfs = []
    for transect in test_transects:
        transect_folder = os.path.join(data_folder, str(transect))
        df = pd.read_csv(os.path.join(transect_folder, f"{transect}.csv"))
        df["image_name"] = df['image_name'].apply(
            lambda x: os.path.join(transect_folder, x)
        )
        test_dfs.append(df)

    test_data = pd.concat(test_dfs).sample(frac=1).reset_index(drop=True)

    # 2. Model Instantiation
    if model_name.startswith("densenet"):
        version = model_name.split("-")[1]
        model = DenseNetSeagrass(version=version)
    elif model_name == "vit":
        model = ViTSeagrass()
    # ... other models

    # 3. Lightning Setup
    lit_model = SeagrassLightningModule(model, learning_rate=learning_rate)

    dm = SeagrassDataModule(train_df=train_data,
                        val_df=val_data,
                        test_df=test_data,
                        batch_size=batch_size,
                        model_name=model_name)
    
    # 4. Training Configuration
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=TensorBoardLogger('logs', name=f'{model_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5),
        ],
        devices=1,
        accelerator="gpu"
    )

    
    # 5. Execute Training
    trainer.fit(lit_model, datamodule = dm)
    trainer.test(lit_model, datamodule = dm)
