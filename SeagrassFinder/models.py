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
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class DenseNetSeagrass(nn.Module):
    def __init__(self, version="201", second_last_layer_size=256, last_layer_size=512):
        super().__init__()

        # Load pretrained model
        if version == "201":
            base_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        elif version == "121":
            base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported DenseNet version: {version}")

        # Freeze pretrained weights
        for param in base_model.parameters():
            param.requires_grad = False

        # Replace classifier
        num_features = base_model.classifier.in_features
        base_model.classifier = nn.Linear(num_features, second_last_layer_size)

        # Custom classification head
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(second_last_layer_size, last_layer_size),
            nn.ReLU(),
            nn.Linear(last_layer_size, 1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base_model(x)
        out = self.classifier(features)
        return out.squeeze(1)


class ViTSeagrass(nn.Module):
    def __init__(self, second_last_layer_size=512, last_layer_size=512):
        super().__init__()

        # Load pretrained ViT
        self.backbone = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Custom head (ViT outputs 1000 features)
        self.classifier = nn.Sequential(
            nn.Linear(1000, second_last_layer_size),
            nn.ReLU(),
            nn.Linear(second_last_layer_size, last_layer_size),
            nn.ReLU(),
            nn.Linear(last_layer_size, 1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out.squeeze(1)
