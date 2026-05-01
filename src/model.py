import torch
import torch.nn as nn
from torchvision import models

class DefectClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DefectClassifier, self).__init__()
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace final layer for our binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)