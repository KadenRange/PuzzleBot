# load_pgn/cnn/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for attention-based feature recalibration"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        return F.relu(out + residual)

class TacticsResNet(nn.Module):
    """
    Enhanced ResNet architecture specifically designed for chess tactics recognition
    with deep residual connections and attention mechanisms
    """
    def __init__(self, num_classes, in_channels=27, num_filters=128, num_blocks=8, drop=0.4):
        super().__init__()
        
        # Initial convolution layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # First residual stage - tactical feature extraction
        self.stage1 = nn.Sequential(
            ResBlock(num_filters, use_se=True),
            ResBlock(num_filters, use_se=False),
            ResBlock(num_filters, use_se=True)
        )
        
        # Second residual stage with spatial downsampling
        self.transition1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU()
        )
        
        self.stage2 = nn.Sequential(
            ResBlock(num_filters*2, use_se=True),
            ResBlock(num_filters*2, use_se=False),
            ResBlock(num_filters*2, use_se=True)
        )
        
        # Third residual stage with further feature abstraction
        self.transition2 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU()
        )
        
        self.stage3 = nn.Sequential(
            ResBlock(num_filters*4, use_se=True),
            ResBlock(num_filters*4, use_se=False)
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop * 0.75),  # Reduced dropout in later layers
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# Keep the old ImprovedTacticCNN class for compatibility
class ImprovedTacticCNN(nn.Module):
    def __init__(self, num_classes, drop=0.5):
        super().__init__()
        # initial conv - now accepts 27 channels
        self.init = nn.Sequential(
            nn.Conv2d(27, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # block 1 - more filters, added SE attention
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128, use_se=True),
            ResBlock(128, use_se=False),
            nn.MaxPool2d(2, 2)
        )
        
        # block 2 - deeper with more filters
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256, use_se=True),
            ResBlock(256, use_se=False),
            nn.MaxPool2d(2, 2)
        )
        
        # block 3 - new layer for deeper representation
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResBlock(512, use_se=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Global pooling to handle variable input sizes better
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # classifier with dropout for regularization
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(drop * 0.5),  # Less dropout in later layers
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        return self.fc(x)