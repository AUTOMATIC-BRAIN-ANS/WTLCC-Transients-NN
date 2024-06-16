from Models.model_utils import ConvolutionBlock, ResidualBlock, FCBlock
import torch.nn as nn
import torch

class InitialModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            ConvolutionBlock(1, 32, 3, 1),
            # ResidualBlock(32, 32, 3, 1),
            ConvolutionBlock(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1),
            # ResidualBlock(64, 64, 3, 1),
            # ResidualBlock(64, 64, 3, 1),
            # ResidualBlock(64, 64, 3, 1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            FCBlock(32, 32, 0.3),
            FCBlock(32, 32, 0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = torch.sigmoid(x.squeeze(1))
        # x = (x >= 0.5).float()
        return x