import numpy as np
import torch
from torch import nn

def save_best_record(test_info, file_path):
    """
    Save the best test results to a file.
    
    Args:
        test_info: Dictionary containing epoch, test_AUC, and test_PR lists
        file_path: Path to save the results
    """
    f = open(file_path, "w")
    f.write("epoch: {}\n".format(test_info["epoch"][-1]))
    f.write("test_AUC: {}\n".format(test_info["test_AUC"][-1]))
    f.write("test_PR: {}\n".format(test_info["test_PR"][-1]))
    f.close()

def FeedForward(dim, repe=4, dropout=0.):
    """
    Feed-forward network with GELU activation.
    
    Args:
        dim: Input/output dimension
        repe: Expansion factor (default: 4)
        dropout: Dropout rate
    
    Returns:
        Sequential feed-forward module
    """
    return nn.Sequential(
        nn.Linear(dim, dim * repe),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * repe, dim),
        nn.GELU(),
    )

class DECOUPLED(nn.Module):
    """
    Decoupled spatial-temporal convolution block.
    Processes spatial (2D) and temporal (1D) dimensions separately.
    """
    def __init__(
        self,
        dim,
        heads,
        kernel=3
    ):
        super().__init__()
        self.heads = heads
        self.norm2d = nn.BatchNorm2d(dim)
        self.norm1d = nn.BatchNorm1d(dim)
        self.conv2d = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=heads)
        self.conv1d = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=heads)

    def forward(self, x):
        """
        Forward pass with decoupled spatial-temporal convolution.
        
        Input shape: (B, T, H, W, C)
        Output shape: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # Spatial convolution: process each time step
        x = x.view(B * T, C, H, W)
        x = self.norm2d(x)
        x = self.conv2d(x)
        
        # Temporal convolution: process each spatial location
        x = x.view(B * H * W, C, T)
        x = self.norm1d(x)
        x = self.conv1d(x)
        
        # Reshape back to original format
        x = x.view(B, T, H, W, C)
        return x