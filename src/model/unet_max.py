import torch.nn as nn


class UnetMax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        B, C, H, W, D = x.shape
        # (B, C, H, W, D) -> (B * D, C, H, W)
        x = x.view(B * D, C, H, W)
        x = self.model(x)  # (B * D, H, W)
        # (B * D, H, W) -> (B, D, H, W)
        x = x.view(B, D, H, W)
        # (B, D, H, W) -> (B, H, W)
        x = x.max(1)[0]
        return x
