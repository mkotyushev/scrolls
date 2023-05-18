import torch.nn as nn


class Unet2d(nn.Module):
    """"Wrapper to convert a volume input to a 2D image with depth as channels"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        B, C, H, W, D = x.shape
        assert C == 1, 'UnetChannels only supports 1 channel input'
        
        # (B, C, H, W, D) -> (B, D, H, W, C)
        x = x.permute(0, 4, 2, 3, 1)

        # Remove C
        x = x.squeeze(4)
        
        return self.model(x)
