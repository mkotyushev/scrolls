import torch.nn as nn


class UnetAgg(nn.Module):
    def __init__(self, model, agg='max'):
        super().__init__()
        self.model = model
        self.agg = agg

    def forward(self, x):
        B, C, H, W, D = x.shape
        
        # (B, C, H, W, D) -> (B * D, C, H, W)
        x = x.view(B * D, C, H, W)
        x = self.model(x)  # (B * D, H, W)

        # (B * D, H, W) -> (B, D, H, W)
        x = x.view(B, D, H, W)
        # (B, D, H, W) -> (B, H, W)
        if self.agg == 'mean':
            x = x.mean(1)
        elif self.agg == 'max':
            x = x.max(1)[0]
        elif self.agg == 'min':
            x = x.min(1)[0]
        
        return x
