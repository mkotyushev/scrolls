import torch.nn as nn


class Unet2dAgg(nn.Module):
    """Wrapper to predict for each Z slice and aggregate the results."""
    def __init__(self, model, agg='max'):
        super().__init__()

        assert agg in ['mean', 'max', 'min']
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
