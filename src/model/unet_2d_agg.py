import torch.nn as nn


class Unet2dAgg(nn.Module):
    """Wrapper to predict for each Z slice and aggregate the results."""
    def __init__(self, model, full_size, size, step=1):
        super().__init__()
        self.model = model
        self.full_size = full_size
        self.size = size
        self.step = step
        # CNN to aggregate the mulitple "channels"
        # into a single channel keeping the spatial dimensions
        assert (full_size - size) % step == 0, 'Invalid size and step'
        self.n_preds = (full_size - size) // step + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_preds, 1, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W, D = x.shape
        assert C == 1, 'Only single channel images supported'
        assert D == self.full_size, f'Full size must match, got {D} != {self.full_size}'

        # (B, C, H, W, D) -> (B, H, W, D)
        x = x.squeeze(1)
        
        # (B, H, W, D) -> (B, H, W, self.n_preds, self.size)
        x = x.unfold(3, self.size, self.step)
        # (B, H, W, self.n_preds, self.size) -> 
        # (B, self.n_preds, self.size, H, W) ->
        # (B * self.n_preds, self.size, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(-1, self.size, H, W)
        x = self.model(x)  # (B * self.n_preds, H, W)

        # (B * self.n_preds, H, W) -> (B, self.n_preds, H, W)
        x = x.view(B, self.n_preds, H, W)
        # (B, self.n_preds, H, W) -> (B, H, W)
        x = self.cnn(x).squeeze(1)
        
        return x
