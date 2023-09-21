import torch
import torch.nn as nn

class CircularAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        '''- args:
                - predicted: predicted angles, shape (B, n_rays, H, W)
                - target: target angles, shape (B, n_rays, H, W)
           - returns:
                - ae_loss: absolute error loss with no reduction, shape (B, n_rays, H, W)
        '''
        angular_diff = torch.abs(predicted - target)
        ae_loss = torch.min(angular_diff, 1 - angular_diff)
        return ae_loss