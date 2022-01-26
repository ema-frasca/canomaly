import torch
from torch.functional import F


def reconstruction_error(input_image: torch.Tensor, reconstruction: torch.Tensor):
    """
    Compute the reconstruction error on a batch of images
    """
    return F.mse_loss(input_image, reconstruction, reduction='none').sum((-3, -2, -1))
