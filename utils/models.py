import torch
from torch import nn
from torchsummary import summary


def show_model_summary(model: nn.Module, device: torch.device, input_size: tuple):
    summary(model.to(device), input_size)
