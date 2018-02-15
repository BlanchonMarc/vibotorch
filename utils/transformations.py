"""Custom Transformations

This module contains custom transformations to apply in torch Callable
Compose.

The module structure is the following:

- The ``to_label`` method convert PIL images to specific format of
  torch.Tensor

- The ``relabel`` method relabel along each channels / this method
  should be used only on annotation images.
"""
import torch
import numpy as np

class to_label:
    """Class to convert PIL images to specific format of torch.Tensor."""
    def __call__(self, _input : torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.array(_input)).long().unsqueeze(0)


class relabel:
    """Class to relabel along each channels a torch.Tensor"""
    def __init__(self, olabel : int, nlabel : int) -> None:
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, _input : torch.Tensor) -> torch.Tensor:
        assert isinstance(_input,
                          torch.LongTensor),
                          'tensor needs to be LongTensor'

        _input[_input == self.olabel] = self.nlabel
        return _input
