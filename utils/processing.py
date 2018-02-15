"""Data Processing tool functions

This module contains methods process data / class outputs.

The module structure is the following:

- The ``tensor_ConcatFromDict`` function to concatenate N tensors
  contained into a dictionnary with str as key arguments following
  the mentionned dimension as argument of the function

  Example:
  Dict = {'1' : torch.Tensor(1x3x10x10) , '2' : torch.Tensor(1x3x10x10)}

  tensor_ConcatFromDict(ds = Dict, inds = ['1','2'], dim = 0)
    |_ returning : torch.Tensor(2x3x10x10)

- The ``tensor_ConcatFromList`` function to concatenate N tensors
  contained into a List following the mentionned dimension as argument
   of the function

  Examples:
  List = [torch.Tensor(1x3x10x10) , torch.Tensor(1x3x10x10)]

  tensor_ConcatFromList(ds = List, dim = 0)
    |_ returning : torch.Tensor(2x3x10x10)

  tensor_ConcatFromList(ds = List, dim = 1)
    |_ returning : torch.Tensor(1x6x10x10)

  . . .

- The ``transform_tensor`` function to transform a tensor using
  the callable method Compose of PyTorch
  see : http://pytorch.org/docs/master/torchvision/transforms.html
  
"""

import torch
from typing import Dict, List, Callable
from PIL import Image
import torchvision

def tensor_ConcatFromDict(ds : Dict[str, torch.Tensor],
                          inds : List[str],
                          dim : int = 0) -> torch.Tensor:
    """Function performing torch.Tensor Concatenation

    Parameters
    ---------
    ds : Dict[str, torch.Tensor]
        The Collection of inputs.

    inds : List[str]
        The Selected keys which data will be concatenated.

    dim: int
        The dimension of concatenation.

    Returns
    -------
    _storage : torch.Tensor
        The resulting Tensor concatenated
    """
    # the storer
    _storage = ds[inds[0]]
    # the update / concatenation
    for indx in range(1,len(inds)):
        torch.cat((_storage, ds[inds[indx]]), dim)

    return _storage


def tensor_ConcatFromList(ds : List[torch.Tensor],
                          dim : int = 0) -> torch.Tensor:
    """Function performing torch.Tensor Concatenation

    Parameters
    ---------
    ds : List[torch.Tensor]
        The Collection of inputs.

    dim: int
        The dimension of concatenation.

    Returns
    -------
    _storage : torch.Tensor
        The resulting Tensor concatenated
    """
    # the storer
    _storage = ds[0]
    # the update / concatenation
    for indx in range(1,len(ds)):
        torch.cat((_storage, ds[indx]), dim)

    return _storage

def transform_tensor(inputs : torch.Tensor,
                     transformation : Callable) -> torch.Tensor:
    """Function performing torch.Tensor transformation

    Parameters
    ---------
    inputs : torch.Tensor
        The input Tensor

    transformation: Callable
        Callable collection made from PyTorch Compose function

    Example:
    in = torch.Tensor(2x3x10x10)
    trans = transforms.Compose([
                        transforms.CenterCrop(10),
                        transforms.ToTensor(),
                        ])

    Returns
    -------
    _out : torch.Tensor
        The resulting Tensor transformed
    """
    endx = list(inputs.size())[0]

    _storage = [torchvision.transforms.ToPILImage[inputs[indx]]
                for indx in range(endx)]

    _out = [transformation(_storage[indy])
            for indy in range(len(_storage))]

    return _out
