import torch

class ToLabel:

    def __call__(self, _input : torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.array(_input)).long().unsqueeze(0)


class Relabel:

    def __init__(self, olabel : int, nlabel : int) -> None:
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, _input : torch.Tensor) -> torch.Tensor:
        assert isinstance(_input,
                          torch.LongTensor),
                          'tensor needs to be LongTensor'

        _input[_input == self.olabel] = self.nlabel
        return _input
