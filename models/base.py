from torch import nn, tensor
from abc import abstractmethod


class BaseAE(nn.Module):
    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: tensor) -> tensor:
        raise NotImplementedError

    def decode(self, input: tensor) -> tensor:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> tensor:
        raise NotImplementedError

    def generate(self, x: tensor, **kwargs) -> tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: tensor) -> tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: tensor, **kwargs) -> tensor:
        pass


