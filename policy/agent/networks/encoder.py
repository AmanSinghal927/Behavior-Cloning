import torch.nn as nn
import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 512

        # TODO: Define convnet with activation layers
        self.convnet = None

        # TODO: Define linear layer to map convnet output to representation dimension
        self.linear = None

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs - 0.5
        # TODO: Forward pass using obs as input
        h = None

        return h
