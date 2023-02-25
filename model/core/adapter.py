import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional


class InputAdapter(nn.Module):
    def __init__(self, num_input_dim: int, *args, **kwargs):
        """Transforms and position-encodes task-specific input to generic encoder input.

        :param num_input_dim: Number of dim of the generic encoder input produced by this adapter.
        """
        super().__init__()
        self._num_input_dim = num_input_dim

    @property
    def num_input_dim(self):
        return self._num_input_dim
    
    def forward(self, x):
        return x


class RotarySupport(InputAdapter):
    def __init__(self, rotated_dim_per_head: int, *args, **kwargs):
        """An input adapter mixin that additionally generates a frequency position encoding for input sequence
        `x`."""
        super().__init__(*args, **kwargs)
        self.frq_pos_encoding = FrequencyPositionEncoding(dim=rotated_dim_per_head)

    def forward(self, x, abs_pos=None):
        if abs_pos is None:
            abs_pos = positions(*x.shape, device=x.device)
        return super().forward(x, abs_pos), self.frq_pos_encoding(abs_pos)


class LatentQuery(nn.Module):
    r"""Provider of learnable cross-attention query input.

    This is the latent array in Perceiver IO encoders and the output query array in most Perceiver IO decoders.
    """

    def __init__(self, num_latents, num_latent_dim, init_scale = 0.02):
        super().__init__()
        self._query = nn.Parameter(torch.empty(num_latents, num_latent_dim))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale):
        with torch.no_grad():
            self._query.normal_(0.0, init_scale)

    @property
    def num_query_dim(self):
        return self._query.shape[-1]

    def forward(self, x=None):
        return rearrange(self._query, "... -> 1 ...")