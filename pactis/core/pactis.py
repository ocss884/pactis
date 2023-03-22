import torch
from torch import nn
import numpy as np
from einops import rearrange

class PositionalEncoding(nn.Module):
    """
    A class implementing the positional encoding for Transformers described in Vaswani et al. (2017).
    Somewhat generalized to allow unaligned or unordered time steps, as long as the time steps are integers.

    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Parameters:
        -----------
        embedding_dim: int
            The dimension of the input and output embeddings for this encoding.
        dropout: float, default to 0.1
            Dropout parameter for this encoding.
        max_length: int, default to 5000
            The maximum time steps difference which will have to be handled by this encoding.
        """
        super().__init__()

        assert embedding_dim % 2 == 0, "PositionEncoding needs an even embedding dimension"

        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_length, embedding_dim)
        possible_pos = torch.arange(0, max_length, dtype=torch.float)[:, None]
        factor = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-np.log(10000.0) / embedding_dim))

        # Alternate between using sine and cosine
        pos_encoding[:, 0::2] = torch.sin(possible_pos * factor)
        pos_encoding[:, 1::2] = torch.cos(possible_pos * factor)

        # Register as a buffer, to automatically be sent to another device if the model is sent there
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, input_encoded: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        input_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            An embedding which will be modified by the position encoding.
        timesteps: torch.IntTensor [batch, series, time steps] or [batch, 1, time steps]
            The time step for each entry in the input.

        Returns:
        --------
        output_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            The modified embedding.
        """
        # Use the time difference between the first time step of each batch and the other time steps.
        # min returns two outputs, we only keep the first.
        min_t = timesteps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        delta_t = timesteps - min_t

        output_encoded = input_encoded + self.pos_encoding[delta_t]
        return self.dropout(output_encoded)


class NormalizationIdentity:
    """
    Trivial normalization helper. Do nothing to its data.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        pass

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        return norm_value


class NormalizationStandardization:
    """
    Normalization helper for the standardization.

    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.

    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        std, mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.std = std.clamp(min=1e-8)
        self.mean = mean

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        value = (value - self.mean) / self.std
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        norm_value = (norm_value * self.std[:, :, :, None]) + self.mean[:, :, :, None]
        return norm_value


class PACTiS(nn.Module):
    """
    """

    def __init__(self, 
                 num_series: int,
                 series_embedding_dim: int,
                 num_embedding_layers: int,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 data_normalization: str = "standardization",
                 positional_encoding = None,
                 ):
        assert encoder.embedding_dim == decoder.input_dim, f"PACTiS: encoder input dim has {encoder.input_adapter.num_input_dim} input dimensions, but decoder has {decoder.input_dim} input dimensions."

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.data_normalization = {
            "": NormalizationIdentity,
            "none": NormalizationIdentity,
            "standardization": NormalizationStandardization,
        }[data_normalization]
        self.series_encoder = nn.Embedding(num_series, series_embedding_dim)

        elayers = nn.ModuleList([])
        for i in range(num_embedding_layers):
            if i == 0:
                elayers.append(
                    nn.Linear(series_embedding_dim + 2, encoder.embedding_dim)
                )  # +1 for the value, +1 for the mask, and the per series embedding
            else:
                elayers.append(nn.Linear(encoder.embedding_dim, encoder.embedding_dim))
            elayers.append(nn.ReLU())
        self.input_encoder = nn.Sequential(*elayers)
        if positional_encoding is not None:
            self.time_encoding = PositionalEncoding(encoder.embedding_dim, **positional_encoding)
            
    def loss(self, value, mask: torch.Tensor):
        num_batches, num_series, num_time_steps = value.shape

        hist_value = value[:, :, mask]
        pred_value = value[:, :, ~mask]
        device = value.device
    
        series_emb = self.series_encoder(torch.arange(num_series, device=device))
        series_emb = series_emb[None, :, :].expand(num_batches, -1, -1)

        normalizer = self.data_normalization(hist_value)
        hist_value = normalizer.normalize(hist_value)
        pred_value = normalizer.normalize(pred_value)
        value[:, :, mask] = hist_value
        value[:, :, ~mask] = pred_value

        encoded = torch.cat(
        [
            rearrange(value, "b s t -> b s t 1"),
            series_emb[:, :, None, :].expand(-1, -1, value.shape[2], -1),
            mask.float().repeat(num_batches, num_series, 1)[:, :, :, None],
        ],
        dim=3)

        encoded = self.input_encoder(encoded)
        # if self.input_encoding_normalization:
        encoded = encoded * self.encoder.embedding_dim**0.5

        if self.time_encoding:
            encoded = self.time_encoding(encoded, 
                                         torch.arange(num_time_steps, device=device).repeat(num_batches, num_series, 1))
        encoded = rearrange(encoded, "b s t d -> b (s t) d")

        encoded = self.encoder(encoded)
        encoded = rearrange(encoded, "b (s t) d -> b s t d", s=num_series)
        loss = self.decoder.loss(encoded, value, mask)

        return loss.mean()/num_series
    
    def sample(self, value, mask):
        num_batches = value.shape[0]
        num_series = value.shape[1]
        hist_value = value[:, :, mask]
        pred_value = value[:, :, ~mask]
        device = value.device
        series_emb = self.series_encoder(torch.arange(num_series, device=device))
        series_emb = series_emb[None, :, :].expand(num_batches, -1, -1)

        normalizer = self.data_normalization(hist_value)
        hist_value = normalizer.normalize(hist_value)
        pred_value = normalizer.normalize(pred_value)
        value[:, :, mask] = hist_value
        value[:, :, ~mask] = pred_value

        encoded = torch.cat(
        [
            rearrange(value, "b s t -> b s t 1"),
            series_emb[:, :, None, :].expand(-1, -1, value.shape[2], -1),
            mask.float().repeat(num_batches, num_series, 1)[:, :, :, None],
        ],
        dim=3)

        encoded = rearrange(encoded, "b s t d -> b (s t) d")
        encoded = self.input_encoder(encoded)
        encoded = self.encoder(encoded)
        encoded = rearrange(encoded, "b (s t) d -> b s t d", s=num_series)
        samples = self.decoder.sample(encoded, value, mask)

        return samples
