import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from .utils import _easy_mlp, TimeSeriesOM
from .marginal import DSFMarginal
from typing import Sequence


class CopulaDecoder(nn.Module):
    """
    A decoder which forecast using a distribution built from a copula and marginal distributions.
    """

    def __init__(
        self,
        input_dim: int,
        min_u: float = 0.0,
        max_u: float = 1.0,
        skip_sampling_marginal: bool = False,
        attentional_copula=None,
        dsf_marginal=None,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            The dimension of the encoded representation (upstream data encoder).
        min_u: float, default to 0.0
        max_u: float, default to 1.0
            The values sampled from the copula will be scaled from [0, 1] to [min_u, max_u] before being sent to the marginal.
        skip_sampling_marginal: bool, default to False
            If set to True, then the output from the copula will not be transformed using the marginal during sampling.
            Does not impact the other transformations from observed values to the [0, 1] range.
        trivial_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a TrivialCopula.
            The options sent to the TrivialCopula is content of this dictionary.
        attentional_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a AttentionalCopula.
            The options sent to the AttentionalCopula is content of this dictionary.
        dsf_marginal: Dict[str, Any], default to None
            If set to a non-None value, uses a DSFMarginal.
            The options sent to the DSFMarginal is content of this dictionary.
        """
        super().__init__()

        assert dsf_marginal is not None, "Must select exactly one type of marginal"

        self.min_u = min_u
        self.max_u = max_u
        self.input_dim = input_dim
        
        self.skip_sampling_marginal = skip_sampling_marginal

        if attentional_copula is not None:
            self.copula = AttentionalCopula(
                input_dim=input_dim, **attentional_copula)

        # if dsf_marginal is not None:
        self.marginal = DSFMarginal(
            context_dim=input_dim, **dsf_marginal).to("cuda")

    def loss(self, encoded: torch.Tensor, true_value: torch.Tensor, mask: Sequence) -> torch.Tensor:
        """
        Compute the loss function of the decoder.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.

        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            Only the values where the mask is set to False will be considered in the loss function.

        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.

        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.
        """

        # Assume that the mask is constant inside the batch
        device = encoded.device
        mask = torch.as_tensor(mask, dtype=bool)

        hist_encoded = encoded[:, :, mask, :]
        pred_encoded = encoded[:, :, ~mask, :]
        hist_true_x = true_value[:, :, mask]
        pred_true_x = true_value[:, :, ~mask]
        # Transform to [0,1] using the marginals
        hist_true_u = self.marginal.forward_no_logdet(
            hist_encoded, hist_true_x)
        pred_true_u, marginal_logdet = self.marginal.forward_logdet(
            pred_encoded, pred_true_x)

        true_u = torch.empty_like(true_value, device=device)
        true_u[:, :, mask] = hist_true_u
        true_u[:, :, ~mask] = pred_true_u

        mask = mask.to(device)
        copula_loss = self.copula.loss(
            encoded=encoded,
            mask=mask,
            true_u=true_u,
            device=device
        )

        # Loss = negative log likelihood
        return copula_loss - marginal_logdet

    def sample(self, encoded, true_value, mask, num_samples=1, device=None) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.

        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            The values where the mask is set to True will be copied as-is in the output.

        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value is masked (available) for the encoder.
            The decoder only forecasts values for which the mask is set to False.

        num_samples: int (default: 1)
            How many samples to generate, must be >= 1.

        Returns:
        --------
        samples: torch.Tensor [samples, batch, series, time steps]
            Samples drawn from the forecasted distribution.
        """
        device =  encoded.device if device is None else device
        mask = torch.as_tensor(mask).bool().to(device)

        num_batch, num_series, _, num_input_dim = encoded.shape

        # Transform to [0,1] using the marginals
        true_u = self.marginal.forward_no_logdet(encoded, true_value)

        all_samples = self.copula.sample(
            encoded=encoded,
            true_u=true_u,
            mask=mask,
            num_samples=num_samples,
            device=device
        )
        all_samples = rearrange(all_samples, "sample batch series timestep -> batch series timestep sample")
        pred_encoded = encoded[..., ~mask, :]
        pred_samples = all_samples[..., ~mask, :]

        if not self.skip_sampling_marginal:
            # Transform away from [0,1] using the marginals
            pred_samples = self.min_u + \
                (self.max_u - self.min_u) * pred_samples

            pred_samples = self.marginal.inverse(
                pred_encoded.reshape(num_batch, -1, num_input_dim),
                pred_samples.reshape(num_batch, -1, num_samples),
            ).reshape(num_samples, num_batch, num_series, -1)

        # [samples, batch, series, time steps]
        samples = torch.empty([num_samples]+list(true_value.shape), device=device)

        samples[..., mask] = true_value[:, :, mask][None, ...]
        samples[..., ~mask] = pred_samples

        return samples


class AttentionalCopula(nn.Module):
    r"""
    Args:
        input_dim: int
            The dimension of the input to the decoder.
        attn_heads: int
            The number of attention heads to use.
        attn_dim: int
            The dimension of the attention heads.
        attn_layers: int
            The number of attention layers to use.
        mlp_dim: int
            The dimension of the MLPs used to extract the distribution parameters.
        mlp_layers: int
            The number of MLP layers to use.
        resolution: int
            The resolution of the multinominal distribution.

    """

    def __init__(self, input_dim, attn_heads, attn_dim, attn_layers, mlp_dim, mlp_layers, resolution,
                 dropout: float = .1):
        super().__init__()

        self.input_dim = input_dim
        self.attn_heads = attn_heads
        self.attn_dim = attn_dim
        self.attn_layers = attn_layers
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers
        self.resolution = resolution
        self.dropout = dropout

        self.dimension_shift = nn.Linear(
            self.input_dim, self.attn_heads*self.attn_dim)
        self.dist_extractors = _easy_mlp(
            input_dim=self.attn_heads * self.attn_dim,
            hidden_dim=self.mlp_dim,
            output_dim=self.resolution,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )

        self.key_creators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _easy_mlp(
                            input_dim=self.input_dim + 1,
                            hidden_dim=self.mlp_dim,
                            output_dim=self.attn_dim,
                            num_layers=self.mlp_layers,
                            activation=nn.ReLU,
                        )
                        for _ in range(self.attn_heads)
                    ]
                )
                for _ in range(self.attn_layers)
            ]
        )
        self.value_creators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _easy_mlp(
                            input_dim=self.input_dim + 1,
                            hidden_dim=self.mlp_dim,
                            output_dim=self.attn_dim,
                            num_layers=self.mlp_layers,
                            activation=nn.ReLU,
                        )
                        for _ in range(self.attn_heads)
                    ]
                )
                for _ in range(self.attn_layers)
            ]
        )
        self.attention_dropouts = nn.ModuleList(
            [nn.Dropout(self.dropout) for _ in range(self.attn_layers)])
        self.attention_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attn_heads * self.attn_dim)
             for _ in range(self.attn_layers)]
        )
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.attn_heads * self.attn_dim,
                              self.attn_heads * self.attn_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.attn_heads * self.attn_dim,
                              self.attn_heads * self.attn_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(self.attn_layers)
            ]
        )
        self.feed_forward_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attn_heads * self.attn_dim)
             for _ in range(self.attn_layers)]
        )

    def loss(self, encoded, true_u, mask, device=None):
        r"""
        Args:
        -----
        encoded: [batch, series, time_steps, input_dim]
        true_u: [batch, series, time_steps]
        mask: [time_steps]
        """

        num_batch, num_series, num_time_steps, _ = encoded.shape
        num_variable = num_series * num_time_steps

        time_horizon = TimeSeriesOM(num_series, mask)
        mid_points_map = {}
        while time_horizon.has_missing_points():
            current_mid_points_map = time_horizon.next_to_fill()
            mid_points_map = {**mid_points_map, **current_mid_points_map}
        num_pred_points = len(mid_points_map)

        encoded = encoded.reshape(num_batch, num_variable, self.input_dim)
        true_u = true_u.reshape(num_batch, num_variable, 1)
        merged_input = torch.cat([encoded, true_u], dim=-1)

        # [[batch, attn_heads, variable, attn_dim]*attn_layers] for history and prediction variable, concat along the variable dimension
        keys_all = [
            torch.cat([mlp(merged_input)[:, None, :, :]
                      for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]
        values_all = [
            torch.cat([mlp(merged_input)[:, None, :, :]
                      for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]

        pred_points = tuple(mid_points_map.keys())
        # [batch, num_pred_points, 2*num_series (neighbor), attn_dim]
        current_encoded = encoded[:, pred_points, :]

        neighbor_index = np.asarray(list(mid_points_map.values()))

        # inf mask
        attn_mask = torch.zeros((num_batch, num_pred_points, self.attn_heads, 2 * num_series), device=device)
        num_missing_ends = 0
        if 0 in mid_points_map:
            num_missing_ends += 1
        if (num_time_steps - 1) in mid_points_map:
            num_missing_ends += 1
        attn_mask[:, 0:num_missing_ends*num_series, num_series:] = float("inf")

        att_value = self.dimension_shift(current_encoded)

        for layer in range(self.attn_layers):
            # att_value: [batch, num_pred_points, attn_heads, attn_dim]
            att_value_heads = att_value.reshape(
                num_batch, num_pred_points, self.attn_heads, self.attn_dim
            )

            neighbor_keys = keys_all[layer][:, :, neighbor_index, :]
            neighbor_values = values_all[layer][:, :, neighbor_index, :]
            product_hist = torch.einsum("bphd, bhpnd -> bphn", att_value_heads, neighbor_keys)

            product_hist = product_hist - attn_mask
            product_hist = self.attn_dim ** (-0.5) * product_hist
            weights = F.softmax(product_hist, dim=-1)
            att = torch.einsum("bphn, bhpnd -> bphd", weights, neighbor_values)

            att = rearrange(att, "batch pred head dim -> batch pred (head dim)")
            att = self.attention_dropouts[layer](att)
            # residual
            att_value = att_value + att

            # pre dropout
            att_value = self.attention_layer_norms[layer](att_value)
            att_feed_forward = self.feed_forwards[layer](att_value)
            # residual
            att_value = att_value + att_feed_forward
            att_value = self.feed_forward_layer_norms[layer](att_value)

        logits = self.dist_extractors(att_value)
        target = torch.clip(torch.floor(true_u * self.resolution).long(), min=0, max=self.resolution - 1)
        logprob = np.log(self.resolution) + F.log_softmax(logits, dim=2)
        target = target[:, pred_points, :]
        logprob = torch.gather(logprob, 2, target)

        # batch
        return -logprob.sum((1, 2))

    def sample_once(self, encoded, true_u, mask, device=None):
        r"""
        encoded: [batch, series, time_steps, embedding_dim]
            Tensor containing embedding of history data points and data points to be forcasted
        true_u: [batch, series, time_steps]
            0 if the data point to be forcasted
        mask: [time_steps]
            1 for history, 0 for prediction
        """
        if device is None:
            device = encoded.device

        num_batch, num_series, num_time_steps, _ = encoded.shape
        num_variable = num_series * num_time_steps

        time_horizon = TimeSeriesOM(num_series, mask)
        encoded = encoded.reshape(num_batch, num_variable, self.input_dim).clone().to(device)
        true_u = true_u.reshape(num_batch, num_variable).clone().to(device)

        # [batch, variable, input_dim + 1]
        merged_input = torch.cat([encoded, true_u[:, :, None]], dim=-1)

        # [[batch, attn_heads, variable, attn_dim]*attn_layers] for history and prediction variable, concat along the attn_heads dimension
        keys_all = [
            torch.cat([mlp(merged_input)[:, None, :, :] for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]
        values_all = [
            torch.cat([mlp(merged_input)[:, None, :, :] for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]

        while time_horizon.has_missing_points():
            # index mask: Some points may not depends on all the values in the neighborhood, e.g. End points can only have one neighbor
            mid_point_map = time_horizon.next_to_fill()
            # [num_series, num_mid_points]
            pred_index_matrix = np.fromiter(mid_point_map.keys(), dtype=int).reshape(-1, num_series).T
            num_pred_points = len(pred_index_matrix[0])

            attn_mask = torch.zeros((num_batch, num_pred_points, self.attn_heads, 2 * num_series), device=device)
            if 0 in mid_point_map or (num_time_steps -1) in mid_point_map:
                attn_mask[:, :, :, num_series:] = float("inf")

            neighbor_index_matrix = np.asarray(list(mid_point_map.values())).reshape(-1, num_series, 2*num_series).transpose(1, 0, 2)
                
            for i in range(num_series):
                current_points_index = pred_index_matrix[i]
                current_pred_encoded = encoded[:, current_points_index, :].reshape(num_batch, num_pred_points, self.input_dim)
                neighbor_index = neighbor_index_matrix[i]
                # [batch, num_pred_points, attn_heads*attn_dim]
                att_value = self.dimension_shift(current_pred_encoded)

                for layer in range(self.attn_layers):
                    # att_value: [batch, num_pred_points, attn_heads, attn_dim]
                    att_value_heads = att_value.reshape(
                        num_batch, num_pred_points, self.attn_heads, self.attn_dim
                    )

                    # [batch, attn_head, num_pred_points, 2 * series, attn_dim]
                    neighbor_keys = keys_all[layer][:, :, neighbor_index, :]
                    neighbor_values = values_all[layer][:, :, neighbor_index, :]

                    product_hist = torch.einsum("bphd, bhpnd -> bphn", att_value_heads, neighbor_keys)
                    product_hist = self.attn_dim ** (-0.5) * product_hist
                    product_hist = product_hist - attn_mask

                    weights = F.softmax(product_hist, dim=-1)
                    att = torch.einsum("bphn, bhpnd -> bphd", weights, neighbor_values)
                    att = rearrange(att, "batch pred head dim -> batch pred (head dim)")
                    att = self.attention_dropouts[layer](att)
                    # residual
                    att_value = att_value + att

                    # pre dropout
                    att_value = self.attention_layer_norms[layer](att_value)
                    att_feed_forward = self.feed_forwards[layer](att_value)
                    # residual
                    att_value = att_value + att_feed_forward
                    att_value = self.feed_forward_layer_norms[layer](att_value)

                # Get the output distribution parameters
                logits = self.dist_extractors(att_value).reshape(num_batch * num_pred_points, self.resolution)
                # Select a single variable in {0, 1, 2, ..., self.resolution-1} according to the probabilities from the softmax
                current_samples = torch.multinomial(input=torch.softmax(logits, dim=1), num_samples=1)
                # Each point in the same bucket is equiprobable, and we used a floor function in the training
                current_samples = current_samples + \
                    torch.rand_like(current_samples, device=device, dtype=torch.float32)
                # Normalize to a variable in the [0, 1) range
                current_samples /= self.resolution
                current_samples = current_samples.reshape(num_batch, num_pred_points)
                true_u[:, current_points_index] = current_samples

                # Compute the key and value associated with the newly sampled variable, for the attention of the next ones.
                # [batch, num_pred_points, embedding_dim+1]
                key_value_input = torch.cat([current_pred_encoded, current_samples[:, :, None]], axis=-1)

                # update attention keys and values matrix
                for layer in range(self.attn_layers):
                    keys_all[layer][:, :, current_points_index, :] = torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.key_creators[layer]], axis=1)
                    values_all[layer][:, :, current_points_index, :] = torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.value_creators[layer]], axis=1)

        return true_u.reshape(num_batch, num_series, num_time_steps)

    def sample(self, encoded, true_u, mask,  num_samples=1, device=None):
        r"""
        encoded: [batch, series, time_steps, embedding_dim]
            Tensor containing embedding of history data points and data points to be forcasted
        true_u: [batch, series, time_steps]
            0 if the data point to be forcasted
        mask: [time_steps]
            1 for history, 0 for prediction
        num_samples: int
            number of samples to draw
        
        """
        device = encoded.device if device is None else device

        num_batch, num_series, num_time_steps, _ = encoded.shape
        num_variable = num_series * num_time_steps

        time_horizon = TimeSeriesOM(num_series, mask)
        encoded = encoded.reshape(num_batch, num_variable, self.input_dim).clone().to(device)
        true_u = true_u.reshape(num_batch, num_variable).clone().to(device)

        # [batch, variable, input_dim + 1]
        merged_input = torch.cat([encoded, true_u[:, :, None]], dim=-1)

        # ==================================================================================================
        # [[samples, batch, attn_heads, variable, attn_dim]*attn_layers] 
        # for history and prediction variable, concat along the attn_heads dimension
        keys_all = [
            torch.cat([mlp(merged_input)[:, None, :, :] for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]
        values_all = [
            torch.cat([mlp(merged_input)[:, None, :, :] for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]
        for layer in range(self.attn_layers):
            keys_all[layer] = repeat(keys_all[layer], "... -> num_samples ...", num_samples=num_samples).clone()
            values_all[layer] = repeat(values_all[layer], "... -> num_samples ...", num_samples=num_samples).clone()

        # ==================================================================================================
        # encoded = [samples, batch, variable, input_dim]
        # true_u = [samples, batch, variable]
        encoded = repeat(encoded, "... -> num_samples ...", num_samples=num_samples).clone()
        true_u = repeat(true_u, "... -> num_samples ...", num_samples=num_samples).clone()

        while time_horizon.has_missing_points():
            # index mask: Some points may not depends on all the values in the neighborhood, e.g. End points can only have one neighbor
            mid_point_map = time_horizon.next_to_fill()
            # [num_series, num_mid_points]
            pred_index_matrix = np.fromiter(mid_point_map.keys(), dtype=int).reshape(-1, num_series).T
            num_pred_points = len(pred_index_matrix[0])

            attn_mask = torch.zeros((num_samples, num_batch, num_pred_points, self.attn_heads, 2 * num_series), device=device)
            if 0 in mid_point_map or (num_time_steps -1) in mid_point_map:
                attn_mask[..., num_series:] = float("inf")
            # [num_series, num_mid_points, 2*num_series]
            neighbor_index_matrix = np.asarray(list(mid_point_map.values())).reshape(-1, num_series, 2*num_series).transpose(1, 0, 2)
                
            for i in range(num_series):
                current_points_index = pred_index_matrix[i]
                current_pred_encoded = encoded[..., current_points_index, :].reshape(num_samples, num_batch, num_pred_points, self.input_dim)
                neighbor_index = neighbor_index_matrix[i]
                # [samples, batch, num_pred_points, attn_heads*attn_dim]
                att_value = self.dimension_shift(current_pred_encoded)

                for layer in range(self.attn_layers):
                    # att_value: [batch, num_pred_points, attn_heads, attn_dim]
                    att_value_heads = att_value.reshape(
                        num_samples, num_batch, num_pred_points, self.attn_heads, self.attn_dim
                    )

                    # [samples, batch, attn_head, num_pred_points, 2 * series, attn_dim]
                    neighbor_keys = keys_all[layer][..., neighbor_index, :]
                    neighbor_values = values_all[layer][..., neighbor_index, :]

                    product_hist = torch.einsum("sbphd, sbhpnd -> sbphn", att_value_heads, neighbor_keys)
                    product_hist = self.attn_dim ** (-0.5) * product_hist
                    product_hist = product_hist - attn_mask

                    weights = F.softmax(product_hist, dim=-1)
                    att = torch.einsum("sbphn, sbhpnd -> sbphd", weights, neighbor_values)
                    att = rearrange(att, "sample batch pred head dim -> sample batch pred (head dim)")
                    att = self.attention_dropouts[layer](att)
                    # residual
                    att_value = att_value + att

                    # pre dropout
                    att_value = self.attention_layer_norms[layer](att_value)
                    att_feed_forward = self.feed_forwards[layer](att_value)
                    # residual
                    att_value = att_value + att_feed_forward
                    att_value = self.feed_forward_layer_norms[layer](att_value)

                # Get the output distribution parameters
                logits = self.dist_extractors(att_value).reshape(num_samples*num_batch * num_pred_points, self.resolution)
                # Select a single variable in {0, 1, 2, ..., self.resolution-1} according to the probabilities from the softmax
                current_samples = torch.multinomial(input=torch.softmax(logits, dim=1), num_samples=1)
                # Each point in the same bucket is equiprobable, and we used a floor function in the training
                current_samples = current_samples + \
                    torch.rand_like(current_samples, device=device, dtype=torch.float32)
                # Normalize to a variable in the [0, 1) range
                current_samples /= self.resolution
                current_samples = current_samples.reshape(num_samples, num_batch, num_pred_points)
                true_u[:, :, current_points_index] = current_samples
                # Compute the key and value associated with the newly sampled variable, for the attention of the next ones.
                # [samples, batch, num_pred_points, embedding_dim+1]
                key_value_input = torch.cat([current_pred_encoded, current_samples[..., None]], axis=-1)

                # update attention keys and values matrix
                for layer in range(self.attn_layers):
                    keys_all[layer][..., current_points_index, :] = torch.cat([mlp(key_value_input)[:, :, None, :, :] for mlp in self.key_creators[layer]], axis=2)
                    values_all[layer][..., current_points_index, :] = torch.cat([mlp(key_value_input)[:, :, None, :, :] for mlp in self.value_creators[layer]], axis=2)

        return true_u.reshape(num_samples, num_batch, num_series, num_time_steps)