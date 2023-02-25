import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .utils import _easy_mlp, TimeSeriesOM

class CopulaDecoder(nn.Module):
    """
    Attentional_copula
    """
    def __init__(self, input_dim, min_u, max_u):
        super().__init__()
    def loss(self, encoded, mask, pred_encoded):
        r"""
        """
        # encoded: (batch_size, seq_len, input_dim)
        # mask: (batch_size, seq_len)
        # pred_encoded: (batch_size, seq_len, input_dim)
        # return: (batch_size, seq_len)
        return
        
class AttentionalCopula(nn.Module):
    r"""
    
    """
    def __init__(self, input_dim, attn_heads, attn_dim, attn_layers, mlp_dim, mlp_layers, resolution,
                 dropout: float = .01):
        super().__init__()

        self.input_dim = input_dim
        self.attn_heads = attn_heads
        self.attn_dim = attn_dim
        self.attn_layers = attn_layers
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers
        self.resolution = resolution
        self.dropout = dropout

        self.dimension_shift = nn.Linear(self.input_dim, self.attn_heads*self.attn_dim)
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
        self.attention_dropouts = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.attn_layers)])
        self.attention_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attn_heads * self.attn_dim) for _ in range(self.attn_layers)]
        )
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.attn_heads * self.attn_dim, self.attn_heads * self.attn_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.attn_heads * self.attn_dim, self.attn_heads * self.attn_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(self.attn_layers)
            ]
        )
        self.feed_forward_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attn_heads * self.attn_dim) for _ in range(self.attn_layers)]
        )
    def loss(self, encoded, true_u, mask):
        r"""
        Args:
        -----
        encoded: [batch, series, time_steps, input_dim]
        true_u: [batch, series, time_steps]
        mask: [batch, time_steps]
        """
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_time_steps = encoded.shape[2]
        input_dim = encoded.shape[3]

        true_pred_u = true_u[:, :, mask.bool()]
        
        # [batch, series, time_steps, input_dim+1]
        merged_input = torch.cat([encoded, true_u[:, :, :, None]], dim=-1)

        # [[batch, attn_heads, series, time_steps, attn_dim]*attn_layers]
        keys = [
            torch.cat([rearrange(mlp(merged_input), "b s t d -> b 1 s t d") for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]
        values = [
            torch.cat([rearrange(mlp(merged_input), "b s t d -> b 1 s t d") for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attn_layers)
        ]
        product_mask = torch.ones(
            (num_batches, self.attn_heads, num_series, num_time_steps)
        )

        return 
    

    def sample_once(self, hist_encoded, true_u, pred_encoded):
        ### for sampling end
        r"""
        hist_encoded: [batch, series, time_steps, embedding_dim]
            Tensor containing embedding of history data points
        true_u: [batch, series, time_steps]
            0 if the data point to be forcasted
        pred_encoded: [batch, series, time_steps=1, embedding_dim]
        """
        hist_encoded = torch.clone(hist_encoded)
        true_u = torch.clone(true_u)
        pred_encoded = torch.clone(pred_encoded)
        
        num_batches = hist_encoded.shape[0]
        num_series = hist_encoded.shape[1]

        merged_input = torch.cat([hist_encoded, true_u[:, :, :, None]], dim=-1)
        keys_all = [
            torch.cat([mlp(merged_input)[:, :, :, None, :] for mlp in self.key_creators[layer]], axis=3)
            for layer in range(self.attn_layers)
        ]
        values_all = [
            torch.cat([mlp(merged_input)[:, :, :, None, :] for mlp in self.value_creators[layer]], axis=3)
            for layer in range(self.attn_layers)
        ]
        for i in range(num_series):
            att_value = self.dimension_shift(pred_encoded[:, i, :, :])
        
    def sample(self, encoded, true_u, mask, device=None):
        r"""
        encoded: [batch, series, time_steps, embedding_dim]
            Tensor containing embedding of history data points and data points to be forcasted
        true_u: [batch, series, time_steps]
            0 if the data point to be forcasted
        mask: [batch_size, time_steps]
            1 for history, 0 for prediction
        """
        time_horizon = TimeSeriesOM(mask)
        
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]

        # [batch, series, time_steps, input_dim + 1]
        merged_input = torch.cat([encoded, true_u[:, :, :, None]], dim=-1)

        # [[batch, series, time_steps, attn_heads, attn_dim]*attn_layers] for history and prediction time steps, concat along the attn_heads dimension
        keys_all = [
            torch.cat([mlp(merged_input)[:, :, :, None, :] for mlp in self.key_creators[layer]], axis=3)
            for layer in range(self.attn_layers)
        ]
        values_all = [
            torch.cat([mlp(merged_input)[:, :, :, None, :] for mlp in self.value_creators[layer]], axis=3)
            for layer in range(self.attn_layers)
        ]

        while time_horizon.has_missing_points():
            mid_point_map = time_horizon.next_time_steps_to_sample()
            mid_points_index = tuple(mid_point_map.keys())
            num_mid_points = len(mid_points_index)

            for i in range(num_series):
                current_pred_encoded = encoded[:, i, mid_points_index, :].reshape(num_batches, num_mid_points, self.input_dim)
                # [batch, num_mid_points, attn_heads*attn_dim]
                att_value = self.dimension_shift(current_pred_encoded)

                for layer in range(self.attn_layers):
                    # att_value: [batch, num_mid_points, attn_heads, attn_dim]
                    # iterate over series make it one-dimensional
                    att_value_heads = att_value.reshape(
                        num_batches, num_mid_points, self.attn_heads, self.attn_dim
                    )
                    # [batch, series, time_steps, attn_heads, attn_dim]
                    keys_hist = keys_all[layer]
                    # keys_hist = torch.cat((keys_hist[:, i, :, :, :], ), axis=2)
                    values_hist = values_all[layer]

                    # [batch, num_mid_points, series, 2-neighbor, attn_head, attn_dim]
                    neighbor_keys_hist = torch.empty((num_batches, num_mid_points, num_series, 2, self.attn_heads, self.attn_dim), device=device)
                    neighbor_values_hist = torch.empty((num_batches, num_mid_points, num_series, 2, self.attn_heads, self.attn_dim), device=device)
                    for mid, (left, right) in enumerate(mid_point_map.values()):
                        neighbor_keys_hist[:, mid, :, :, :, :] = torch.cat((
                                                            keys_hist[:, :, left:left+1, :, :],
                                                            torch.cat((keys_hist[:, :i, mid:mid+1, :, :], keys_hist[:, i:, right:right+1, :, :]), axis=1)
                                                            ), 
                                                            axis=2)
                        neighbor_values_hist[:, mid, :, :, :, :] = torch.cat((
                                                            values_hist[:, :, left:left+1, :, :],
                                                            torch.cat((values_hist[:, :i, mid:mid+1, :, :], values_hist[:, i:, right:right+1, :, :]), axis=1)
                                                            ), 
                                                            axis=2)
                    # neighbor_keys_hist = torch.cat([rearrange(keys_hist[:, :, neighbors, :, :], "batch series nbr head dim -> batch 1 head series nbr dim") 
                    #                                 for neighbors in mid_point_map.values()], axis=1)
                    # neighbor_values_hist = torch.cat([rearrange(values_hist[:, :, neighbors, :, :], "batch series nbr head dim -> batch 1 head series nbr dim") 
                    #                                 for neighbors in mid_point_map.values()], axis=1)
                    product_hist = torch.einsum("bmhd, bmsnhd -> bmhsn", att_value_heads, neighbor_keys_hist)
                    product_hist = self.attn_dim ** (-0.5) * product_hist

                    weights = F.softmax(product_hist, dim=-1)
                    att = torch.einsum("bmhsn, bmsnhd -> bmhd", weights, neighbor_values_hist)
                    att = rearrange(att, "batch mid_points head dim -> batch mid_points (head dim)")
                    att = self.attention_dropouts[layer](att)
                    att_value = att_value + att

                    # pre dropout
                    att_value = self.attention_layer_norms[layer](att_value)
                    att_feed_forward = self.feed_forwards[layer](att_value)
                    # residual connection
                    att_value = att_value + att_feed_forward
                    att_value = self.feed_forward_layer_norms[layer](att_value)

                # Get the output distribution parameters
                logits = self.dist_extractors(att_value).reshape(num_batches * num_mid_points, self.resolution)
                # Select a single variable in {0, 1, 2, ..., self.resolution-1} according to the probabilities from the softmax
                current_samples = torch.multinomial(input=torch.softmax(logits, dim=1), num_samples=1)
                # Each point in the same bucket is equiprobable, and we used a floor function in the training
                current_samples = current_samples + torch.rand_like(current_samples, device=device, dtype=torch.float32)
                # Normalize to a variable in the [0, 1) range
                current_samples /= self.resolution
                current_samples = current_samples.reshape(num_batches, num_mid_points)
                true_u[:, i, mid_points_index] = current_samples

                # Compute the key and value associated with the newly sampled variable, for the attention of the next ones.
                # [batch, num_mid_points, embedding_dim+1]
                key_value_input = torch.cat([current_pred_encoded, current_samples[:, :, None]], axis=-1)

                for layer in range(self.attn_layers):
                    keys_all[layer][:, i, mid_points_index, :, :] = torch.cat([mlp(key_value_input)[:, :, None, :] for mlp in self.key_creators[layer]], axis=2)
                    values_all[layer][:, i, mid_points_index, :, :] = torch.cat([mlp(key_value_input)[:, :, None, :] for mlp in self.value_creators[layer]], axis=2)
            print(f"Filled time steps {mid_points_index}")

        return true_u