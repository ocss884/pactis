from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from .utils import Sequential, Residual, _easy_mlp

class RotaryPositionEmbedding:
    # Specified in https://arxiv.org/abs/2104.09864
    # Modified from https://github.com/lucidrains/rotary-embedding-torch

    def __init__(self, frq_pos_enc: torch.Tensor, right_align: bool = False):
        # frq_pos_enc shape is (b, n, c).
        # frq_pos_enc is broadcast to (b, h, n, c).
        self.frq_pos_enc = rearrange(frq_pos_enc, "b n c -> b 1 n c")
        self.rotate_dim = frq_pos_enc.shape[-1]
        self.right_align = right_align

    def rotate(self, t):
        seq_len = t.shape[-2]
        if self.right_align:
            # q and k are right-aligned in Perceiver AR
            pos_enc = self.frq_pos_enc[..., -seq_len:, :]
        else:
            # q and k are left-aligned
            pos_enc = self.frq_pos_enc[..., :seq_len, :]

        t_rot, t_pass = t[..., : self.rotate_dim], t[..., self.rotate_dim :]
        t_rot = (t_rot * pos_enc.cos()) + (self._rotate_half(t_rot) * pos_enc.sin())

        return torch.cat((t_rot, t_pass), dim=-1)

    @staticmethod
    def _rotate_half(x):
        # Rearranges channel dimension [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        x = rearrange(x, "... (c r) -> ... c r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... c r -> ... (c r)")


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_dim: int,
        num_kv_input_dim: int,
        num_qk_dim: Optional[int] = None,
        num_v_dim: Optional[int] = None,
        num_output_dim: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        batch_first: bool = True,
    ):
        """Multi-head attention modified from https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.
        Args:
        :param num_heads: Number of attention heads.
        :param num_q_input_dim: Number of query input dim.
        :param num_kv_input_dim: Number of key/value input dim.
        :param num_qk_dim: Number of query and key dim. Default is number `num_q_input_dim`
        :param num_v_dim: Number of value dim. Default is `num_qk_dim`.
        :param num_output_dim: Number of output dim. Default is `num_q_input_dim`
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_dim is None:
            num_qk_dim = num_q_input_dim

        if num_v_dim is None:
            num_v_dim = num_qk_dim

        if num_output_dim is None:
            num_output_dim = num_q_input_dim

        if num_qk_dim % num_heads != 0:
            raise ValueError("n_qk_dim must be divisible by num_heads")

        if num_v_dim % num_heads != 0:
            raise ValueError("n_v_dim must be divisible by num_heads")

        num_qk_dim_per_head = num_qk_dim // num_heads

        self.dp_scale = num_qk_dim_per_head ** -0.5
        self.num_heads = num_heads
        self.causal_attention = causal_attention

        self.q_proj = nn.Linear(num_q_input_dim, num_qk_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_dim, num_qk_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_dim, num_v_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_dim, num_output_dim, bias=out_bias)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(
        self,
        query,
        key,
        value,
        pad_mask=None,
        rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
    ):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input dim (= `n_q_input_dim`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input dim (= `n_kv_input_dim`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output dim (= `num_output_dim`)
        """
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        attn = torch.einsum("b h i c, b h j c -> b h i j", q, k)
        attn_max_neg = -torch.finfo(attn.dtype).max
        # attn_max_neg = float("-inf")

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")
            attn.masked_fill_(pad_mask, attn_max_neg)

        # if self.causal_attention:
        #     i = q.shape[2]
        #     j = k.shape[2]

        #     # If q and k have different length, causal masking only works if they are right-aligned.
        #     causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)
        #     attn.masked_fill_(causal_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b h i j, b h j c -> b h i c", attn, v)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_dim: int,
        num_qk_dim: Optional[int] = None,
        num_v_dim: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(num_dim)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_dim=num_dim,
            num_kv_input_dim=num_dim,
            num_qk_dim=num_qk_dim,
            num_v_dim=num_v_dim,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            batch_first=batch_first,
        )

    def forward(self, x, pad_mask=None, rot_pos_emb=None):
        return self.attention(x, x, x, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb, rot_pos_emb_k=rot_pos_emb)


class SelfAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_dim: int,
        num_qk_dim: Optional[int] = None,
        num_v_dim: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        norm1 = nn.LayerNorm(num_dim)
        norm2 = nn.LayerNorm(num_dim)

        self_attn = SelfAttention(
            num_heads = num_heads,
            num_dim = num_dim,
            num_qk_dim = num_qk_dim,
            num_v_dim = num_v_dim,
            causal_attention = causal_attention,
            dropout = dropout,
            qkv_bias = qkv_bias,
            out_bias = out_bias,
            batch_first = batch_first,
        )
        mlp = _easy_mlp(input_dim = num_dim, 
                             hidden_dim = widening_factor * num_dim,
                             output_dim = num_dim,
                             num_layers = 1,
                             activation = nn.GELU,
                             )
        if norm_first:
            super().__init__(
                norm1,
                Residual(self_attn),
                norm2,
                Residual(mlp),
            )
        else:
            super().__init__(
                Residual(self_attn),
                norm1,
                Residual(mlp), 
                norm2
            )


class SelfAttentionBlock(Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_dim: int,
        num_qk_dim: Optional[int] = None,
        num_v_dim: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_dim=num_dim,
                num_qk_dim=num_qk_dim,
                num_v_dim=num_v_dim,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                batch_first=batch_first,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        ]

        super().__init__(*layers)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_dim: int,
        num_kv_input_dim: int,
        num_qk_dim: Optional[int] = None,
        num_v_dim: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        out_bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_dim=num_q_input_dim,
            num_kv_input_dim=num_kv_input_dim,
            num_qk_dim=num_qk_dim,
            num_v_dim=num_v_dim,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            batch_first=batch_first,
        )

    def forward(self, query, key, value, pad_mask=None, rot_pos_emb_q=None, rot_pos_emb_k=None):
        return self.attention(query, key, value, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb_q, rot_pos_emb_k=rot_pos_emb_k)


class CrossAttentionLayer(nn.Module):
    r"""
    Args:
        num_heads: number of attention heads
        num_q_input_dim: input dimension of query
        num_kv_input_dim: input dimension of key and value
        num_qk_dim: connected dimension of query and key
        num_v_dim: connected dimension of value
        dropout: dropout rate, 0.1 by default
        batch_first: whether the input is batch first, default is True
        norm_first: Pre-LayerNorm or Post-LayerNorm, default is Pre-LayerNorm
    """
    def __init__(self,
        num_heads: int,
        num_q_input_dim: int,
        num_kv_input_dim: int,
        num_qk_dim: Optional[int] = None,
        num_v_dim: Optional[int] = None,
        qkv_bias: bool = True,
        out_bias: bool = True,
        widening_factor: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.norm_first = norm_first

        self.cross_attn = CrossAttention(num_heads, num_q_input_dim, num_kv_input_dim, num_qk_dim, num_v_dim,
                                            qkv_bias=qkv_bias, out_bias=out_bias, dropout=dropout, batch_first=batch_first)
        # self.mlp = Sequential(
        #     nn.Linear(num_q_input_dim, widening_factor * num_q_input_dim, bias=mlp_bias),
        #     nn.GELU(),
        #     nn.Linear(widening_factor * num_q_input_dim, num_q_input_dim, bias=mlp_bias),
        # )
        self.mlp = _easy_mlp(input_dim = num_q_input_dim, 
                            hidden_dim = widening_factor * num_q_input_dim, 
                            output_dim = num_q_input_dim, 
                            num_layers = 1,
                            activation = nn.GELU
                            )

        if norm_first:
            # pre-LayerNorm
            self.q_norm = nn.LayerNorm(num_q_input_dim)
            self.kv_norm = nn.LayerNorm(num_kv_input_dim)
            self.attn_out_norm = nn.LayerNorm(num_q_input_dim)

            self._ca_layer = Sequential(
                                Residual(self.cross_attn),
                                Residual(Sequential(
                                            self.attn_out_norm,
                                            self.mlp
                                        )
                                ),
                            )
        else:
            # post-LayerNorm
            self.attn_out_norm = nn.LayerNorm(num_q_input_dim)
            self.mlp_out_norm = nn.LayerNorm(num_q_input_dim)

            self._ca_layer = Sequential(
                                Residual(self.cross_attn),
                                self.attn_out_norm,
                                Residual(self.mlp),
                                self.mlp_out_norm
                            )
            
    def forward(self, query, key, value, pad_mask=None):
        if self.norm_first:
            query, key, value = self.q_norm(query), self.kv_norm(key), self.kv_norm(value)
        
        return self._ca_layer(query, key, value, pad_mask)