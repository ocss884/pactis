import torch
from torch import nn
from typing import Optional
from .modules import CrossAttentionLayer, SelfAttentionBlock
from .adapter import LatentQuery, InputAdapter
from .utils import Sequential
from .config import Config, PerceiverEncoderConfig, PerceiverDecoderConfig

class Base:
    def __init__(self, device = None, dtype=None):
        self.kwargs = {device: device, dtype: dtype}

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        return cls(**config.dict, **kwargs)

class PerceiverEncoder(nn.Module, Base):
    def __init__(
        self,
        input_adapter: InputAdapter,

        num_latents: int,
        num_latent_dim: int,
        init_scale: float = 0.02,
        
        num_cross_attn_heads: int = 4,
        num_cross_attn_qk_dim: Optional[int] = None,
        num_cross_attn_v_dim: Optional[int] = None,
        first_cross_attn_layer_shared: bool = False,
        cross_attn_widening_factor: int = 1,
        cross_attn_norm_first: bool = True,

        num_self_attn_heads: int = 4,
        num_self_attn_qk_dim: Optional[int] = None,
        num_self_attn_v_dim: Optional[int] = None,
        num_self_attn_layers_per_block: int = 6,
        first_self_attn_block_shared: bool = True,
        self_attn_widening_factor: int = 1,
        self_attn_norm_first: bool = True,
        dropout: float = 0.0,
        
        num_cross_attn_layers: int = 1,
        num_self_attn_blocks: int = 1,

        batch_first: bool = True,
    ):
        super().__init__()

        self.latent_provider = LatentQuery(num_latents, num_latent_dim, init_scale=init_scale)
        self.input_adapter = input_adapter

        assert num_cross_attn_layers > 0, "num_cross_attention_layers must be > 0"

        if num_self_attn_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attn_layers > num_self_attn_blocks:
            raise ValueError("num_cross_attn_layers must be <= num_self_attn_blocks")

        self.num_cross_attn_layers = num_cross_attn_layers
        self.num_self_attn_blocks = num_self_attn_blocks

        self.first_cross_attn_layer_shared = first_cross_attn_layer_shared
        self.first_self_attn_block_shared = first_self_attn_block_shared

        def cross_attn():
            return CrossAttentionLayer(
                num_heads=num_cross_attn_heads,
                num_q_input_dim=num_latent_dim,
                num_kv_input_dim=input_adapter.num_input_dim,
                num_qk_dim=num_cross_attn_qk_dim,
                num_v_dim=num_cross_attn_v_dim,
                widening_factor=cross_attn_widening_factor,
                dropout=dropout,
                norm_first=cross_attn_norm_first,
                batch_first=batch_first,
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attn_layers_per_block,
                num_heads=num_self_attn_heads,
                num_dim=num_latent_dim,
                num_qk_dim=num_self_attn_qk_dim,
                num_v_dim=num_self_attn_v_dim,
                widening_factor=self_attn_widening_factor,
                dropout=dropout,
                batch_first=batch_first,
                norm_first=self_attn_norm_first
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

    # >>> INIT BEHAVIOR IS NOT IMPLEMENTED YET
    #     self._init_parameters(init_scale)
    # def _init_parameters(self, init_scale: float):
    #     with torch.no_grad():
    #         init_parameters(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attn_layers > 1 and not self.first_cross_attn_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attn_blocks > 1 and not self.first_self_attn_block_shared

    def forward(self, x, pad_mask=None, return_adapted_input=False):
        x_adapted = self.input_adapter(x)
        x_latent = self.latent_provider()

        x_latent = self.cross_attn_1(x_latent, x_adapted, x_adapted, pad_mask=pad_mask)
        x_latent = self.self_attn_1(x_latent)

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attn_blocks):
            if i < self.num_cross_attn_layers:
                x_latent = cross_attn_n(x_latent, x_adapted, pad_mask=pad_mask)
            x_latent = self_attn_n(x_latent)

        if return_adapted_input:
            return x_latent, x_adapted
        else:
            return x_latent

class PerceiverDecoder(nn.Module, Base):
    def __init__(self, 
                num_latents: int,
                num_latent_dim: int,
                num_heads: int,
                num_q_input_dim: int,
                num_kv_input_dim: int,
                num_qk_dim: int,
                num_v_dim: int,
                qkv_bias: bool = True,
                out_bias: bool = True,
                widening_factor: int = 1,
                dropout: float = 0.0,
                norm_first: bool = True,
                batch_first: bool = True,
                init_scale: float = 0.02
    ):
        super().__init__()
        #[batch, seies*time steps, latent_dim]
        self.latent_provider = LatentQuery(num_latents, num_latent_dim)
        self.init_scale = init_scale
        # self._init_parameters(0.02)

        self.cross_attn = CrossAttentionLayer(num_heads=num_heads,
                                              num_q_input_dim=num_q_input_dim,
                                              num_kv_input_dim=num_kv_input_dim,
                                              num_qk_dim=num_qk_dim,
                                              num_v_dim=num_v_dim,
                                              widening_factor=widening_factor,
                                              qkv_bias=qkv_bias,
                                              out_bias=out_bias,
                                              dropout=dropout,
                                              norm_first=norm_first,
                                              batch_first=batch_first
                                              )
    @property
    def latent_shape(self):
        return (self.latent_provider.num_latents, self.latent_provider.num_latent_dim)
    
    # def _init_parameters(self, init_scale: float):
    #     with torch.no_grad():
    #         self.latent_provider.normal_(0.0, self.init_scale)

    # @classmethod
    # def from_config(cls, config: PerceiverDecoderConfig, device = None, dtype = None):
    #     return cls(**config.dict, device=device, dtype=dtype)

    def forward(self, encoder_output):
        #[batch, series*time steps, latent_dim] reshape done in next part
        query = self.latent_provider()
        out = self.cross_attn(query, encoder_output, encoder_output)
        return out
    

class PerceiverIO(Sequential, Base):
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)
        self.embedding_dim = encoder.input_adapter.num_input_dim

    @classmethod
    def from_config(cls, encoder_config: PerceiverEncoderConfig, decoder_config: PerceiverDecoder, **kwargs):
        encoder = PerceiverEncoder.from_config(encoder_config, **kwargs)
        decoder = PerceiverDecoder.from_config(decoder_config, **kwargs)
        return cls(encoder, decoder)
