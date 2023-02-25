import torch
from typing import Optional, Sequence
from dataclasses import dataclass, asdict, field, KW_ONLY
from .adapter import InputAdapter

@dataclass
class Config:
    @property
    def dict(self):
        return asdict(self)
    @classmethod
    def from_config(cls):
        raise NotImplementedError

@dataclass
class GeneralArgs:
    _: KW_ONLY
    dropout: float = 0.0
    batch_first: bool = True
    # norm_first: bool = True

@dataclass
class LatentQueryConfig(Config):
    num_latents: int
    num_latent_dim: int
    init_scale: float = 0.02

@dataclass
class CrossAttentionLayerConfig(Config, GeneralArgs):
    num_heads: int
    num_q_input_dim: int
    num_kv_input_dim: int
    _: KW_ONLY
    num_qk_dim: Optional[int] = None
    num_v_dim: Optional[int] = None
    widening_factor: int = 1


@dataclass
class SelfAttentionLayerConfig(Config, GeneralArgs):
    num_heads: int
    num_dim: int
    num_qk_dim: Optional[int] = None
    num_v_dim: Optional[int] = None
    widening_factor: int = 1

@dataclass
class SelfAttentionBlockConfig(SelfAttentionLayerConfig):
    num_layers: int = 1

@dataclass
class PerceiverEncoderConfig(Config):
    input_adapter: InputAdapter
    num_latents: int
    num_latent_dim: int
    init_scale: float = 0.02

    num_cross_attn_heads: int = 4
    num_cross_attn_qk_dim: Optional[int] = None
    num_cross_attn_v_dim: Optional[int] = None
    first_cross_attn_layer_shared: bool = False
    cross_attn_widening_factor: int = 1
    cross_attn_norm_first: bool = True

    num_self_attn_heads: int = 4
    num_self_attn_qk_dim: Optional[int] = None
    num_self_attn_v_dim: Optional[int] = None
    num_self_attn_layers_per_block: int = 6
    self_attn_widening_factor: int = 1
    
    num_cross_attn_layers: int = 1
    num_self_attn_blocks: int = 1

    batch_first: bool = True
    dropout: float = 0.0

    @property
    def latent_query_config(self) -> LatentQueryConfig:
        return LatentQueryConfig(
            num_latents = self.num_latents,
            num_latent_dim = self.num_latent_dim,
            init_scale = self.init_scale
        )

    @property
    def cross_attn_config(self) -> CrossAttentionLayerConfig:
        return CrossAttentionLayerConfig(
            num_heads = self.num_cross_attn_heads,
            num_q_input_dim = self.num_latent_dim,
            num_kv_input_dim = self.input_adapter.num_latent_dim,
            num_qk_dim = self.num_cross_attn_qk_dim,
            num_v_dim = self.num_cross_attn_v_dim,
            widening_factor = self.cross_attn_widening_factor,
            dropout = self.dropout,
            batch_first = self.batch_first,
        )

    @property
    def self_attn_config(self) -> SelfAttentionBlockConfig:
        return SelfAttentionBlockConfig(
            num_layers=self.num_self_attn_layers_per_block,
            num_heads = self.num_self_attn_heads,
            num_dim = self.num_latent_dim,
            num_qk_dim = self.num_self_attn_qk_dim,
            num_v_dim = self.num_self_attn_v_dim,
            widening_factor=self.self_attn_widening_factor,
            dropout = self.dropout,
            batch_first = self.batch_first,
            # norm_first=self.self_attn_norm_first,
        )
    
    
    @classmethod
    def create_from_config(cls, input_adapter: InputAdapter, 
                           latent_query_config: LatentQueryConfig, 
                           cross_attn_config: CrossAttentionLayerConfig, 
                           self_attn_config: SelfAttentionBlockConfig,
                           num_cross_attn_layers: int = 1,
                           num_self_attn_blocks: int = 1,
                           ):
        return cls(
            input_adapter = input_adapter,

            **latent_query_config.dict,

            num_cross_attn_heads = cross_attn_config.num_heads,
            num_cross_attn_qk_dim = cross_attn_config.num_qk_dim,
            num_cross_attn_v_dim = cross_attn_config.num_v_dim,
            cross_attn_widening_factor = cross_attn_config.widening_factor,
            # cross_attn_norm_first = cross_attn_config.norm_first,

            num_self_attn_heads = self_attn_config.num_heads,
            num_self_attn_qk_dim = self_attn_config.num_qk_dim,
            num_self_attn_v_dim = self_attn_config.num_v_dim,
            num_self_attn_layers_per_block = self_attn_config.num_layers,
            self_attn_widening_factor = self_attn_config.widening_factor,
            # self_attn_norm_first = self_attn_config.norm_first,

            num_cross_attn_layers = num_cross_attn_layers,
            num_self_attn_blocks = num_self_attn_blocks,
        )

@dataclass
class PerceiverDecoderConfig(CrossAttentionLayerConfig):
    num_latents: int
    num_latent_dim: int
    init_scale: float = 0.02

    @classmethod
    def create_from_config(cls, latent_query_config: LatentQueryConfig, 
                           cross_attn_config: CrossAttentionLayerConfig,
                           **kwargs
                           ):
        return cls(
            num_latents = latent_query_config.num_latents,
            num_latent_dim = latent_query_config.num_latent_dim,
            init_scale = latent_query_config.init_scale,

            num_heads = cross_attn_config.num_heads,
            num_q_input_dim = cross_attn_config.num_q_input_dim,
            num_kv_input_dim = cross_attn_config.num_kv_input_dim,
            num_qk_dim = cross_attn_config.num_qk_dim,
            num_v_dim = cross_attn_config.num_v_dim,
            widening_factor = cross_attn_config.widening_factor,
            **kwargs
        )

