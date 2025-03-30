from typing import Optional
from dataclasses import dataclass
from .base import BaseConfig

@dataclass
class DeepSeekV2MiniConfig(BaseConfig): # these arent right 
    # MoE
    ffw_hidden_size: int = 9216
    num_experts: int = 32
    top_k: int = 2  # top-2 routing on MoE
    aux_loss_coef: float = 0.01

    # Attn
    num_heads: int = 64
    head_dim: int = 128
    group_size: int = 8  # for Group-Query Attention
    num_layers: int = 80
    linear_per_softmax: int = 7 # 7 transnormers, then 1 transformer
    hidden_size: int = 6144

    # RoPE
    rope_fraction: float = 0.5
    rope_base_freq: float = 10000
    
    compressed_dim_kv: Optional[int] = None
    compressed_dim_q: Optional[int] = None
    rope_head_dim: Optional[int] = None
    
    rms_norm_epsilon: float = 1e-6
    
    vocab_size: int = 32000,
    max_seq_len: int = 1024,