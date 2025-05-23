import abc
from typing import Optional

import jax.numpy as jnp


class BaseConfig(abc.ABC):
    ffw_hidden_size: int
    num_experts: Optional[int]
    top_k: Optional[int]

    num_heads: int
    head_dim: int
    group_size: Optional[int]
    num_layers: int
    hidden_size: int

    rope_fraction: float
    rope_base_freq: float

    # MLA
    compressed_dim_kv: Optional[int]
    compressed_dim_q: Optional[int]
    rope_head_dim: Optional[int]

    # Normalization
    rms_norm_epsilon: Optional[float]

    vocab_size: int
    max_seq_len: int

    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
