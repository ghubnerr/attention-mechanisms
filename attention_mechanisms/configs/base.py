import abc
from typing import Optional

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

