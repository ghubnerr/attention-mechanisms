from typing import Optional
from dataclasses import dataclass, field, asdict
from .base import BaseConfig


@dataclass(frozen=True)
class GPT2ModelConfig(BaseConfig):
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    max_seq_len: int = 1024
    num_experts: Optional[int] = field(default=8)
    top_k: Optional[int] = field(default=2)
    aux_loss_coef: Optional[float] = field(default=0.01)
    rope_fraction: float = field(default=0.5)
    rope_base_freq: float = field(default=10000.0)
    compressed_dim_kv: Optional[int] = field(default=None)
    compressed_dim_q: Optional[int] = field(default=None)
    rope_head_dim: Optional[int] = field(default=None)
    rms_norm_epsilon: Optional[float] = field(default=1e-4)

    @property
    def num_layers(self) -> int:
        return self.n_layer

    @property
    def num_heads(self) -> int:
        return self.n_head

    @property
    def hidden_size(self) -> int:
        return self.n_embd

    @property
    def head_dim(self) -> int:
        if self.n_head == 0:
            return 0
        return self.n_embd // self.n_head

    @property
    def ffw_hidden_size(self) -> int:
        return self.n_embd * 4

    @property
    def group_size(self) -> Optional[int]:
        return 1

    def __post_init__(self):
        if self.n_head > 0 and self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.num_experts == 0:
            self.top_k = None

    def to_dict(self):
        data = asdict(self)
        data['num_layers'] = self.num_layers
        data['num_heads'] = self.num_heads
        data['hidden_size'] = self.hidden_size
        data['head_dim'] = self.head_dim
        data['ffw_hidden_size'] = self.ffw_hidden_size
        data['group_size'] = self.group_size
        return data
