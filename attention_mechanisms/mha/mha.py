from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Optional
from ..configs import BaseConfig
from ..utils.rope import RotaryPositionEmbedding
from ..utils import xavier_uniform
import jax

class MHSAttention(nn.Module):
    """
    Implements standard scaled dot-product multi-head self-attention (MHSA)
    """
    config: BaseConfig
    
    def setup(self):
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        self.scale = 1.0 / jnp.sqrt(self.head_dim)  # Precomputed scaling factor
        self.rope = RotaryPositionEmbedding(config=self.config)
        
        # Define projections
        self.q_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="q_proj")
        self.k_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="k_proj")
        self.v_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="v_proj")
        self.out_proj = nn.Dense(features=self.hidden_size,
                                 kernel_init=xavier_uniform, name="out_proj")
        
    def __call__(self,
                 hidden_states: Float[Array, "batch seq_len hidden_size"],
                 mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None
                ) -> Float[Array, "batch seq_len hidden_size"]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q, k = self.rope(q, k)
        
        q, k, v = map(lambda x: jnp.transpose(x, (0, 2, 1, 3)), (q, k, v))
        
        attn_scores = jax.lax.dot_general(
            q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))
        ) * self.scale
        
        if mask is not None:
            mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))
            mask = jnp.repeat(mask, self.num_heads, axis=1)
            attn_scores += mask * -1e9
                    
        attn_probs = nn.softmax(attn_scores, axis=-1)
        
        attn_output = jax.lax.dot_general(
            attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1)))
        )
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return self.out_proj(attn_output)
    