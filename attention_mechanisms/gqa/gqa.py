from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from ..configs import BaseConfig
from ..utils.rope import RotaryPositionEmbedding
from ..utils import xavier_uniform

from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Optional
from ..configs import BaseConfig
from ..utils.rope import RotaryPositionEmbedding
from ..utils import xavier_uniform
import jax

class GQAAttention(nn.Module):
    """
    Implements standard scaled dot-product attention with an optional
    Grouped Query Attention (GQA) mechanism using repeated keys and values.
    """
    config: BaseConfig

    def setup(self):
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.group_size = self.config.group_size   # => "kv heads"
        self.hidden_size = self.config.hidden_size

        # For GQA:
        # - Q is projected to all heads (e.g., 64)
        # - K, V are projected to a reduced number of heads (e.g., 8) and repeated at runtime.
        self.num_kv_heads = self.num_heads // self.group_size  # 64 // 8 = 8

        # Precomputed scaling factor for efficiency
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        
        self.rope = RotaryPositionEmbedding(config=self.config)

        self.q_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="q_proj")
        self.k_proj = nn.Dense(features=self.num_kv_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="k_proj")
        self.v_proj = nn.Dense(features=self.num_kv_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="v_proj")
        self.out_proj = nn.Dense(features=self.hidden_size,
                                 kernel_init=xavier_uniform, name="out_proj")

    def __call__(self,
                 hidden_states: Float[Array, "batch seq_len hidden_size"],
                 mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None
                ) -> Float[Array, "batch seq_len hidden_size"]:
        """
        Forward pass of the softmax attention mechanism.

        Args:
            hidden_states: A float Tensor of shape (batch, seq_len, hidden_size),
                where hidden_size can be num_heads * head_dim or similar.
            mask: An optional boolean or float mask of shape
                (batch, 1, seq_len, seq_len) indicating which positions
                are valid (True) or invalid (False). If True means keep,
                those entries are left as-is. If False, they are set
                to a large negative number before softmax.

        Returns:
            A float Tensor of shape (batch, seq_len, hidden_size)
            after applying the attention mechanism.
        """

        # Project inputs to query, key, and value tensors
        q = self.q_proj(hidden_states)  # Shape: (b, s, 8192)
        k = self.k_proj(hidden_states)  # Shape: (b, s, 1024)
        v = self.v_proj(hidden_states)  # Shape: (b, s, 1024)

        # Reshape Q, K, V for multi-head attention
        # Q: (b, s, 64, 128), K: (b, s, 8, 128), V: (b, s, 8, 128)
        batch_size, seq_len, _ = hidden_states.shape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary position embeddings to Q and K
        q, k = self.rope(q, k)

        # Transpose for attention computation
        # Q: (b, 64, s, 128), K: (b, 8, s, 128), V: (b, 8, s, 128)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        if self.group_size > 1:
            k = jnp.repeat(k, self.group_size, axis=1)  # (b, 64, s, 128)
            v = jnp.repeat(v, self.group_size, axis=1)  # (b, 64, s, 128)
        
        attn_scores = jax.lax.dot_general(
            q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))
        ) * self.scale  # Shape: (b, 64, s, s)

        if mask is not None:
            mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))
            mask = jnp.repeat(mask, self.num_heads, axis=1)
            attn_scores += mask * -1e9
            
        attn_probs = nn.softmax(attn_scores, axis=-1)

        attn_output = jax.lax.dot_general(
            attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1)))
        )  # (b, 64, s, 128)
        
        # Reshape the attention output back to the original format
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return self.out_proj(attn_output)