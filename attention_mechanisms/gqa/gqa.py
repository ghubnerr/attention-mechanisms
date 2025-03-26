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
    
    
class AutoRegGQAAttention(nn.Module):
    config: BaseConfig
    
    """
    Implements standard auto-regressive scaled dot-product attention with an optional
    Grouped Query Attention (GQA) mechanism using repeated keys and values.
    """
    config: BaseConfig

    def setup(self):
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.group_size = self.config.group_size
        self.hidden_size = self.config.hidden_size

        self.num_kv_heads = self.num_heads // self.group_size
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

        self.rope = RotaryPositionEmbedding(config=self.config)

        self.q_proj = nn.Dense(self.num_heads * self.head_dim, kernel_init=xavier_uniform, name="q_proj")
        self.k_proj = nn.Dense(self.num_kv_heads * self.head_dim, kernel_init=xavier_uniform, name="k_proj")
        self.v_proj = nn.Dense(self.num_kv_heads * self.head_dim, kernel_init=xavier_uniform, name="v_proj")
        self.out_proj = nn.Dense(self.hidden_size, kernel_init=xavier_uniform, name="out_proj")

    def __call__(self,
                 hidden_states: Float[Array, "batch seq_len hidden_size"],
                 past_key: Optional[Float[Array, "batch num_heads past_len head_dim"]] = None,
                 past_value: Optional[Float[Array, "batch num_heads past_len head_dim"]] = None,
                 mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None) -> Tuple[Float[Array, "batch seq_len hidden_size"], Float[Array, "batch num_heads seq_len head_dim"], Float[Array, "batch num_heads seq_len head_dim"]]:
        """
        Forward pass for AutoRegGQAAttention.
        
        Args:
            hidden_states (Float[Array, "batch seq_len hidden_size"]): The input hidden states of shape (batch, seq_len, hidden_size).
            past_key (Optional[Float[Array, "batch num_heads past_len head_dim"]]): The cached keys from previous steps. Defaults to None.
            past_value (Optional[Float[Array, "batch num_heads past_len head_dim"]]): The cached values from previous steps. Defaults to None.
            mask (Optional[Float[Array, "batch 1 seq_len seq_len"]]): The attention mask to prevent attending to future tokens. Defaults to None.
        
        Returns:
            Tuple[Float[Array, "batch seq_len hidden_size"], Float[Array, "batch num_heads seq_len head_dim"], Float[Array, "batch num_heads seq_len head_dim"]]:
                - attn_output: The attention-processed output of shape (batch, seq_len, hidden_size).
                - new_past_key: Updated keys including the current input, of shape (batch, num_heads, seq_len, head_dim).
                - new_past_value: Updated values including the current input, of shape (batch, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self.rope(q, k)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        if past_key is not None and past_value is not None:
            k = jnp.concatenate([past_key, k], axis=2)
            v = jnp.concatenate([past_value, v], axis=2)

        # Store new keys and values
        new_past_key = k
        new_past_value = v

        if self.group_size > 1:
            k = jnp.repeat(k, self.group_size, axis=1)
            v = jnp.repeat(v, self.group_size, axis=1)

        attn_scores = jax.lax.dot_general(q, k, dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))) * self.scale

        if mask is not None:
            seq_len_k = k.shape[2]
            mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len_k))
            mask = jnp.repeat(mask, self.num_heads, axis=1)
            attn_scores += mask * -1e9

        attn_probs = nn.softmax(attn_scores, axis=-1)

        attn_output = jax.lax.dot_general(attn_probs, v, dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))))

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, new_past_key, new_past_value