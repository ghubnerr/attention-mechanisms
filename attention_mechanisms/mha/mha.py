from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from ..configs import BaseConfig
from ..utils.rope import RotaryPositionEmbedding
from ..utils import xavier_uniform

class SoftmaxAttention(nn.Module):
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
        # - Q is projected to 64 heads
        # - K, V are projected to only 8 heads, repeated at runtime.
        self.num_kv_heads = self.num_heads // self.group_size  # 64 // 8 = 8

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

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)  # (b, s, 8192)
        k = self.k_proj(hidden_states)  # (b, s, 1024)
        v = self.v_proj(hidden_states)  # (b, s, 1024)

        # Q => (b, s, 64, 128), K => (b, s, 8, 128), V => (b, s, 8, 128)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self.rope(q, k)

        # Q => (b, 64, s, 128), K => (b, 8, s, 128), V => (b, 8, s, 128)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # K, V => (b, 8, s, 128) => replicate 8 times => (b, 64, s, 128)
        k = jnp.repeat(k, self.group_size, axis=1)
        v = jnp.repeat(v, self.group_size, axis=1)

        # Scaled dot-product attention => attn_scores => shape (b, 64, s, s)
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        attn_scores = attn_scores / jnp.sqrt(self.head_dim)

        # (Optional) Apply causal or padding mask
        if mask is not None:
            # mask shape is (b, 1, s, s), broadcast to (b, 64, s, s)
            big_neg = jnp.array(-1e9, dtype=attn_scores.dtype)
            attn_scores = jnp.where(mask, attn_scores, big_neg)

        # Softmax over the last dimension (the "key" positions)
        attn_probs = nn.softmax(attn_scores, axis=-1)

        # Apply attention to values => (b, 64, s, 128)
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)

        # Reshape back to (b, s, 64*128=8192) => then project to 6144
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Note, from the paper Sec. 2.0 ¶3, it's hard to understand the
        # "hidden size" == 6144. Usually it's num_heads * head_dim (8192 for us)
        # The assumption is that there is this down-projection at the end
        return self.out_proj(attn_output)