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
        # Precomputed scaling factor
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        self.rope = RotaryPositionEmbedding(config=self.config)

        # Define projections
        self.q_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="q_proj",
                               dtype=self.config.dtype,
                               param_dtype=self.config.param_dtype)
        self.k_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="k_proj",
                               dtype=self.config.dtype,
                               param_dtype=self.config.param_dtype)
        self.v_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="v_proj",
                               dtype=self.config.dtype,
                               param_dtype=self.config.param_dtype)
        self.out_proj = nn.Dense(features=self.hidden_size,
                                 kernel_init=xavier_uniform, name="out_proj",
                                 dtype=self.config.dtype,
                                 param_dtype=self.config.param_dtype)

    def __call__(self,
                 hidden_states: Float[Array, "batch seq_len hidden_size"],
                 mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None
                 ) -> Float[Array, "batch seq_len hidden_size"]:

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim)

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

        attn_probs = nn.softmax(attn_scores, axis=-1).astype(jnp.float32)

        attn_output = jax.lax.dot_general(
            attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1)))
        )

        attn_output = attn_output.transpose(
            0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


class AutoRegMHSAttention(nn.Module):
    """
    Implements Incremental Multi-Head Self-Attention (IMHSA)
    """
    config: BaseConfig

    def setup(self):
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        # Precomputed scaling factor
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        self.rope = RotaryPositionEmbedding(config=self.config)

        # Define projections
        self.q_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="q_proj",
                               dtype=self.config.dtype,
                               param_dtype=self.config.param_dtype)
        self.k_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="k_proj",
                               dtype=self.config.dtype,
                               param_dtype=self.config.param_dtype)
        self.v_proj = nn.Dense(features=self.num_heads * self.head_dim,
                               kernel_init=xavier_uniform, name="v_proj",
                               dtype=self.config.dtype,
                               param_dtype=self.config.param_dtype)
        self.out_proj = nn.Dense(features=self.hidden_size,
                                 kernel_init=xavier_uniform, name="out_proj",
                                 dtype=self.config.dtype,
                                 param_dtype=self.config.param_dtype)

    def __call__(self,
                 hidden_states: Float[Array, "batch seq_len hidden_size"],
                 mask: Optional[Float[Array,
                                      "batch 1 seq_len seq_len"]] = None,
                 past_key: Optional[Float[Array,
                                          "batch num_heads past_len head_dim"]] = None,
                 past_value: Optional[Float[Array,
                                            "batch num_heads past_len head_dim"]] = None
                 ) -> Tuple[Float[Array, "batch seq_len hidden_size"], Float[Array, "batch num_heads seq_len head_dim"], Float[Array, "batch num_heads seq_len head_dim"]]:

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim)

        q, k = self.rope(q, k)

        q, k, v = map(lambda x: jnp.transpose(x, (0, 2, 1, 3)), (q, k, v))

        # Concatenate past keys and values if provided (Incremental processing)
        if past_key is not None and past_value is not None:
            k = jnp.concatenate([past_key, k], axis=2)
            v = jnp.concatenate([past_value, v], axis=2)

        attn_scores = jax.lax.dot_general(
            q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))
        ) * self.scale

        if mask is not None:
            seq_len_total = k.shape[2]
            mask = jnp.broadcast_to(
                mask, (batch_size, 1, seq_len, seq_len_total))
            mask = jnp.repeat(mask, self.num_heads, axis=1)
            attn_scores += mask * -1e9

        attn_probs = nn.softmax(
            attn_scores, axis=-1).astype(jnp.float32)

        attn_output = jax.lax.dot_general(
            attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1)))
        )

        attn_output = attn_output.transpose(
            0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.out_proj(attn_output), k, v
