from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Optional
from ..configs import BaseConfig
from ..utils.rope import RotaryPositionEmbedding
from ..utils import xavier_uniform
import jax

class MLAttention(nn.Module):
    """
    Multi-Latent Attention (MLA) Module.

    This module implements a custom multi-head attention mechanism with support for
    compressed query, key, and value representations, as well as rotary positional embeddings.

    Attributes:
        config (BaseConfig): Configuration object containing attention-related hyperparameters.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        hidden_size (int): Dimensionality of the input and output representations.
        compressed_dim_kv (int): Compressed dimensionality for keys and values.
        compressed_dim_q (int): Compressed dimensionality for queries.
        rope_head_dim (int): Dimensionality of the rotary positional embedding component.
        rope (RotaryPositionEmbedding): Module for computing rotary positional embeddings.
    """
    config: BaseConfig

    def setup(self):
        """
        Initializes the MLA module, defining the layers and parameters for the attention mechanism.

        Creates projection layers for compressed queries, keys, and values, along with output layers.
        Also initializes rotary positional embedding components.
        """
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        self.compressed_dim_kv = self.config.compressed_dim_kv
        self.compressed_dim_q = self.config.compressed_dim_q
        self.rope_head_dim = self.config.rope_head_dim
        self.rope = RotaryPositionEmbedding(config=self.config)
        self.scale = 1.0 / jnp.sqrt(self.head_dim + self.rope_head_dim)

        # W_DKV: (compressed_dim_kv, hidden_size)
        self.W_DKV = nn.Dense(self.compressed_dim_kv, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DKV")

        # W_UK:  (num_heads*head_dim, compressed_dim_kv)
        self.W_UK = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UK")

        # W_UV:  (num_heads*head_dim, compressed_dim_kv)
        self.W_UV = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UV")

        # W_DQ: (compressed_dim_q, seq_len)
        self.W_DQ = nn.Dense(self.compressed_dim_q, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DQ")

        # W_UQ: (num_heads*head_dim, seq_len)
        self.W_UQ = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UQ")

        # W_QR: (num_heads * rope_head_dim, compressed_dim_q)
        self.W_QR = nn.Dense(self.num_heads * self.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_QR")

        # W_KR: (rope_head_dim, seq_len)
        self.W_KR = nn.Dense(self.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_KR")

        # W_O: (hidden_size, num_heads * head_dim)
        self.W_O = nn.Dense(self.hidden_size, use_bias=False,
                            kernel_init=xavier_uniform, name="W_O")

    def __call__(self,
                hidden_states: Float[Array, "batch seq_len hidden_size"],
                mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None
            ) -> Float[Array, "batch seq_len hidden_size"]:
        """
        Applies the MLA mechanism to the input hidden states.

        Args:
            hidden_states (Float[Array, "batch seq_len hidden_size"]):
                The input tensor of shape (batch_size, seq_len, hidden_size) representing the input sequence.

            mask (Optional[Float[Array, "batch 1 seq_len seq_len"]], optional):
                Attention mask of shape (batch_size, 1, seq_len, seq_len). If provided, positions with value 1 are masked out.

        Returns:
            Float[Array, "batch seq_len hidden_size"]:
                The output tensor of the attention mechanism with the same shape as the input tensor.

        """
        batch_size, seq_len, hidden_dims = hidden_states.shape
        assert hidden_dims == self.hidden_size, "Input hidden size does not match config"

        c_KV = self.W_DKV(hidden_states)
        k_C = self.W_UK(c_KV)
        v_C = self.W_UV(c_KV)

        c_Q = self.W_DQ(hidden_states)
        q_C = self.W_UQ(c_Q)

        q_R = self.W_QR(c_Q).reshape(batch_size, seq_len, self.num_heads, self.rope_head_dim)
        k_R = self.W_KR(hidden_states).reshape(batch_size, seq_len, 1, self.rope_head_dim)

        q_R, k_R = self.rope(q_R, k_R)
        k_R = jnp.broadcast_to(k_R, (batch_size, seq_len, self.num_heads, self.rope_head_dim))

        q_C = q_C.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k_C = k_C.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q = jnp.concatenate([q_C, q_R], axis=-1)
        k = jnp.concatenate([k_C, k_R], axis=-1)

        v = v_C.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_scores = jax.lax.dot_general(q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))) * self.scale

        if mask is not None:
            seq_len_total = k.shape[2]
            mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len_total))
            mask = jnp.repeat(mask, self.num_heads, axis=1)
            attn_scores += mask * -1e9

        attn_probs = nn.softmax(attn_scores, axis=-1)

        attn_output = jax.lax.dot_general(attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))))

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.W_O(attn_output)


class AutoRegMLAttention(nn.Module):
    """
    Implements Incremental Multi-Latent Attention (AutoRegMLA) for autoregressive inference.
    """
    config: BaseConfig

    def setup(self):
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        self.compressed_dim_kv = self.config.compressed_dim_kv
        self.compressed_dim_q = self.config.compressed_dim_q
        self.rope_head_dim = self.config.rope_head_dim
        self.scale = 1.0 / jnp.sqrt(self.head_dim + self.rope_head_dim)
        self.rope = RotaryPositionEmbedding(config=self.config)

        self.W_DKV = nn.Dense(self.compressed_dim_kv, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DKV")

        self.W_UK = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UK")

        self.W_UV = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UV")

        self.W_DQ = nn.Dense(self.compressed_dim_q, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DQ")

        self.W_UQ = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UQ")

        self.W_QR = nn.Dense(self.num_heads * self.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_QR")

        self.W_KR = nn.Dense(self.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_KR")

        self.W_O = nn.Dense(self.hidden_size, use_bias=False,
                            kernel_init=xavier_uniform, name="W_O")

    def __call__(self,
                 hidden_states: Float[Array, "batch 1 hidden_size"],
                 mask: Optional[Float[Array, "batch 1 1 total_len"]] = None,
                 cached_c_KV: Optional[Float[Array, "batch cache_len compressed_dim_kv"]] = None,
                 cached_k_R: Optional[Float[Array, "batch cache_len rope_head_dim"]] = None
                ) -> Tuple[Float[Array, "batch 1 hidden_size"],
                           Float[Array, "batch cache_len+1 compressed_dim_kv"],
                           Float[Array, "batch cache_len+1 rope_head_dim"]]:

        batch_size, _, hidden_dims = hidden_states.shape
        assert hidden_dims == self.hidden_size, "Input hidden size does not match config"

        c_KV = self.W_DKV(hidden_states)
        k_C = self.W_UK(c_KV).reshape(batch_size, 1, self.num_heads, self.head_dim)
        v_C = self.W_UV(c_KV).reshape(batch_size, 1, self.num_heads, self.head_dim)

        c_Q = self.W_DQ(hidden_states)
        q_C = self.W_UQ(c_Q).reshape(batch_size, 1, self.num_heads, self.head_dim)

        q_R = self.W_QR(c_Q).reshape(batch_size, 1, self.num_heads, self.rope_head_dim)
        k_R = self.W_KR(hidden_states).reshape(batch_size, 1, 1, self.rope_head_dim)

        q_R, k_R_current = self.rope(q_R, k_R)
        k_R_current = jnp.broadcast_to(k_R_current, (batch_size, 1, self.num_heads, self.rope_head_dim))

        q = jnp.concatenate([q_C, q_R], axis=-1)
        k_current = jnp.concatenate([k_C, k_R_current], axis=-1)

        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, 1, dim)
        k_current = jnp.transpose(k_current, (0, 2, 1, 3))  # (batch, heads, 1, dim)
        v_C = jnp.transpose(v_C, (0, 2, 1, 3))  # (batch, heads, 1, head_dim)

        if cached_c_KV is not None and cached_k_R is not None:
            # Recompute keys and values from cached compressed KV latent
            cached_k_C = self.W_UK(cached_c_KV).reshape(batch_size, -1, self.num_heads, self.head_dim)
            cached_v_C = self.W_UV(cached_c_KV).reshape(batch_size, -1, self.num_heads, self.head_dim)
            cached_k_R_expanded = jnp.broadcast_to(cached_k_R[..., None, :], (batch_size, cached_k_R.shape[1], self.num_heads, self.rope_head_dim))

            cached_k = jnp.concatenate([cached_k_C, cached_k_R_expanded], axis=-1)

            cached_k = jnp.transpose(cached_k, (0, 2, 1, 3))
            cached_v_C = jnp.transpose(cached_v_C, (0, 2, 1, 3))

            k = jnp.concatenate([cached_k, k_current], axis=2)
            v = jnp.concatenate([cached_v_C, v_C], axis=2)
        else:
            k, v = k_current, v_C

        attn_scores = jax.lax.dot_general(q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))) * self.scale

        if mask is not None:
            attn_scores += mask * -1e9

        attn_probs = nn.softmax(attn_scores, axis=-1)

        attn_output = jax.lax.dot_general(attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))))

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, 1, -1)

        new_cached_c_KV = jnp.concatenate([cached_c_KV, c_KV], axis=1) if cached_c_KV is not None else c_KV
        new_cached_k_R = jnp.concatenate([cached_k_R, k_R.squeeze(2)], axis=1) if cached_k_R is not None else k_R.squeeze(2)

        return self.W_O(attn_output), new_cached_c_KV, new_cached_k_R