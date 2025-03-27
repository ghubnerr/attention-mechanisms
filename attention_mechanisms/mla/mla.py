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
        self.config.num_heads = self.config.num_heads
        self.config.head_dim = self.config.head_dim
        self.config.hidden_size = self.config.hidden_size
        self.config.compressed_dim_kv = self.config.compressed_dim_kv
        self.config.compressed_dim_q = self.config.compressed_dim_q
        self.config.rope_head_dim = self.config.rope_head_dim
        self.rope = RotaryPositionEmbedding(config=self.config)
        self.scale = 1.0 / jnp.sqrt(self.config.head_dim + self.config.rope_head_dim)

        # W_DKV: (compressed_dim_kv, hidden_size)
        self.W_DKV = nn.Dense(self.config.compressed_dim_kv, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DKV")

        # W_UK:  (num_heads*head_dim, compressed_dim_kv)
        self.W_UK = nn.Dense(self.config.head_dim * self.config.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UK")

        # W_UV:  (num_heads*head_dim, compressed_dim_kv)
        self.W_UV = nn.Dense(self.config.head_dim * self.config.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UV")

        # W_DQ: (compressed_dim_q, seq_len)
        self.W_DQ = nn.Dense(self.config.compressed_dim_q, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DQ")

        # W_UQ: (num_heads*head_dim, seq_len)
        self.W_UQ = nn.Dense(self.config.head_dim * self.config.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UQ")

        # W_QR: (num_heads * rope_head_dim, compressed_dim_q)
        self.W_QR = nn.Dense(self.config.num_heads * self.config.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_QR")

        # W_KR: (rope_head_dim, seq_len)
        self.W_KR = nn.Dense(self.config.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_KR")

        # W_O: (hidden_size, num_heads * head_dim)
        self.W_O = nn.Dense(self.config.hidden_size, use_bias=False,
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
        assert hidden_dims == self.config.hidden_size, "Input hidden size does not match config"

        c_KV = self.W_DKV(hidden_states)
        k_C = self.W_UK(c_KV)
        v_C = self.W_UV(c_KV)

        c_Q = self.W_DQ(hidden_states)
        q_C = self.W_UQ(c_Q)

        q_R = self.W_QR(c_Q).reshape(batch_size, seq_len, self.config.num_heads, self.config.rope_head_dim)
        k_R = self.W_KR(hidden_states).reshape(batch_size, seq_len, 1, self.config.rope_head_dim)

        q_R, k_R = self.rope(q_R, k_R)
        k_R = jnp.broadcast_to(k_R, (batch_size, seq_len, self.config.num_heads, self.config.rope_head_dim))

        q_C = q_C.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        k_C = k_C.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        q = jnp.concatenate([q_C, q_R], axis=-1)
        k = jnp.concatenate([k_C, k_R], axis=-1)

        v = v_C.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_scores = jax.lax.dot_general(q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))) * self.scale

        if mask is not None:
            seq_len_total = k.shape[2]
            mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len_total))
            mask = jnp.repeat(mask, self.config.num_heads, axis=1)
            attn_scores += mask * -1e9

        attn_probs = nn.softmax(attn_scores, axis=-1)

        attn_output = jax.lax.dot_general(attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))))

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = attn_output.reshape(batch_size, seq_len, self.config.num_heads * self.config.head_dim)

        return self.W_O(attn_output)


class AutoRegMLAttention(nn.Module):
    """
    Implements Incremental Multi-Latent Attention (AutoRegMLA) for autoregressive inference.
    """
    config: BaseConfig
    
    @nn.compact
    def __call__(self,
                 hidden_states: jnp.ndarray,      # (B,1,hidden_size)
                 mask: jnp.ndarray = None,        # (B,1,1,total_prefix+1) if used
                 cached_cKV: jnp.ndarray = None,  # (B,prefix_len,compressed_dim_kv)
                 cached_kR: jnp.ndarray = None    # (B,prefix_len,num_heads,rope_head_dim)
                ):
        """
        Returns:
          output:         (B,1,hidden_size)
          new_cached_cKV: (B,prefix_len+1,compressed_dim_kv)
          new_cached_kR:  (B,prefix_len+1,num_heads,rope_head_dim)
        """
        B, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "Incremental decode expects seq_len=1."

        # === 1) Parameters ===
        # Down-projections
        W_DQ  = self.param("W_DQ",  xavier_uniform, 
                           (self.config.hidden_size, self.config.compressed_dim_q))
        W_DKV = self.param("W_DKV", xavier_uniform, 
                           (self.config.hidden_size, self.config.compressed_dim_kv))

        # Up-proj for Q's compressed -> multi-head "C" dimension
        W_UQ_C = self.param("W_UQ_C", xavier_uniform,
                            (self.config.compressed_dim_q, self.config.num_heads, self.config.head_dim))
        # Up-proj for Q's compressed -> multi-head "R" dimension (rope)
        W_UQ_R = self.param("W_UQ_R", xavier_uniform,
                            (self.config.compressed_dim_q, self.config.num_heads, self.config.rope_head_dim))

        # Up-proj for K/V's compressed -> multi-head "C"
        W_UK_C = self.param("W_UK_C", xavier_uniform,
                            (self.config.compressed_dim_kv, self.config.num_heads, self.config.head_dim))
        W_UV_C = self.param("W_UV_C", xavier_uniform,
                            (self.config.compressed_dim_kv, self.config.num_heads, self.config.head_dim))

        # Decoupled rope key
        W_KR   = self.param("W_KR", xavier_uniform,
                            (self.config.hidden_size, self.config.num_heads, self.config.rope_head_dim))

        # Final output projection
        # shape: (num_heads, head_dim, hidden_size)
        W_O = self.param("W_O", xavier_uniform,
                         (self.config.num_heads, self.config.head_dim, self.config.hidden_size))

        # === 2) Build compressed Q for the new token: cQ_t ===
        # shape (B,1,compressed_dim_q)
        cQ_t = jnp.einsum("bsh,hq->bsq", hidden_states, W_DQ)

        # === 3) Build q^C_t and q^R_t (split heads) ===
        # qC_t: (B,1,num_heads,head_dim)
        qC_t = jnp.einsum("bsq,qnd->bsnd", cQ_t, W_UQ_C)
        # qR_t: (B,1,num_heads,rope_head_dim)
        qR_t = jnp.einsum("bsq,qnr->bsnr", cQ_t, W_UQ_R)

        # === 4) Build decoupled rope key kR_t_raw for the new token, then apply RoPE ===
        # shape -> (B,1,num_heads,rope_head_dim)
        kR_t_raw = jnp.einsum("bsh,hnr->bsnr", hidden_states, W_KR)

        rope = RotaryPositionEmbedding(self.config)  # or pass config
        qR_t, kR_t = rope(qR_t, kR_t_raw)  # each => (B,1,num_heads,rope_head_dim)

        # === 5) Compressed KV for new token, shape (B,1,compressed_dim_kv) ===
        cKV_t = jnp.einsum("bsh,hv->bsv", hidden_states, W_DKV)

        # === 6) Build k^C_t, v^C_t for *just* the new token (but not prefix) ===
        # kC_t: (B,1,num_heads,head_dim)
        kC_t = jnp.einsum("bsv,vnd->bsnd", cKV_t, W_UK_C)
        vC_t = jnp.einsum("bsv,vnd->bsnd", cKV_t, W_UV_C)

        # === 7) If there's no cache, make empty prefix arrays ===
        if cached_cKV is None:
            cKV_prefix = jnp.zeros((B,0,self.config.compressed_dim_kv), hidden_states.dtype)
            kR_prefix  = jnp.zeros((B,0,self.config.num_heads,self.config.rope_head_dim), hidden_states.dtype)
        else:
            cKV_prefix = cached_cKV
            kR_prefix  = cached_kR
        prefix_len = cKV_prefix.shape[1]

        # === 8) Compute prefix's kC, shape (B,prefix_len,num_heads,head_dim) ===
        kC_prefix = jnp.einsum("btv,vnd->btnd", cKV_prefix, W_UK_C)

        # === 9) Dot with qC_t => prefix_scores_C: (B,1,num_heads,prefix_len) ===
        prefix_scores_C = jnp.einsum("bsnd,btnd->bsnt", qC_t, kC_prefix)

        # === 10) Compute prefix's rope key => (B,prefix_len,num_heads,rope_head_dim) is already cached
        # We simply do qR_t dot kR_prefix => prefix_scores_R: (B,1,num_heads,prefix_len)
        prefix_scores_R = jnp.einsum("bsnr,btnr->bsnt", qR_t, kR_prefix)

        # Summation => (B,1,num_heads,prefix_len)
        prefix_scores = prefix_scores_C + prefix_scores_R

        # === 11) Score for new token => qC_t dot kC_t + qR_t dot kR_t => (B,1,num_heads,1) ===
        # We do it in two steps (over head_dim, then rope_dim)
        new_score_C = jnp.einsum("bsnd,bsnd->bsn", qC_t, kC_t)  # (B,1,num_heads)
        new_score_R = jnp.einsum("bsnr,bsnr->bsn", qR_t, kR_t)  # (B,1,num_heads)
        new_score   = new_score_C + new_score_R                 # (B,1,num_heads)
        new_score   = new_score[...,None]                       # => (B,1,num_heads,1)

        # === 12) Concatenate prefix + new => (B,1,num_heads,prefix_len+1) ===
        attn_scores = jnp.concatenate([prefix_scores, new_score], axis=-1)

        # === 13) Scale + Mask => shape still (B,1,num_heads,prefix_len+1) ===
        scale_factor = jnp.sqrt(self.config.head_dim + self.config.rope_head_dim).astype(attn_scores.dtype)
        attn_scores = attn_scores / scale_factor

        if mask is not None:
            # mask: (B,1,1,prefix_len+1) => broadcast over num_heads dimension
            attn_scores = attn_scores + mask * -1e9

        # === 14) Softmax => (B,1,num_heads,prefix_len+1) ===
        attn_probs = nn.softmax(attn_scores, axis=-1)

        # === 15) Multiply prefix's cKV by W_UV_C => vC_prefix => (B,prefix_len,num_heads,head_dim) ===
        vC_prefix = jnp.einsum("btv,vnd->btnd", cKV_prefix, W_UV_C)

        # Weighted sum over prefix => (B,1,num_heads,head_dim)
        prefix_value_agg = jnp.einsum("bsnt,btnd->bsnd",
                                      attn_probs[...,:prefix_len], vC_prefix)

        # === 16) The new token's value => (B,1,num_heads,head_dim) * broadcast prob => same shape
        new_value_agg = attn_probs[..., -1:].reshape(B,1,self.config.num_heads,1) * vC_t

        # Sum => final attn_out => (B,1,num_heads,head_dim)
        attn_out = prefix_value_agg + new_value_agg

        # === 17) Output projection => (B,1,hidden_size)
        # W_O: (num_heads, head_dim, hidden_size)
        # sum over (num_heads,head_dim)
        output = jnp.einsum("bsnd,ndh->bsh", attn_out, W_O)

        # === 18) Update cache with new cKV, kR => keep shapes consistent
        if cached_cKV is not None:
            new_cached_cKV = jnp.concatenate([cached_cKV, cKV_t], axis=1)       # (B,prefix_len+1,compressed_dim_kv)
            new_cached_kR  = jnp.concatenate([cached_kR,  kR_t],  axis=1)       # (B,prefix_len+1,num_heads,rope_head_dim)
        else:
            new_cached_cKV = cKV_t  # (B,1,compressed_dim_kv)
            new_cached_kR  = kR_t   # (B,1,num_heads,rope_head_dim)

        return output, new_cached_cKV, new_cached_kR