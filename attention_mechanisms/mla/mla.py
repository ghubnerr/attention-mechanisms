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
    Autoregressive MLA module with 'decoupled' RoPE, closely following the
    DeepSeek-V2 paper.  This approach keeps a single per-token RoPE key vector
    (k^R_t of size `rope_head_dim`), then broadcasts it across heads after the RoPE
    transform.  Meanwhile, each head has its own per-head rope queries (q^R_t),
    so W_UQ_R is shaped (compressed_dim_q, num_heads*rope_head_dim).

    The final "absorption trick" is also applied, so we keep cKV untransformed
    in the cache, and only re-transform new queries to match cKV space.

    Parameters in this module:
      - W_DQ (hidden_size, compressed_dim_q)
      - W_DKV (hidden_size, compressed_dim_kv)
      - W_UQ_C (compressed_dim_q, num_heads*head_dim)
      - W_UQ_R (compressed_dim_q, num_heads*rope_head_dim)
      - W_KR   (hidden_size, rope_head_dim)  # single per-token RoPE key
      - W_UK_C (compressed_dim_kv, num_heads*head_dim)
      - W_UV_C (compressed_dim_kv, num_heads*head_dim)
      - W_O    (num_heads*head_dim, hidden_size)
    """
    config: BaseConfig

    def setup(self):
        """
        Initializes the model parameters using Xavier uniform initialization.

        Parameters:
        - W_DQ: Projects hidden states to a compressed query representation.
        - W_DKV: Projects hidden states to a compressed key-value representation.
        - W_UQ_C: Maps compressed queries to contextual queries.
        - W_UQ_R: Maps compressed queries to RoPE-enhanced queries (per-head).
        - W_KR: Projects hidden states to a single (rope_head_dim) key for RoPE.
        - W_UK_C: Transforms compressed keys for the "absorption trick."
        - W_UV_C: Transforms compressed values for the "absorption trick."
        - W_O: Final projection back to hidden_size.
        - rope: Rotary Position Embedding module (partial or full).
        """
        # 1) Low-rank (down) projection for queries and for KV
        self.W_DQ   = self.param(
            "W_DQ", nn.initializers.xavier_uniform(),
            (self.config.hidden_size, self.config.compressed_dim_q)
        )
        self.W_DKV  = self.param(
            "W_DKV", nn.initializers.xavier_uniform(),
            (self.config.hidden_size, self.config.compressed_dim_kv)
        )

        # 2) Up-projection for queries: qC_t and qR_t
        #    shape => (compressed_dim_q, num_heads*head_dim) or
        #              (compressed_dim_q, num_heads*rope_head_dim)
        self.W_UQ_C = self.param(
            "W_UQ_C", nn.initializers.xavier_uniform(),
            (self.config.compressed_dim_q, self.config.num_heads * self.config.head_dim)
        )
        self.W_UQ_R = self.param(
            "W_UQ_R", nn.initializers.xavier_uniform(),
            (self.config.compressed_dim_q, self.config.num_heads * self.config.rope_head_dim)
        )

        # 3) Single per-token rope key => shape (hidden_size, rope_head_dim)
        self.W_KR   = self.param(
            "W_KR", nn.initializers.xavier_uniform(),
            (self.config.hidden_size, self.config.rope_head_dim)
        )

        # 4) Up-projection for compressed KV => used by the "absorption trick."
        self.W_UK_C = self.param(
            "W_UK_C", nn.initializers.xavier_uniform(),
            (self.config.compressed_dim_kv, self.config.num_heads * self.config.head_dim)
        )
        self.W_UV_C = self.param(
            "W_UV_C", nn.initializers.xavier_uniform(),
            (self.config.compressed_dim_kv, self.config.num_heads * self.config.head_dim)
        )

        # 5) Final output projection => shape (num_heads*head_dim, hidden_size)
        self.W_O    = self.param(
            "W_O", nn.initializers.xavier_uniform(),
            (self.config.num_heads * self.config.head_dim, self.config.hidden_size)
        )

        # RoPE module for q^R and k^R
        self.rope = RotaryPositionEmbedding(self.config)

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_size"],
        mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None,
        cached_cKV: Optional[Float[Array, "batch prefix_len compressed_dim_kv"]] = None,
        cached_kR: Optional[Float[Array, "batch prefix_len rope_head_dim"]] = None
    ) -> Tuple[
        Float[Array, "batch seq_len hidden_size"],
        Float[Array, "batch prefix_len compressed_dim_kv"],
        Float[Array, "batch prefix_len rope_head_dim"]
    ]:
        """
        Performs autoregressive MLA with "decoupled RoPE" while maintaining per-head attention.
        """
        B, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "This implementation is designed for one token per step."
        nH, rH = self.config.num_heads, self.config.rope_head_dim
        head_dim = self.config.head_dim

        if cached_cKV is None:
            cached_cKV = jnp.zeros((B, 0, self.config.compressed_dim_kv), dtype=hidden_states.dtype)
        if cached_kR is None:
            cached_kR = jnp.zeros((B, 0, rH), dtype=hidden_states.dtype)
        prefix_len = cached_cKV.shape[1]

        # Compute compressed representations (same as before)
        cQ_t = jnp.einsum('bsh,hq->bsq', hidden_states, self.W_DQ)
        cKV_t = jnp.einsum('bsh,hv->bsv', hidden_states, self.W_DKV)

        # Compute per-head queries (contextual and RoPE parts)
        qC_t = jnp.einsum('bsq,qc->bsc', cQ_t, self.W_UQ_C).reshape(B, seq_len, nH, head_dim)
        qR_t = jnp.einsum('bsq,qr->bsr', cQ_t, self.W_UQ_R).reshape(B, seq_len, nH, rH)

        # Compute RoPE key (same as before)
        kR_t = jnp.einsum('bsh,hr->bsr', hidden_states, self.W_KR).reshape(B, seq_len, 1, rH)
        qR_t, kR_t = self.rope(qR_t, kR_t)
        kR_t_broadcasted = jnp.broadcast_to(kR_t, (B, seq_len, nH, rH))
        
        # Absorption trick: instead of computing k_C = W_UK * cKV for each token,
        # we compute q_eff = q_C * W_UK^T for the current token
        W_UK_C_T = jnp.transpose(self.W_UK_C, (1, 0))
        W_UK_C_reshaped = W_UK_C_T.reshape(nH, head_dim, -1)
        
        # KEY CHANGE: Maintain per-head structure for scores
        # Reshape qC_t to (B, seq_len, nH, head_dim) for per-head processing
        qC_per_head = qC_t
        
        # Compute effective query per head using the absorption trick
        qC_eff = jnp.einsum('bsnh,nhk->bsnk', qC_per_head, W_UK_C_reshaped)
        
        # Compute scores for prefix tokens (per head)
        if prefix_len > 0:
            # Compute contextual score component per head
            score_Cprefix = jnp.einsum('bsnk,bLk->bsnL', qC_eff, cached_cKV)
            
            # Compute RoPE score component per head
            cached_kR_expanded = jnp.repeat(cached_kR, 1, axis=2).reshape(B, prefix_len, 1, rH)
            cached_kR_broadcasted = jnp.broadcast_to(cached_kR_expanded, (B, prefix_len, nH, rH))
            score_Rprefix = jnp.einsum('bsnr,bLnr->bsnL', qR_t, cached_kR_broadcasted)
            
            # Combine score components
            score_prefix = score_Cprefix + score_Rprefix
        else:
            score_prefix = jnp.zeros((B, seq_len, nH, 0), dtype=hidden_states.dtype)
        
        # Compute score for the new token (per head)
        new_token_score_C = jnp.einsum('bsnk,bsk->bsn', qC_eff, cKV_t)[..., None]
        new_token_score_R = jnp.einsum('bsnr,bsnr->bsn', qR_t, kR_t_broadcasted)[..., None]
        new_token_score = new_token_score_C + new_token_score_R
        
        # Concatenate prefix and new token scores
        scores = jnp.concatenate([score_prefix, new_token_score], axis=-1)
        
        # Scale and mask
        scores = scores * (1.0 / jnp.sqrt(head_dim + rH))
        if mask is not None:
            # Expand mask for all heads
            expanded_mask = jnp.broadcast_to(mask, (B, nH, seq_len, mask.shape[-1]))
            scores = scores + expanded_mask * -1e9
        
        # Apply softmax per head
        attn_probs = nn.softmax(scores, axis=-1)
        
        # Value computation with absorption trick
        # Reshape W_UV_C to per-head format for the absorption trick
        W_UV_C_reshaped = self.W_UV_C.reshape(-1, nH, head_dim)
        W_O_reshaped = self.W_O.reshape(nH, head_dim, -1)

        # Compute output with attention probabilities (per head)
        if prefix_len > 0:
            # Process prefix context
            prefix_values = jnp.einsum('bLk,knh->bLnh', cached_cKV, W_UV_C_reshaped)
            prefix_agg = jnp.einsum('bsnL,bLnh->bsnh', attn_probs[..., :-1], prefix_values)
        else:
            prefix_agg = 0.0

        # Process new token
        new_values = jnp.einsum('bsk,knh->bsnh', cKV_t, W_UV_C_reshaped)
        new_agg = attn_probs[..., -1:] * new_values

        # Combine aggregated values
        attn_output = prefix_agg + new_agg

        # Project to output dimension
        output = jnp.einsum('bsnh,nhc->bsc', attn_output, W_O_reshaped)
        
        # Update cache
        new_cached_cKV = jnp.concatenate([cached_cKV, cKV_t], axis=1)
        new_cached_kR = jnp.concatenate([cached_kR, kR_t[:, :, 0, :]], axis=1)
        
        return output, new_cached_cKV, new_cached_kR