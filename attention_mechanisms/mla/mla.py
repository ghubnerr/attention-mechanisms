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
    config: BaseConfig

    def setup(self):
        """
        Initializes the model parameters using Xavier uniform initialization.

        Parameters:
        - W_DQ: Projects hidden states to a compressed query representation.
        - W_DKV: Projects hidden states to a compressed key-value representation.
        - W_UQ_C: Maps compressed queries to contextual queries.
        - W_UQ_R: Maps compressed queries to RoPE-enhanced queries.
        - W_KR: Projects hidden states to RoPE-enhanced keys.
        - W_UK_C: Transforms compressed keys for absorption trick.
        - W_UV_C: Transforms compressed values for absorption trick.
        - W_O: Projects final attention output back to hidden size.
        - rope: Rotary Position Embedding module for positional encoding.
        """
        self.W_DQ   = self.param("W_DQ",   xavier_uniform,
                                 (self.config.hidden_size, self.config.compressed_dim_q))
        self.W_DKV  = self.param("W_DKV",  xavier_uniform,
                                 (self.config.hidden_size, self.config.compressed_dim_kv))
        self.W_UQ_C = self.param("W_UQ_C", xavier_uniform,
                                 (self.config.compressed_dim_q,
                                  self.config.num_heads * self.config.head_dim))
        self.W_UQ_R = self.param("W_UQ_R", xavier_uniform,
                                 (self.config.compressed_dim_q,
                                  self.config.num_heads * self.config.rope_head_dim))
        self.W_KR   = self.param("W_KR",   xavier_uniform,
                                 (self.config.hidden_size,
                                  self.config.num_heads * self.config.rope_head_dim))

        self.W_UK_C = self.param("W_UK_C", xavier_uniform,
                                 (self.config.compressed_dim_kv,
                                  self.config.num_heads * self.config.head_dim))
        self.W_UV_C = self.param("W_UV_C", xavier_uniform,
                                 (self.config.compressed_dim_kv,
                                  self.config.num_heads * self.config.head_dim))

        self.W_O    = self.param("W_O", xavier_uniform,
                                 (self.config.num_heads * self.config.head_dim,
                                  self.config.hidden_size))
        self.rope = RotaryPositionEmbedding(self.config)

    def __call__(self,
                 hidden_states: Float[Array, "batch seq_len hidden_size"],
                 mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None,
                 cached_cKV: Optional[Float[Array, "batch prefix_len compressed_dim_kv"]] = None,
                 cached_kR: Optional[Float[Array, "batch prefix_len num_heads rope_head_dim"]] = None
                 ) -> tuple[Float[Array, "batch seq_len hidden_size"],
                            Float[Array, "batch prefix_len compressed_dim_kv"],
                            Float[Array, "batch prefix_len num_heads rope_head_dim"]]:
        """
        Args:
            hidden_states (Float[Array, "batch seq_len hidden_size"]): Input tensor.
            mask (Optional[Float[Array, "batch 1 seq_len seq_len"]]): Mask tensor.
            cached_cKV (Optional[Float[Array, "batch prefix_len compressed_dim_kv"]]): Cached keys/values.
            cached_kR (Optional[Float[Array, "batch prefix_len num_heads rope_head_dim"]]): Cached keys for RoPE.

        Returns:
            tuple: A tuple containing the final attention output, updated cached compressed keys/values, and updated cached RoPE-enhanced keys.
        
        The implementation performs the following steps:
        1. Computes compressed query 'cQ_t' by projecting the hidden states via W_DQ.
        2. Splits the query into 'qC_t' (contextual query) and 'qR_t' (RoPE-enhanced query).
        3. Computes RoPE-enhanced key 'kR_t' using W_KR.
        4. Computes compressed keys/values 'cKV_t' using W_DKV.
        5. Applies absorption trick to avoid repeated transformation of keys/values.
        6. Combines queries and keys to obtain final attention scores.
        7. Computes final attention probabilities and produces output.
        8. Updates caches with new compressed keys/values and RoPE-enhanced keys.
        """

        B, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "This implementation handles 1 token per step."

        # === 1) Compressed query:  cQ_t = W_DQ * h_t  ===
        cQ_t = jnp.einsum("bsh,hq->bsq", hidden_states, self.W_DQ)

        # === 2) Query parts: qC_t and qR_t ===
        #     qC_t = W_UQ_C * cQ_t
        #     qR_t = RoPE( W_UQ_R * cQ_t ), etc.
        qC_t = jnp.einsum("bsq,qc->bsc", cQ_t, self.W_UQ_C)
        qR_t = jnp.einsum("bsq,qr->bsr", cQ_t, self.W_UQ_R)

        # === 3) Key part for RoPE: kR_t ===
        kR_t = jnp.einsum("bsh,hr->bsr", hidden_states, self.W_KR)

        # --- Reshape qR_t and kR_t for RoPE ---
        B, _, num_heads_rope_head_dim = qR_t.shape  # qR_t is (B, 1, num_heads * rope_head_dim)
        num_heads = self.config.num_heads
        rope_head_dim = self.config.rope_head_dim

        # Reshape: (B, 1, num_heads, rope_head_dim)
        qR_t = qR_t.reshape(B, 1, num_heads, rope_head_dim)
        kR_t = kR_t.reshape(B, 1, num_heads, rope_head_dim)

        # --- Apply RoPE ---
        qR_t, kR_t = self.rope(qR_t, kR_t)

        # --- Flatten back to original shape ---
        qR_t = qR_t.reshape(B, 1, num_heads * rope_head_dim)
        kR_t = kR_t.reshape(B, 1, num_heads * rope_head_dim)


        # === 4) Compressed KV: cKV_t = W_DKV * h_t  ===
        cKV_t = jnp.einsum("bsh,hv->bsv", hidden_states, self.W_DKV)

        # Prepare empty cache if none
        if cached_cKV is None:
            cached_cKV = jnp.zeros((B, 0, self.config.compressed_dim_kv),
                                   dtype=hidden_states.dtype)
            cached_kR  = jnp.zeros((B, 0,
                                    self.config.num_heads*self.config.rope_head_dim),
                                   dtype=hidden_states.dtype)

        # ------------------------------------------------------------------
        # ABSORPTION TRICK:
        #
        # Instead of computing
        #     kC_prefix = cached_cKV @ W_UK_C
        # each time, we rewrite the dot product as
        #     (qC_t)ᵀ · (W_UK_C @ cKV_prefix) = (W_UK_Cᵀ @ qC_t)ᵀ · cKV_prefix
        #
        # so we only re‐transform the new query by W_UK_Cᵀ.  The same applies
        # for the new token’s cKV (for keys).  Then for values we fold W_UV_C
        # into the output projection W_O at the end.
        # ------------------------------------------------------------------

        # === 5) "Absorbed" query for the compressed keys: qC_eff = qC_t * W_UK_C^T
        #     shape: (B, 1, compressed_dim_kv)
        W_UK_C_T = jnp.transpose(self.W_UK_C, (1,0))  # shape (n_heads*head_dim, compressed_dim_kv)
        qC_eff   = jnp.einsum("bsc,ck->bsk", qC_t, W_UK_C_T)
        # Now qC_eff can dot with cached_cKV or cKV_t directly.

        # Dot product for prefix
        if cached_cKV.shape[1] > 0:
            score_Cprefix = jnp.einsum("bsk,bLk->bsL", qC_eff, cached_cKV)
        else:
            score_Cprefix = jnp.zeros((B, 1, 0), dtype=hidden_states.dtype)

        # Dot product for new token
        new_token_score_C = jnp.einsum("bsk,bsk->bs", qC_eff, cKV_t)[:, None]

        # === 6) The "R" part: we still do normal qR . kR dot products.
        #     shape (B,1, prefix_len) and (B,1,1), same as above
        if cached_kR.shape[1] > 0:
            score_Rprefix = jnp.einsum("bsr,bLr->bsL", qR_t, cached_kR)
        else:
            score_Rprefix = jnp.zeros_like(score_Cprefix)

        new_token_score_R = jnp.einsum("bsr,bsr->bs", qR_t, kR_t)[:, None]

        # === 7) Combine prefix & new token for final attention scores  ===
        prefix_scores = score_Cprefix + score_Rprefix  # shape (B,1,prefix_len)
        new_token_score = new_token_score_C + new_token_score_R  # shape (B,1)

        scores = jnp.concatenate([prefix_scores, new_token_score], axis=-1)
        scores = scores / jnp.sqrt(self.config.head_dim + self.config.rope_head_dim)
        if mask is not None:
            scores += mask * -1e9

        attn_probs = nn.softmax(scores, axis=-1)  # shape (B,1, prefix_len+1)

        # === 8) Now absorb W_UV_C into W_O so we do not re‐project cKV.  ===
        # We define W_VO = W_UV_C @ W_O.
        #   - W_UV_C : (compressed_dim_kv, n_heads*head_dim)
        #   - W_O    : (n_heads*head_dim, hidden_size)
        # => W_VO is shape (compressed_dim_kv, hidden_size)
        W_VO = jnp.einsum("kv,vc->kc", self.W_UV_C, self.W_O)  # (compressed_dim_kv, hidden_size)

        # For the prefix tokens, do: prefix_value = cached_cKV @ W_VO => shape (B, prefix_len, hidden_size)
        if cached_cKV.shape[1] > 0:
            prefix_value = jnp.einsum("bLk,kc->bLc", cached_cKV, W_VO)  # (B,prefix_len,hidden_size)
            prefix_value_agg = jnp.einsum(
                "bsl,blc->bsc", attn_probs[..., :-1], prefix_value
            )  # => (B,1,hidden_size)
        else:
            prefix_value_agg = 0.0

        # For the newly generated token: new_value = cKV_t @ W_VO => (B,1,hidden_size)
        new_value = jnp.einsum("bsk,kc->bsc", cKV_t, W_VO)
        new_value_agg = attn_probs[..., -1:] * new_value  # shape (B,1,hidden_size)

        # Final attention output: (B,1,hidden_size)
        output = prefix_value_agg + new_value_agg

        # === 9) Append the new cKV & kR to the caches.  ===
        new_cached_cKV = jnp.concatenate([cached_cKV, cKV_t], axis=1)
        new_cached_kR  = jnp.concatenate([cached_kR,  kR_t ], axis=1)

        return output, new_cached_cKV, new_cached_kR
