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
    Implements Multi-Latent Attention (MLA)    
    """
    config: BaseConfig
    
    def setup(self):
        """
        Initializes the MLA module
        """
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        self.compressed_dim_kv = self.config.compressed_dim_kv
        self.compressed_dim_q = self.config.compressed_dim_q
        self.rope_head_dim = self.config.rope_head_dim
        self.rope = RotaryPositionEmbedding(config=self.config)


        self.W_DKV = nn.Dense(self.compressed_dim_kv, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DKV")
        
        # W_UK:  (num_heads*head_dim, compressed_dim_kv)
        # W_UV:  (num_heads*head_dim, compressed_dim_kv)
        self.W_UK = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UK")
        self.W_UV = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UV")
        
        # W_DQ: (compressed_dim_q, seq_len)
        # W_UQ: (num_heads*head_dim, seq_len)
        self.W_DQ = nn.Dense(self.compressed_dim_q, use_bias=False,
                              kernel_init=xavier_uniform, name="W_DQ")
        self.W_UQ = nn.Dense(self.head_dim * self.num_heads, use_bias=False,
                              kernel_init=xavier_uniform, name="W_UQ")
        
        # W_QR: (num_heads * rope_head_dim, compressed_dim_q)
        # W_KR: (rope_head_dim, seq_len)
        self.W_QR = nn.Dense(self.num_heads * self.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_QR")
        self.W_KR = nn.Dense(self.rope_head_dim, use_bias=False,
                              kernel_init=xavier_uniform, name="W_KR")
        
        self.W_O = nn.Dense(self.hidden_size, use_bias=False,
                            kernel_init=xavier_uniform, name="W_O")
        
    def __call__(self,
                hidden_states: Float[Array, "batch seq_len hidden_size"],
                mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None
            ) -> Float[Array, "batch seq_len hidden_size"]:
        """
        hidden_states: shape (batch_size, seq_len, hidden_dim)
        mask: shape (batch_size, 1, seq_len, seq_len) or None
        """
        
        batch_size, seq_len, hidden_dims = hidden_states.shape
        assert hidden_dims == self.hidden_size, "Input hidden size does not match config"
        
        c_KV = self.W_DKV(hidden_states)
        k_C = self.W_UK(c_KV)
        v_C = self.W_UV(c_KV)
        
        c_Q = self.W_DQ(hidden_states)
        q_C = self.W_UQ(c_Q)
        
        q_R = self.W_QR(c_Q)
        q_R = q_R.reshape(batch_size, seq_len, self.num_heads, self.rope_head_dim)
        
        k_R = self.W_KR(hidden_states)
        k_R = k_R.reshape(batch_size, seq_len, 1, self.rope_head_dim)
        
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
        
        attn_scores = jax.lax.dot_general(
            q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))
        ) / jnp.sqrt(self.head_dim + self.rope_head_dim)
        
        if mask is not None:
            seq_len_total = k.shape[2]
            mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len_total))
            mask = jnp.repeat(mask, self.num_heads, axis=1)
            attn_scores += mask * -1e9
            
        attn_probs = nn.softmax(attn_scores, axis=-1)
        
        attn_output = jax.lax.dot_general(
            attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1)))
        )
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        return self.W_O(attn_output)
        