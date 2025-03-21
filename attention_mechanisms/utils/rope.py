from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple
import matplotlib.pyplot as plt
from ..configs import BaseConfig



class RotaryPositionEmbedding(nn.Module):
    """
    A Flax module that applies partial Rotary Position Embeddings (RoPE) to
    query and key tensors. By default, only a fraction of each head dimension
    (specified by `rope_fraction`) is rotated, and the remaining dimensions
    remain unaltered.

    Attributes:
        config (MiniMaxConfig): A configuration object specifying RoPE fraction
            and base frequency.
    """
    config: BaseConfig

    def _get_rotary_matrix(
        self,
        seq_len: int,
        rot_dim: int
    ) -> Tuple[Float[Array, "seq rot_dim//2"], Float[Array, "seq rot_dim//2"]]:
        """
        Generate the sine and cosine matrices for the rotary transformations.

        Args:
            seq_len (int): Length of the sequence (number of positions).
            rot_dim (int): Number of head-dimension channels to which
                           RoPE is applied.

        Returns:
            sin (Float[Array, "seq rot_dim//2"]):
                The sine values, shape = [seq_len, rot_dim//2].
            cos (Float[Array, "seq rot_dim//2"]):
                The cosine values, shape = [seq_len, rot_dim//2].
        """
        theta = 1.0 / (self.config.rope_base_freq **
                      (2 * jnp.arange(0, rot_dim // 2) / rot_dim))
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        angles = positions[:, None] * theta[None, :]
        return jnp.sin(angles), jnp.cos(angles)

    @nn.compact
    def __call__(
        self,
        q: Float[Array, "batch seq_len num_heads head_dim"],
        k: Float[Array, "batch seq_len num_heads head_dim"]
    ) -> Tuple[
        Float[Array, "batch seq_len num_heads head_dim"],
        Float[Array, "batch seq_len num_heads head_dim"]
    ]:
        """
        Applies rotary position embeddings to the first `rot_dim` channels
        of query (q) and key (k) tensors. The remaining channels of each head
        dimension are left unrotated.

        Args:
            q (Float[Array, "batch seq_len num_heads head_dim"]):
                The query tensor.
            k (Float[Array, "batch seq_len num_heads head_dim"]):
                The key tensor.

        Returns:
            A tuple (q_rot, k_rot) where:
            - q_rot (Float[Array, "batch seq_len num_heads head_dim"]):
                The query tensor after applying partial RoPE.
            - k_rot (Float[Array, "batch seq_len num_heads head_dim"]):
                The key tensor after applying partial RoPE.
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        rot_dim = int(self.config.rope_fraction * head_dim)

        # Generate sine and cosine for the rotary transformation
        sin, cos = self._get_rotary_matrix(seq_len, rot_dim)

        # Reshape for broadcasting:
        # sin, cos -> [1, seq, 1, rot_dim//2, 1]
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]

        def rotate_tensor(x: Float[Array, "batch seq_len num_heads head_dim"]
                          ) -> Float[Array, "batch seq_len num_heads head_dim"]:
            """
            Rotate the first `rot_dim` channels of x with the RoPE transformation.
            """
            x_rot = x[..., :rot_dim].reshape(*x.shape[:-1], rot_dim // 2, 2)
            x_rot = jnp.stack([
                x_rot[..., 0] * cos - x_rot[..., 1] * sin,
                x_rot[..., 0] * sin + x_rot[..., 1] * cos
            ], axis=-1)

            return jnp.concatenate([
                x_rot.reshape(*x.shape[:-1], rot_dim),
                x[..., rot_dim:]
            ], axis=-1)

        return rotate_tensor(q), rotate_tensor(k)