from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm) using Flax.

    RMSNorm normalizes the input across the last dimension based on the
    root mean square of the values in that dimension.

    Attributes:
        epsilon (float): A small constant added for numerical stability to
            avoid division by zero.
    """
    epsilon: float = 1e-6

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch seq_len hidden_size"],
    ) -> Float[Array, "batch seq_len hidden_size"]:
        """
        Forward pass of the RMSNorm layer.

        This normalizes the input tensor along its last dimension, then
        optionally scales it by a learned parameter vector.

        Args:
            x (Float[Array, "batch seq_len hidden_size"]):
                The input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Float[Array, "batch seq_len hidden_size"]:
                The normalized output tensor of the same shape as the input.
        """
        dtype = x.dtype

        # Convert input to float32 for numerical stability in norm calculations
        x = x.astype(jnp.float32)

        g = self.param(
            'scale',
            nn.initializers.ones,
            (x.shape[-1],)
        )

        rms = jnp.sqrt(
            jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.epsilon
        )

        x_norm = x / rms

        # Multiply by the learned scale, then cast back to the original dtype
        return (g * x_norm).astype(dtype)