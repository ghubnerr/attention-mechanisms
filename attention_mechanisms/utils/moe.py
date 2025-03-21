from flax import linen as nn
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Bool, Array
from typing import Tuple
from ..configs.base import BaseConfig
from ..utils import xavier_uniform

class ExpertMLP(nn.Module):
    """Implements individual expert FFN with hidden dimension expansion.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden] or [tokens, hidden]

    Returns:
        Transformed tensor with same shape as input
    """
    config: BaseConfig

    @nn.compact
    def __call__(self, x: Float[Array, "... hidden_dims"]) -> Float[Array, "... hidden_dims"]:
        # Expand to intermediate dimension
        x = nn.Dense(self.config.ffw_hidden_size,
                   kernel_init=xavier_uniform,
                   name="expert_expand")(x)
        x = nn.relu(x)
        return nn.Dense(self.config.hidden_size,
                      kernel_init=xavier_uniform,
                      name="expert_contract")(x)
        
        
class GlobalRouter(nn.Module):
    """Handles token-to-expert routing with top-k selection and load balancing.

    Args:
        x: Flattened input tensor of shape [tokens, hidden]

    Returns:
        Tuple of (expert indices, routing scores, expert mask, aux loss)
    """
    config: BaseConfig

    @nn.compact
    def __call__(self, x: Float[Array, "tokens hidden"]
                ) -> Tuple[Int[Array, "tokens top_k"],
                           Float[Array, "tokens top_k"],
                           Bool[Array, "tokens top_k experts"],
                           Float[Array, ""]]:
        gate_logits = nn.Dense(self.config.num_experts,
                               kernel_init=xavier_uniform, name="router_gate")(x)

        # Top-k expert selection
        scores, expert_indices = jax.lax.top_k(gate_logits, self.config.top_k)
        scores = jax.nn.softmax(scores, axis=-1)

        # Create expert assignment mask
        expert_mask = jax.nn.one_hot(expert_indices, self.config.num_experts)

        f_i = jnp.mean(expert_mask, axis=(0, 1))
        m_i = jnp.mean(jax.nn.softmax(gate_logits, axis=-1), axis=0)  # Mean probs
        aux_loss = self.config.aux_loss_coef * jnp.sum(f_i * m_i) / self.config.num_experts

        return expert_indices, scores, expert_mask, aux_loss


class MoEBlock(nn.Module):
    """MoE transformer block implementing token-drop strategy with capacity limits."""
    config: BaseConfig

    def setup(self):
        self.router = GlobalRouter(self.config)
        self.experts = [ExpertMLP(self.config) for _ in range(self.config.num_experts)]

    def __call__(self, x: Float[Array, "batch seq_len hidden"]) -> Tuple[Float[Array, "batch seq_len hidden"], Float[Array, ""]]:
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.reshape(num_tokens, -1)  # [tokens, hidden]

        # Get routing decisions
        expert_indices, scores, expert_mask, aux_loss = self.router(x_flat)

        assert expert_indices.shape == (num_tokens, self.config.top_k), "Expert indices shape mismatch"
        assert scores.shape == (num_tokens, self.config.top_k), "Scores shape mismatch"
        assert expert_mask.shape == (num_tokens, self.config.top_k, self.config.num_experts), "Expert mask shape mismatch"

        output = jnp.zeros_like(x_flat)
        for expert_idx in range(self.config.num_experts):
            output += self._process_expert(
                x_flat,
                expert_idx,
                expert_mask[..., expert_idx],  # [tokens, top_k]
                scores,  # [tokens, top_k]
                num_tokens
            )

        return output.reshape(batch_size, seq_len, -1), aux_loss

    def _process_expert(
        self,
        x: Float[Array, "tokens hidden"],
        expert_idx: int,
        mask: Bool[Array, "tokens top_k"],
        scores: Float[Array, "tokens top_k"],
        num_tokens: int
    ) -> Float[Array, "tokens hidden"]:
        """Process tokens through a single expert with capacity constraints."""
        capacity = max((num_tokens * self.config.top_k) // self.config.num_experts, 1)

        # Select tokens and scores with validation
        tokens, scores_expert = self._select_tokens(x, scores, mask, capacity)
        assert tokens.shape[0] <= capacity, "Token selection exceeds capacity"
        assert scores_expert.shape == (capacity,), "Scores shape mismatch"

        # Process through expert
        expert_out = self.experts[expert_idx](tokens)
        assert expert_out.shape == (capacity, self.config.hidden_size), "Expert output shape mismatch"

        # Scatter outputs with dimension checks
        scattered = self._scatter_outputs(
            expert_out, scores_expert, mask, capacity, num_tokens
        )
        assert scattered.shape == (num_tokens, self.config.hidden_size), "Scattering shape mismatch"

        return scattered

    def _select_tokens(
        self,
        x: Float[Array, "tokens hidden"],
        scores: Float[Array, "tokens top_k"],
        mask: Bool[Array, "tokens top_k"],
        capacity: int
    ) -> Tuple[Float[Array, "capacity hidden"], Float[Array, "capacity"]]:
        """Select tokens for expert processing based on routing scores."""

        # Get raw indices [capacity, 2]
        token_indices = jnp.argwhere(mask, size=capacity, fill_value=-1)

        # Validate indices
        assert token_indices.shape == (capacity, 2), f"Indices shape {token_indices.shape} != ({capacity}, 2)"

        # Extract coordinates
        i_indices = token_indices[..., 0]  # [capacity]
        j_indices = token_indices[..., 1]  # [capacity]
        valid_mask = (i_indices != -1) & (j_indices != -1)

        # Get scores for selected positions
        scores_expert = scores[i_indices, j_indices] * valid_mask

        # Sort by scores (descending) with valid entries first
        sort_order = jnp.argsort(-scores_expert)

        return x[i_indices][sort_order], scores_expert[sort_order]

    def _scatter_outputs(
        self,
        expert_out: Float[Array, "capacity hidden"],
        scores: Float[Array, "capacity"],
        mask: Bool[Array, "tokens top_k"],
        capacity: int,
        num_tokens: int
    ) -> Float[Array, "tokens hidden"]:
        """Distribute expert outputs back to original token positions."""
        # Get token indices (i) from mask [capacity]
        token_indices = jnp.argwhere(mask, size=capacity, fill_value=-1)[..., 0]
        valid = token_indices != -1

        # Calculate weighted outputs
        weighted = expert_out * scores[:, None] * valid[:, None]

        # Validate scattering dimensions
        assert weighted.shape == (capacity, self.config.hidden_size), f"Weighted shape {weighted.shape} mismatch"
        assert token_indices.shape == (capacity,), f"Indices shape {token_indices.shape} mismatch"

        # Aggregate outputs
        return jax.ops.segment_sum(
            weighted,
            token_indices,
            num_segments=num_tokens
        )