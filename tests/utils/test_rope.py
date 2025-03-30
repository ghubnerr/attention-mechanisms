from attention_mechanisms.utils.rope import RotaryPositionEmbedding
from attention_mechanisms.configs.minimax_config import MiniMaxConfig
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

def test_rope():
    os.makedirs("./outputs", exist_ok=True)
    
    config = MiniMaxConfig(
        rope_fraction=0.5,
        rope_base_freq=10000.0
    )

    rope_module = RotaryPositionEmbedding(config=config)

    batch_size = 1
    seq_length = 50
    num_heads = 4
    head_dim = 128


    # Random queries and keys for demonstration
    q = jnp.ones((batch_size, seq_length, num_heads, head_dim))
    k = jnp.ones((batch_size, seq_length, num_heads, head_dim))

    variables = rope_module.init(jax.random.PRNGKey(0), q, k)
    
    q_rot, k_rot = rope_module.apply(variables, q, k)

    # Visualization of the rotated query tensor
    # Taking the first head from the first batch for visualization
    q_rot_np = np.array(q_rot[0, :, 0, :])
    
    plt.figure(figsize=(20, 8))
    plt.imshow(q_rot_np, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Rotary Positional Embeddings (Applied to Query)')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.tight_layout()
    plt.savefig(f'./outputs/rotary_positional_embeddings_test_{datetime.date}.png', dpi=300)
    
    
if __name__ == "__main__":
    test_rope()
    