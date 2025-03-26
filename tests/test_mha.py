from attention_mechanisms.mha.mha import MHSAttention, AutoRegMHSAttention
from attention_mechanisms.configs.minimax_config import MiniMaxConfig
import jax.numpy as jnp
import jax

def test_mha():
    config = MiniMaxConfig(
        num_heads=64,
        head_dim=128,
        hidden_size=6144,
        rope_fraction=0.5,
        rope_base_freq=10000.0
    )

    attention = MHSAttention(config=config)
    
    rng = jax.random.PRNGKey(0)
    batch, seq_len, hidden_size = 2, 10, 6144
    dummy_inputs = jnp.ones((batch, seq_len, hidden_size))

    mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=bool))

    # Initialize parameters
    params = attention.init(rng, dummy_inputs, mask)
    print("Parameter shapes:")
    for k, v in params["params"].items():
        print(k, jax.tree.map(jnp.shape, v))

    output = attention.apply(params, dummy_inputs, mask=mask)
    print("Output shape:", output.shape)
    
def test_autoregmha():
    config = MiniMaxConfig(
        num_heads=64,
        head_dim=128,
        hidden_size=6144,
        rope_fraction=0.5,
        rope_base_freq=10000.0
    )

    attention = AutoRegMHSAttention(config=config)

    rng = jax.random.PRNGKey(0)
    batch, seq_len, hidden_size = 2, 10, 6144
    dummy_inputs = jnp.ones((batch, seq_len, hidden_size))

    params = attention.init(rng, dummy_inputs)
    print("Parameter shapes (AutoRegMHSAttention):")
    for k, v in params["params"].items():
        print(k, jax.tree.map(jnp.shape, v))

    past_key, past_value = None, None

    for i in range(seq_len):
        current_input = dummy_inputs[:, i:i+1, :]
        output, past_key, past_value = attention.apply(
            params, current_input, past_key=past_key, past_value=past_value
        )
        print(f"Output shape at step {i}: {output.shape}")
        print(f"Past Key shape at step {i}: {past_key.shape}")
        print(f"Past Value shape at step {i}: {past_value.shape}")

    
if __name__ == "__main__":
    test_mha()
    test_autoregmha()