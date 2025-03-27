import jax
import jax.numpy as jnp
from attention_mechanisms.configs.minimax_config import MiniMaxConfig  # or wherever your BaseConfig is
from attention_mechanisms.mla.mla import MLAttention  # adjust import path to match your project

def test_mla():
    config = MiniMaxConfig(
        num_heads=8,
        head_dim=64,
        hidden_size=512,
        compressed_dim_kv=128,
        compressed_dim_q=192,
        rope_head_dim=32,
        rope_fraction=0.5,      
        rope_base_freq=10000.0, 
    )

    attention = MLAttention(config=config)

    rng = jax.random.PRNGKey(0)
    batch_size, seq_len, hidden_dim = 2, 12, config.hidden_size
    dummy_inputs = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)

    # For causal mask: shape (batch, 1, seq_len, seq_len)
    # Lower-triangular (True = keep, False = block) or vice versa depending on your usage
    causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=bool))

    # 1) Initialize parameters
    params = attention.init(rng, dummy_inputs, mask=causal_mask)
    print("Parameter shapes:")
    for param_name, weight in params["params"].items():
        if isinstance(weight, dict):
            # If weight is a nested dict, loop further
            for w_name, w_value in weight.items():
                print(f"  {param_name}.{w_name}: {w_value.shape}")
        else:
            print(f"  {param_name}: {weight.shape}")

    # 2) Forward pass
    output = attention.apply(params, dummy_inputs, mask=causal_mask)
    print("Output shape:", output.shape)

    # Optionally, add assertions to verify correctness:
    assert output.shape == (batch_size, seq_len, hidden_dim), \
        f"MLA output shape should be {(batch_size, seq_len, hidden_dim)} but got {output.shape}"

    # Possibly check for NaNs or other anomalies
    assert not jnp.isnan(output).any(), "Output contains NaNs, check your attention computations."

if __name__ == "__main__":
    test_mla()