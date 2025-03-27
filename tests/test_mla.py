import jax
import jax.numpy as jnp
from attention_mechanisms.configs.minimax_config import MiniMaxConfig  
from attention_mechanisms.mla.mla import MLAttention, AutoRegMLAttention  

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
    
    causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=bool))

    params = attention.init(rng, dummy_inputs, mask=causal_mask)
    print("Parameter shapes:")
    for param_name, weight in params["params"].items():
        if isinstance(weight, dict):
            for w_name, w_value in weight.items():
                print(f"  {param_name}.{w_name}: {w_value.shape}")
        else:
            print(f"  {param_name}: {weight.shape}")

    output = attention.apply(params, dummy_inputs, mask=causal_mask)
    print("Output shape:", output.shape)

    assert output.shape == (batch_size, seq_len, hidden_dim), \
        f"MLA output shape should be {(batch_size, seq_len, hidden_dim)} but got {output.shape}"

    assert not jnp.isnan(output).any(), "Output contains NaNs, check your attention computations."

def test_autoregmla():
    config = MiniMaxConfig(
        num_heads=8,
        head_dim=64,
        hidden_size=512,
        compressed_dim_kv=128,
        compressed_dim_q=192,
        rope_head_dim=32,
        rope_fraction=0.5,
        rope_base_freq=10000.0
    )

    attention = AutoRegMLAttention(config=config)

    rng = jax.random.PRNGKey(0)
    batch, seq_len, hidden_size = 2, 10, 512

    dummy_inputs = jnp.ones((batch, seq_len, hidden_size))

    params = attention.init(rng, dummy_inputs[:, :1, :])

    cached_c_KV, cached_k_R = None, None

    for i in range(seq_len):
        current_input = dummy_inputs[:, i:i+1, :]
        output, cached_c_KV, cached_k_R = attention.apply(params, current_input, cached_c_KV=cached_c_KV, cached_k_R=cached_k_R)

        print(f"Output shape at step {i}: {output.shape}")
        print(f"Cached c_KV shape at step {i}: {cached_c_KV.shape}")
        print(f"Cached k_R shape at step {i}: {cached_k_R.shape}")
    
if __name__ == "__main__":
    test_mla()
    test_autoregmla()
    
    