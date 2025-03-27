import pickle
from flax.core import freeze, unfreeze, FrozenDict
import jax
import jax.numpy as jnp
from attention_mechanisms.configs.minimax_config import MiniMaxConfig

def _convert_weights(mqa_params):
    """
    Convert weights from MQAttention to AutoRegMQAttention.
    Both share the same parameter structure, so this is a direct mapping.
    """
    mqa_params = unfreeze(mqa_params)
    
    autoreg_params = {
        "q_proj": mqa_params["q_proj"],
        "k_proj": mqa_params["k_proj"],
        "v_proj": mqa_params["v_proj"],
        "out_proj": mqa_params["out_proj"]
    }
    
    return freeze(autoreg_params)

def convert_weights_from_path(path):
    with open(path, "rb") as f:
        trained_params = pickle.load(f)

    autoreg_params = _convert_weights(trained_params)
    return autoreg_params


def convert_weights(model: FrozenDict):

    autoreg_params = _convert_weights(model)
    return autoreg_params


if __name__ == "__main__":
    from attention_mechanisms.mqa.mqa import MQAttention, AutoRegMQAttention
    
    # Example configuration
    config = MiniMaxConfig(
        num_heads=8,
        head_dim=64,
        hidden_size=512,
        rope_fraction=1.0,
        rope_base_freq=10000.0
    )
    
    mqa_model = MQAttention(config)
    batch_size = 1
    seq_len = 1
    mem_len = 1
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, config.hidden_size))
    memory_states = jax.random.normal(jax.random.PRNGKey(1), (batch_size, mem_len, config.hidden_size))
    
    mqa_params = mqa_model.init(jax.random.PRNGKey(0), hidden_states, memory_states)["params"]
    
    autoreg_params = _convert_weights(mqa_params)
    
    autoreg_model = AutoRegMQAttention(config)
    
    mqa_output = mqa_model.apply({"params": mqa_params}, hidden_states, memory_states)
    
    autoreg_output, cached_k, cached_v = autoreg_model.apply(
        {"params": autoreg_params},
        hidden_states,
        memory_states,
        past_key=None,
        past_value=None
    )
    
    # Verify outputs match
    if jnp.allclose(mqa_output, autoreg_output, atol=1e-5):
        print("Outputs match! Conversion successful.")
    else:
        max_diff = jnp.max(jnp.abs(mqa_output - autoreg_output))
        print(f"Outputs differ. Maximum difference: {max_diff}")
        
    # Save the converted weights if needed
    # with open("converted_autoreg_mqa_params.pkl", "wb") as f:
    #     pickle.dump(autoreg_params, f)