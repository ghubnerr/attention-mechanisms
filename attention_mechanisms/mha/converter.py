import pickle
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
from attention_mechanisms.configs.minimax_config import MiniMaxConfig

def _convert_weights(mhsa_params):
    """
    Convert weights from MHSAttention to AutoRegMHSAttention.
    Both share the same parameter structure, so this is a direct mapping.
    """
    mhsa_params = unfreeze(mhsa_params)
    
    # Since both implementations use the same parameter structure,
    # we can directly map the weights
    autoreg_params = {
        "q_proj": mhsa_params["q_proj"],
        "k_proj": mhsa_params["k_proj"],
        "v_proj": mhsa_params["v_proj"],
        "out_proj": mhsa_params["out_proj"]
    }
    
    return freeze(autoreg_params)

def convert_weights(path):
    """
    Load weights from a pickle file and convert them for AutoRegMHSAttention.
    
    Args:
        path: Path to the pickle file containing MHSAttention weights
        
    Returns:
        Converted weights for AutoRegMHSAttention
    """
    with open(path, "rb") as f:
        trained_params = pickle.load(f)

    autoreg_params = _convert_weights(trained_params)
    return autoreg_params


if __name__ == "__main__":
    from attention_mechanisms.mha.mha import MHSAttention, AutoRegMHSAttention
    
    # Example configuration
    config = MiniMaxConfig(
        num_heads=8,
        head_dim=64,
        hidden_size=512,
        rope_fraction=1.0,
        rope_base_freq=10000.0
    )
    
    # Initialize the standard attention model
    mhsa_model = MHSAttention(config)
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (1, 1, config.hidden_size))
    mhsa_params = mhsa_model.init(jax.random.PRNGKey(0), hidden_states)["params"]
    
    # Convert the weights
    autoreg_params = _convert_weights(mhsa_params)
    
    # Initialize the autoregressive attention model
    autoreg_model = AutoRegMHSAttention(config)
    
    # Test the models to ensure outputs match
    mhsa_output = mhsa_model.apply({"params": mhsa_params}, hidden_states)
    
    autoreg_output, cached_k, cached_v = autoreg_model.apply(
        {"params": autoreg_params},
        hidden_states,
        past_key=None,
        past_value=None
    )
    
    # Verify outputs match
    if jnp.allclose(mhsa_output, autoreg_output, atol=1e-5):
        print("Outputs match! Conversion successful.")
    else:
        max_diff = jnp.max(jnp.abs(mhsa_output - autoreg_output))
        print(f"Outputs differ. Maximum difference: {max_diff}")
