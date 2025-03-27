import pickle
from flax.core import freeze, unfreeze, FrozenDict
import jax
import jax.numpy as jnp
from attention_mechanisms.configs.minimax_config import MiniMaxConfig

def _convert_weights(gqa_params):
    """
    Convert weights from GQAAttention to AutoRegGQAAttention.
    Both share the same parameter structure, so this is a direct mapping.
    """
    gqa_params = unfreeze(gqa_params)
    
    autoreg_params = {
        "q_proj": gqa_params["q_proj"],
        "k_proj": gqa_params["k_proj"],
        "v_proj": gqa_params["v_proj"],
        "out_proj": gqa_params["out_proj"]
    }
    
    return freeze(autoreg_params)

def convert_weights_from_path(path):
    """
    Load weights from a pickle file and convert them for AutoRegGQAAttention.
    
    Args:
        path: Path to the pickle file containing GQAAttention weights
        
    Returns:
        Converted weights for AutoRegGQAAttention
    """
    with open(path, "rb") as f:
        trained_params = pickle.load(f)

    autoreg_params = _convert_weights(trained_params)
    return autoreg_params


def convert_weights(model: FrozenDict):
    """
    Convert weights directly from a model parameter dictionary.
    
    Args:
        model: FrozenDict containing GQAAttention parameters
        
    Returns:
        Converted weights for AutoRegGQAAttention
    """
    autoreg_params = _convert_weights(model)
    return autoreg_params


if __name__ == "__main__":
    from attention_mechanisms.gqa.gqa import GQAAttention, AutoRegGQAAttention
    
    # Example configuration
    config = MiniMaxConfig(
        num_heads=8,
        head_dim=64,
        hidden_size=512,
        group_size=2,  # For GQA, we need a group size > 1
        rope_fraction=1.0,
        rope_base_freq=10000.0
    )
    
    # Initialize the standard attention model
    gqa_model = GQAAttention(config)
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (1, 1, config.hidden_size))
    gqa_params = gqa_model.init(jax.random.PRNGKey(0), hidden_states)["params"]
    
    # Convert the weights
    autoreg_params = _convert_weights(gqa_params)
    
    # Initialize the autoregressive attention model
    autoreg_model = AutoRegGQAAttention(config)
    
    # Test the models to ensure outputs match
    gqa_output = gqa_model.apply({"params": gqa_params}, hidden_states)
    
    autoreg_output, cached_k, cached_v = autoreg_model.apply(
        {"params": autoreg_params},
        hidden_states,
        past_key=None,
        past_value=None
    )
    
    # Verify outputs match
    if jnp.allclose(gqa_output, autoreg_output, atol=1e-5):
        print("Outputs match! Conversion successful.")
    else:
        max_diff = jnp.max(jnp.abs(gqa_output - autoreg_output))
        print(f"Outputs differ. Maximum difference: {max_diff}")
        
    # Save the converted weights if needed
    # with open("converted_autoreg_gqa_params.pkl", "wb") as f:
    #     pickle.dump(autoreg_params, f)