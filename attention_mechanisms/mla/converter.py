import pickle
from flax.core import freeze, unfreeze, FrozenDict
import jax
import jax.numpy as jnp
from attention_mechanisms.configs.minimax_config import MiniMaxConfig

def _convert_weights(mla_params):
    """
    Convert weights from MLAttention (dense-based) to AutoRegMLAttention (param-based).
    Adjust transpositions where necessary.
    """
    mla_params = unfreeze(mla_params)

    autoreg_params = {
        "W_DQ": mla_params["W_DQ"]["kernel"],  
        "W_DKV": mla_params["W_DKV"]["kernel"], 
        "W_UQ_C": mla_params["W_UQ"]["kernel"], 
        "W_UQ_R": mla_params["W_QR"]["kernel"],
        "W_KR": mla_params["W_KR"]["kernel"],
        "W_UK_C": mla_params["W_UK"]["kernel"],
        "W_UV_C": mla_params["W_UV"]["kernel"],
        "W_O": mla_params["W_O"]["kernel"]  
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
    from attention_mechanisms.mla.mla import MLAttention, AutoRegMLAttention

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

    mla_model = MLAttention(config)
    mla_params = mla_model.init(jax.random.PRNGKey(0), jax.random.normal(jax.random.PRNGKey(0), (1, 1, config.hidden_size)))["params"]

    # with open("initialized_mla_params.pkl", "wb") as f:
    #     pickle.dump(mla_params, f)
        
    autoreg_params = _convert_weights(mla_params)
    
    autoreg_model = AutoRegMLAttention(config)

    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (1, 1, config.hidden_size))

    mla_output = mla_model.apply({"params": mla_params}, hidden_states)

    autoreg_output, cached_cKV, cached_kR = autoreg_model.apply(
        {"params": autoreg_params},
        hidden_states,
        cached_cKV=None,
        cached_kR=None
    )

    if jnp.allclose(mla_output, autoreg_output, atol=1e-5):
        print("Outputs match!")
    else:
        max_diff = jnp.max(jnp.abs(mla_output - autoreg_output))
        print("Outputs differ. Maximum difference:", max_diff)
