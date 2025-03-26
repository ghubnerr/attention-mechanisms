from attention_mechanisms.mha.mha import MHSAttention
from attention_mechanisms.configs.minimax_config import MiniMaxConfig
import jax.numpy as jnp
import jax
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils

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

def test_mha_sharded():
    config = MiniMaxConfig(
        num_heads=64,
        head_dim=128,
        hidden_size=6144,
        rope_fraction=0.5,
        rope_base_freq=10000.0
    )
    
    attention_module = MHSAttention(config)

    devices = mesh_utils.create_device_mesh((2,4))
    mesh = Mesh(devices, ('data', 'heads'))

    with mesh:
        @pjit(
            in_shardings=(None, PartitionSpec('data', None, None), None),
            out_shardings=PartitionSpec('data', None, None)
        )
        def sharded_forward(params, hidden_states, mask):
            return attention_module.apply({'params': params}, hidden_states, mask)

        params = attention_module.init(jax.random.PRNGKey(0), jnp.ones((8,128,512)))['params']
        output = sharded_forward(params, jnp.ones((8,128,512)), mask=None)
        print(f"{output.shape=}")
    
if __name__ == "__main__":
    test_mha()
    # test_mha_sharded()