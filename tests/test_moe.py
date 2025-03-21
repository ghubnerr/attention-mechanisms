from attention_mechanisms.utils.moe import ExpertMLP, GlobalRouter, MoEBlock
from attention_mechanisms.configs.minimax_config import MiniMaxConfig
import jax.numpy as jnp
import jax

def test_ffn():
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((2, 5, 6144))
    expert = ExpertMLP(MiniMaxConfig())
    params = expert.init(rng, dummy_input)

    expand_weights = params['params']['expert_expand']['kernel']
    contract_weights = params['params']['expert_contract']['kernel']

    print("\n=== FFN Test ===")
    print(f"Expand layer weights - Mean: {jnp.mean(expand_weights):.4f}")
    print(f"Expand layer weights - Std: {jnp.std(expand_weights):.4f}")
    print(f"Contract layer weights - Mean: {jnp.mean(contract_weights):.4f}")
    print(f"Contract layer weights - Std: {jnp.std(contract_weights):.4f}")

def test_global_router():
    rng = jax.random.PRNGKey(0)
    num_tokens = 10  # batch_size * seq_len
    dummy_input = jax.random.normal(rng, (num_tokens, 6144))

    router = GlobalRouter(MiniMaxConfig())
    params = router.init(rng, dummy_input)
    indices, scores, mask, loss = router.apply(params, dummy_input)

    print("\n=== GlobalRouter Test ===")
    print(f"Expert indices shape: {indices.shape}")
    print(f"Routing scores shape: {scores.shape}")
    print(f"Expert mask shape: {mask.shape}")
    print(f"Aux loss value: {loss:.4f}")
    print(f"Aux loss is scalar: {loss.ndim == 0}")
    
    
def test_moe_block():
    config = MiniMaxConfig(
        ffw_hidden_size=9216,
        num_experts=8,
        top_k=2,
        hidden_size=6144
    )
    batch_size = 4
    seq_len = 512
    
    print("\n=== Creating MoE Block ===")
    print(f"Config: {config}")
    moe_block = MoEBlock(config)
    print("MoE block created with:")
    print(f"- {config.num_experts} experts")
    print(f"- Top-{config.top_k} routing")
    print(f"- Expert hidden size: {config.ffw_hidden_size}")
    
    rng = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(rng, (batch_size, seq_len, config.hidden_size))
    params = moe_block.init(rng, dummy_input)
    
    print("\n=== MoE Block Forward Pass ===")
    
    # Generate test input with random values
    test_input = jax.random.normal(rng, (batch_size, seq_len, config.hidden_size))
    print(f"Input shape: {test_input.shape}")
    print(f"Input stats - Mean: {jnp.mean(test_input):.4f}, Std: {jnp.std(test_input):.4f}")

    # Run forward pass
    output, aux_loss = moe_block.apply(params, test_input)

    print("\n=== Output Verification ===")
    print(f"Output shape: {output.shape}")
    print(f"Output matches input shape: {output.shape == test_input.shape}")
    print(f"Output stats - Mean: {jnp.mean(output):.4f}, Std: {jnp.std(output):.4f}")

    print("\n=== Auxiliary Loss Check ===")
    print(f"Aux loss value: {aux_loss:.4f}")
    print(f"Aux loss is scalar: {aux_loss.ndim == 0}")

    print("\n=== Sanity Checks ===")
    # Verify non-zero output
    print(f"Output is all zeros: {jnp.allclose(output, 0)}")

    # Verify expert diversity
    gate_params = params['params']['router']['router_gate']
    print(f"Gate weights shape: {gate_params['kernel'].shape}")
    print(f"Gate bias shape: {gate_params['bias'].shape}")


if __name__ == "__main__":
    test_ffn()
    test_global_router()