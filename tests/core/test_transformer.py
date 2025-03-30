import jax
import jax.numpy as jnp
from attention_mechanisms.configs.deepseekv2mini_config import DeepSeekV2MiniConfig
from attention_mechanisms.core.transformer import TransformerBlock, TransformerModel

# Helper function to recursively print parameter shapes
def print_param_shapes(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict):
            print_param_shapes(value, prefix=f"{prefix}{key}.")
        else:
            print(f"{prefix}{key}: {value.shape}")

def create_test_config():
    """Create a standard test configuration"""
    return DeepSeekV2MiniConfig(
        num_heads=8,
        head_dim=64,
        hidden_size=512,
        compressed_dim_kv=128,
        compressed_dim_q=192,
        rope_head_dim=32,
        rope_fraction=0.5,
        rope_base_freq=10000.0,
        group_size=2,  # for GQA
        num_experts=4,
        top_k=2,
        ffw_hidden_size=1024,
        rms_norm_epsilon=1e-6,
        vocab_size=32000,
        max_seq_len=512
    )

# Test standard transformer blocks with different attention mechanisms
def test_standard_transformer_blocks():
    attention_types = ["mha", "gqa", "mqa", "mla"]
    
    for attn_type in attention_types:
        config = create_test_config()
        print(f"\nTesting standard transformer block with {attn_type.upper()} attention:")
        
        transformer_block = TransformerBlock(config=config, attn_type=attn_type, autoregressive=False)
        rng = jax.random.PRNGKey(0)
        batch_size, seq_len, hidden_dim = 2, 12, config.hidden_size
        dummy_inputs = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
        
        # For MQA, we need memory states
        memory_states = None
        if attn_type == "mqa":
            memory_states = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
        
        causal_mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool))
        params = transformer_block.init(rng, dummy_inputs, mask=causal_mask, memory_states=memory_states)
        
        print(f"  Number of parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params['params']))}")
        
        output, aux_loss = transformer_block.apply(
            params, dummy_inputs, mask=causal_mask, memory_states=memory_states
        )
        
        print(f"  Output shape: {output.shape}")
        print(f"  Auxiliary loss: {aux_loss}")
        
        assert output.shape == (batch_size, seq_len, hidden_dim), \
            f"Output shape should be {(batch_size, seq_len, hidden_dim)} but got {output.shape}"
        assert not jnp.isnan(output).any(), "Output contains NaNs, check your computations."
        
        print(f"  {attn_type.upper()} standard test passed!")

# Test autoregressive transformer blocks with different attention mechanisms
def test_autoregressive_transformer_blocks():
    attention_types = ["mha", "gqa", "mqa", "mla"]
    
    for attn_type in attention_types:
        config = create_test_config()
        print(f"\nTesting autoregressive transformer block with {attn_type.upper()} attention:")
        
        transformer_block = TransformerBlock(config=config, attn_type=attn_type, autoregressive=True)
        rng = jax.random.PRNGKey(0)
        batch, seq_len, hidden_size = 2, 5, config.hidden_size
        dummy_inputs = jnp.ones((batch, seq_len, hidden_size))
        
        # For MQA, we need memory states
        memory_states = None
        if attn_type == "mqa":
            memory_states = jnp.ones((batch, seq_len, hidden_size))
        
        # Initialize the parameters with the first token
        params = transformer_block.init(rng, dummy_inputs[:, :1, :], memory_states=memory_states[:, :1, :] if memory_states is not None else None)
        
        # Initialize cache variables based on attention type
        if attn_type == "mla":
            cached_states = {"cached_cKV": None, "cached_kR": None}
        else:  # mha, gqa, mqa
            cached_states = {"past_key": None, "past_value": None}
        
        for i in range(seq_len):
            current_input = dummy_inputs[:, i:i+1, :]
            current_memory = memory_states[:, i:i+1, :] if memory_states is not None else None
            
            # Prepare kwargs based on attention type and current cache
            kwargs = {}
            if attn_type == "mla":
                if cached_states["cached_cKV"] is not None:
                    kwargs["cached_cKV"] = cached_states["cached_cKV"]
                    kwargs["cached_kR"] = cached_states["cached_kR"]
            else:  # mha, gqa, mqa
                if cached_states["past_key"] is not None:
                    kwargs["past_key"] = cached_states["past_key"]
                    kwargs["past_value"] = cached_states["past_value"]
            
            # Apply the transformer block
            output, cache = transformer_block.apply(
                params, current_input, memory_states=current_memory, **kwargs
            )
            
            # Update cache
            if attn_type == "mla":
                cached_states["cached_cKV"] = cache["cached_cKV"]
                cached_states["cached_kR"] = cache["cached_kR"]
                print(f"  Step {i} - Output: {output.shape}, cached_cKV: {cached_states['cached_cKV'].shape}, cached_kR: {cached_states['cached_kR'].shape}")
            else:  # mha, gqa, mqa
                cached_states["past_key"] = cache["past_key"]
                cached_states["past_value"] = cache["past_value"]
                print(f"  Step {i} - Output: {output.shape}, past_key: {cached_states['past_key'].shape}, past_value: {cached_states['past_value'].shape}")
        
        print(f"  {attn_type.upper()} autoregressive test passed!")

# Test standard transformer models with different attention mechanisms
def test_standard_transformer_models():
    attention_types = ["mha", "gqa", "mqa", "mla"]
    
    for attn_type in attention_types:
        config = create_test_config()
        print(f"\nTesting standard transformer model with {attn_type.upper()} attention:")
        
        num_layers = 2
        model = TransformerModel(config=config, num_layers=num_layers, attn_type=attn_type, autoregressive=False)
        rng = jax.random.PRNGKey(0)
        batch_size, seq_len = 2, 12
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        # For MQA, we need memory ids
        memory_ids = None
        if attn_type == "mqa":
            memory_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        causal_mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool))
        params = model.init(rng, input_ids, mask=causal_mask, memory_ids=memory_ids)
        
        print(f"  Number of parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params['params']))}")
        
        logits, aux_loss = model.apply(
            params, input_ids, mask=causal_mask, memory_ids=memory_ids
        )
        
        print(f"  Logits shape: {logits.shape}")
        print(f"  Auxiliary loss: {aux_loss}")
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size), \
            f"Logits shape should be {(batch_size, seq_len, config.vocab_size)} but got {logits.shape}"
        assert not jnp.isnan(logits).any(), "Logits contain NaNs, check your computations."
        
        print(f"  {attn_type.upper()} standard model test passed!")

# Test autoregressive transformer models with different attention mechanisms
def test_autoregressive_transformer_models():
    attention_types = ["mha", "gqa", "mqa", "mla"]
    
    for attn_type in attention_types:
        config = create_test_config()
        print(f"\nTesting autoregressive transformer model with {attn_type.upper()} attention:")
        
        num_layers = 2
        model = TransformerModel(config=config, num_layers=num_layers, attn_type=attn_type, autoregressive=True)
        rng = jax.random.PRNGKey(0)
        batch, seq_len = 2, 5
        input_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
        
        # For MQA, we need memory ids
        memory_ids = None
        if attn_type == "mqa":
            memory_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
        
        # Initialize the parameters with the first token
        params = model.init(
            rng, 
            input_ids[:, :1], 
            memory_ids=memory_ids[:, :1] if memory_ids is not None else None
        )
        
        # Initialize layer caches based on attention type
        layer_caches = {}
        for layer_idx in range(num_layers):
            if attn_type == "mla":
                layer_caches[layer_idx] = {"cached_cKV": None, "cached_kR": None}
            else:  # mha, gqa, mqa
                layer_caches[layer_idx] = {"past_key": None, "past_value": None}
        
        for i in range(seq_len):
            current_input = input_ids[:, i:i+1]
            current_memory = memory_ids[:, i:i+1] if memory_ids is not None else None
            
            # Prepare kwargs based on attention type and current cache
            kwargs = {}
            for layer_idx in range(num_layers):
                layer_prefix = f"layer_{layer_idx}"
                if attn_type == "mla":
                    if layer_caches[layer_idx]["cached_cKV"] is not None:
                        kwargs[f"{layer_prefix}_cached_cKV"] = layer_caches[layer_idx]["cached_cKV"]
                        kwargs[f"{layer_prefix}_cached_kR"] = layer_caches[layer_idx]["cached_kR"]
                else:  # mha, gqa, mqa
                    if layer_caches[layer_idx]["past_key"] is not None:
                        kwargs[f"{layer_prefix}_past_key"] = layer_caches[layer_idx]["past_key"]
                        kwargs[f"{layer_prefix}_past_value"] = layer_caches[layer_idx]["past_value"]
            
            # Apply the model
            hidden_states, cache = model.apply(
                params, current_input, memory_ids=current_memory, **kwargs
            )
            
            # Update caches
            for layer_idx in range(num_layers):
                layer_prefix = f"layer_{layer_idx}"
                if attn_type == "mla":
                    layer_caches[layer_idx]["cached_cKV"] = cache[f"{layer_prefix}_cached_cKV"]
                    layer_caches[layer_idx]["cached_kR"] = cache[f"{layer_prefix}_cached_kR"]
                else:  # mha, gqa, mqa
                    layer_caches[layer_idx]["past_key"] = cache[f"{layer_prefix}_past_key"]
                    layer_caches[layer_idx]["past_value"] = cache[f"{layer_prefix}_past_value"]
            
            print(f"  Step {i} - Hidden states shape: {hidden_states.shape}")
            
            # Print the first step's cache shapes
            if i == 0:
                for layer_idx in range(num_layers):
                    if attn_type == "mla":
                        print(f"    Layer {layer_idx} cached_cKV shape: {layer_caches[layer_idx]['cached_cKV'].shape}")
                        print(f"    Layer {layer_idx} cached_kR shape: {layer_caches[layer_idx]['cached_kR'].shape}")
                    else:  # mha, gqa, mqa
                        print(f"    Layer {layer_idx} past_key shape: {layer_caches[layer_idx]['past_key'].shape}")
                        print(f"    Layer {layer_idx} past_value shape: {layer_caches[layer_idx]['past_value'].shape}")
        
        print(f"  {attn_type.upper()} autoregressive model test passed!")

if __name__ == "__main__":
    # Test transformer blocks
    test_standard_transformer_blocks()
    test_autoregressive_transformer_blocks()
    
    # Test transformer models
    test_standard_transformer_models()
    test_autoregressive_transformer_models()