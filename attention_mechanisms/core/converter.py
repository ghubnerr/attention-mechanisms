import pickle
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze, FrozenDict
from typing import Dict, Any, Optional, Tuple, Union, Callable

from attention_mechanisms.mha.mha import MHSAttention, AutoRegMHSAttention
from attention_mechanisms.gqa.gqa import GQAAttention, AutoRegGQAAttention
from attention_mechanisms.mqa.mqa import MQAttention, AutoRegMQAttention
from attention_mechanisms.mla.mla import MLAttention, AutoRegMLAttention
from attention_mechanisms.configs.deepseekv2mini_config import DeepSeekV2MiniConfig


def convert_mha_attention(attn_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MHSAttention parameters to AutoRegMHSAttention parameters.
    Both have the same parameter structure, so this is a direct mapping.
    """
    return attn_params

def convert_gqa_attention(attn_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert GQAAttention parameters to AutoRegGQAAttention parameters.
    Both have the same parameter structure, so this is a direct mapping.
    """
    return attn_params

def convert_mqa_attention(attn_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Special converter for MQAttention to AutoRegMQAttention.
    
    This function handles any special considerations for converting MQA parameters.
    The main issue with MQA is ensuring memory states are handled correctly.
    """
    attn_params = unfreeze(attn_params)
    
    autoreg_params = {
        "q_proj": attn_params["q_proj"],
        "k_proj": attn_params["k_proj"],
        "v_proj": attn_params["v_proj"],
        "out_proj": attn_params["out_proj"]
    }
    
    return freeze(autoreg_params)

def convert_mla_attention(attn_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Special converter for MLAttention to AutoRegMLAttention.
    
    This function handles the conversion of parameters for MLAttention to AutoRegMLAttention.
    The parameters need to be restructured to match the expected format.
    """
    attn_params = unfreeze(attn_params)
    
    autoreg_params = {
        "W_DQ": attn_params["W_DQ"]["kernel"],  
        "W_DKV": attn_params["W_DKV"]["kernel"], 
        "W_UQ_C": attn_params["W_UQ"]["kernel"], 
        "W_UQ_R": attn_params["W_QR"]["kernel"],
        "W_KR": attn_params["W_KR"]["kernel"],
        "W_UK_C": attn_params["W_UK"]["kernel"],
        "W_UV_C": attn_params["W_UV"]["kernel"],
        "W_O": attn_params["W_O"]["kernel"]  
    }
    
    return freeze(autoreg_params)

def convert_transformer_model(
    trained_params: FrozenDict,
    attn_type: str,
    num_layers: int,
) -> FrozenDict:
    """
    Convert a trained non-autoregressive transformer model to its autoregressive counterpart.
    
    This function handles the conversion of parameters for different attention mechanisms.
    Each layer's attention parameters are converted using the appropriate converter function.
    
    Args:
        trained_params: FrozenDict containing trained model parameters
        attn_type: The type of attention used ("mha", "gqa", "mqa", or "mla")
        num_layers: Number of transformer layers in the model
        
    Returns:
        FrozenDict containing converted parameters for the autoregressive model
    """
    trained_params = unfreeze(trained_params)
    
    # Select the appropriate attention converter based on the attention type
    if attn_type == "mha":
        attention_converter = convert_mha_attention
    elif attn_type == "gqa":
        attention_converter = convert_gqa_attention
    elif attn_type == "mqa":
        attention_converter = convert_mqa_attention
    elif attn_type == "mla":
        attention_converter = convert_mla_attention
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")
    
    autoreg_params = {}
    
    autoreg_params["embed"] = trained_params["params"]["embed"]
    autoreg_params["final_ln"] = trained_params["params"]["final_ln"]
    
    for i in range(num_layers):
        layer_prefix = f"layers_{i}"
        autoreg_layer_params = {}
        
        # Copy parameters that don't need conversion
        autoreg_layer_params["attn_ln"] = trained_params["params"][layer_prefix]["attn_ln"]
        autoreg_layer_params["ffn_ln"] = trained_params["params"][layer_prefix]["ffn_ln"]
        autoreg_layer_params["ffn"] = trained_params["params"][layer_prefix]["ffn"]
        
        # Convert attention parameters
        attn_params = trained_params["params"][layer_prefix]["attn"]
        autoreg_layer_params["attn"] = attention_converter(attn_params)
        
        autoreg_params[layer_prefix] = autoreg_layer_params
    
    # Copy lm_head parameters if present
    if "lm_head" in trained_params["params"]:
        autoreg_params["lm_head"] = trained_params["params"]["lm_head"]
    
    return {"params": freeze(autoreg_params)}

def verify_conversion(
    config: DeepSeekV2MiniConfig,
    attn_type: str,
    trained_model,
    autoreg_model,
    trained_params: FrozenDict,
    autoreg_params: FrozenDict
) -> bool:
    """Verification that matches the standalone tests more closely"""
    # Generate completely different keys for input and memory
    main_key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(main_key)
    
    if attn_type == "mla":
        mla_model = MLAttention(config)
        autoreg_mla_model = AutoRegMLAttention(config)
        
        hidden_states = jax.random.normal(key1, (1, 1, config.hidden_size))
        
        layer_key = next((k for k in trained_params["params"].keys() if k.startswith("layers_")), None)
        mla_params = trained_params["params"][layer_key]["attn"]
        autoreg_mla_params = autoreg_params["params"][layer_key]["attn"]
        
        try:
            mla_output = mla_model.apply({"params": mla_params}, hidden_states)
            autoreg_output, _, _ = autoreg_mla_model.apply(
                {"params": autoreg_mla_params}, 
                hidden_states,
                cached_cKV=None, 
                cached_kR=None
            )
            
            result = jnp.allclose(mla_output, autoreg_output, atol=1e-4)
            max_diff = jnp.max(jnp.abs(mla_output - autoreg_output))
            
            print(f"Output comparison for {attn_type}:")
            print(f"  Output shape 1: {mla_output.shape}")
            print(f"  Output shape 2: {autoreg_output.shape}")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Outputs {'match' if result else 'differ'}")
            
            return result
            
        except Exception as e:
            print(f"Error testing {attn_type} conversion:")
            print(f"  {type(e).__name__}: {str(e)}")
            
            # Check parameter structure
            print("\nParameter structure:")
            if isinstance(autoreg_mla_params, dict):
                for key, value in autoreg_mla_params.items():
                    if isinstance(value, dict) and "array" in value:
                        print(f"  {key}: CORRECT - has 'array' key")
                    else:
                        print(f"  {key}: INCORRECT - missing 'array' key")
            
            return False
            
    elif attn_type == "mqa":
        # Test individual attention layer
        mqa_model = MQAttention(config)
        autoreg_mqa_model = AutoRegMQAttention(config)
        
        # Create input and memory as tensors
        hidden_states = jax.random.normal(key1, (1, 1, config.hidden_size))
        memory_states = jax.random.normal(key2, (1, 1, config.hidden_size))
        
        # Extract layer 0 attn params
        layer_key = next((k for k in trained_params["params"].keys() if k.startswith("layers_")), None)
        mqa_params = trained_params["params"][layer_key]["attn"]
        autoreg_mqa_params = autoreg_params["params"][layer_key]["attn"]
        
        # Direct comparison between attention outputs
        try:
            mqa_output = mqa_model.apply({"params": mqa_params}, hidden_states, memory_states)
            autoreg_output, _, _ = autoreg_mqa_model.apply(
                {"params": autoreg_mqa_params}, 
                hidden_states,
                memory_states,
                past_key=None, 
                past_value=None
            )
            
            # Compare outputs
            result = jnp.allclose(mqa_output, autoreg_output, atol=1e-4)
            max_diff = jnp.max(jnp.abs(mqa_output - autoreg_output))
            
            print(f"Output comparison for {attn_type}:")
            print(f"  Output shape 1: {mqa_output.shape}")
            print(f"  Output shape 2: {autoreg_output.shape}")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Outputs {'match' if result else 'differ'}")
            
            return result
            
        except Exception as e:
            print(f"Error testing {attn_type} conversion:")
            print(f"  {type(e).__name__}: {str(e)}")
            return False
            
    else:
        input_ids = jax.random.randint(key1, (1, 1), 0, config.vocab_size)
        
        class HiddenStateModel(trained_model.__class__):
            def __call__(self, input_ids, **kwargs):
                x = self.embed(input_ids)
                
                total_aux_loss = 0.0
                
                for layer in self.layers:
                    x, aux_loss = layer(x, **kwargs)
                    total_aux_loss += aux_loss
                
                x = self.final_ln(x)
                
                return x, total_aux_loss
        
        hidden_model = HiddenStateModel(
            config=config,
            num_layers=trained_model.num_layers,
            attn_type=attn_type,
            autoregressive=False
        )
        
        trained_hidden, _ = hidden_model.apply(
            trained_params, 
            input_ids,
            deterministic=True
        )
        
        autoreg_output, _ = autoreg_model.apply(
            autoreg_params, 
            input_ids,
            deterministic=True
        )
        
        result = jnp.allclose(trained_hidden, autoreg_output, atol=1e-4)
        max_diff = jnp.max(jnp.abs(trained_hidden - autoreg_output))
        
        print(f"Output comparison for {attn_type}:")
        print(f"  Output shape 1: {trained_hidden.shape}")
        print(f"  Output shape 2: {autoreg_output.shape}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Outputs {'match' if result else 'differ'}")
        
        return result

def save_converted_model(autoreg_params: FrozenDict, output_path: str):
    """
    Save the converted autoregressive model parameters to a file.
    
    Args:
        autoreg_params: The converted autoregressive model parameters
        output_path: Path to save the parameters
    """
    with open(output_path, "wb") as f:
        pickle.dump(autoreg_params, f)
    print(f"Converted model saved to {output_path}")

def load_and_convert_model(
    input_path: str, 
    attn_type: str, 
    num_layers: int, 
    output_path: Optional[str] = None
) -> FrozenDict:
    """
    Load a trained model, convert it, and optionally save the converted model.
    
    Args:
        input_path: Path to the trained model parameters
        attn_type: The type of attention used
        num_layers: Number of transformer layers in the model
        output_path: Optional path to save the converted parameters
        
    Returns:
        FrozenDict containing converted parameters for the autoregressive model
    """
    # Load the trained model
    with open(input_path, "rb") as f:
        trained_params = pickle.load(f)
    
    # Convert the model
    autoreg_params = convert_transformer_model(trained_params, attn_type, num_layers)
    
    # Save the converted model if requested
    if output_path is not None:
        save_converted_model(autoreg_params, output_path)
    
    return autoreg_params

def create_train_to_inference_pipeline(
    config: DeepSeekV2MiniConfig,
    attn_type: str,
    num_layers: int,
    train_model_class,
    inference_model_class
):
    """
    Create a pipeline for training a model and converting it for inference.
    
    Args:
        config: Configuration containing model hyperparameters
        attn_type: The type of attention to use
        num_layers: Number of transformer layers in the model
        train_model_class: The model class for training (non-autoregressive)
        inference_model_class: The model class for inference (autoregressive)
        
    Returns:
        A tuple of (train_model, inference_model, conversion_function)
    """
    train_model = train_model_class(
        config=config,
        num_layers=num_layers,
        attn_type=attn_type,
        autoregressive=False
    )
    
    inference_model = inference_model_class(
        config=config,
        num_layers=num_layers,
        attn_type=attn_type,
        autoregressive=True
    )
    
    def convert_trained_to_inference(trained_params: FrozenDict) -> FrozenDict:
        """
        Convert the trained model parameters to inference model parameters.
        
        Args:
            trained_params: Parameters of the trained model
            
        Returns:
            Parameters for the inference model
        """
        return convert_transformer_model(trained_params, attn_type, num_layers)
    
    return train_model, inference_model, convert_trained_to_inference


if __name__ == "__main__":
    from attention_mechanisms.configs.deepseekv2mini_config import DeepSeekV2MiniConfig
    from attention_mechanisms.core.transformer import TransformerModel
    
    # Example configuration
    config = DeepSeekV2MiniConfig(
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        vocab_size=32000,
        max_seq_len=1024,
        group_size=2,
        compressed_dim_kv=192,
        compressed_dim_q=256,
        rope_head_dim=32,
        rope_base_freq=10000.0,
        rope_fraction=1.0,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.01,
        ffw_hidden_size=2048,
        rms_norm_epsilon=1e-6,
    )
    
    num_layers = 2
    
    attention_types = ["mha", "gqa", "mqa", "mla"]
    
    for attn_type in attention_types:
        print(f"\nTesting conversion for {attn_type.upper()} attention...")
        
        train_model, inference_model, convert_func = create_train_to_inference_pipeline(
            config=config,
            attn_type=attn_type,
            num_layers=num_layers,
            train_model_class=TransformerModel,
            inference_model_class=TransformerModel
        )
        
        key = jax.random.PRNGKey(0)
        input_shape = (1, 8)  
        input_ids = jax.random.randint(key, input_shape, 0, config.vocab_size)
        
        memory_ids = None
        if attn_type == "mqa":
            memory_ids = jax.random.randint(key, (1, 16), 0, config.vocab_size)
        
        trained_vars = train_model.init(
            key, 
            input_ids,
            memory_ids=memory_ids
        )
        
        inference_vars = convert_func(trained_vars)
        
        verify_conversion(
            config=config,
            attn_type=attn_type,
            trained_model=train_model,
            autoreg_model=inference_model,
            trained_params=trained_vars,
            autoreg_params=inference_vars
        )
        
        # Example of saving and loading (commented out)
        # save_converted_model(inference_vars, f"autoreg_{attn_type}_model.pkl")
        # load_and_convert_model("trained_model.pkl", attn_type, num_layers, f"autoreg_{attn_type}_model.pkl")
