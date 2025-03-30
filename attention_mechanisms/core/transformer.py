from attention_mechanisms.mha.mha import MHSAttention, AutoRegMHSAttention
from attention_mechanisms.gqa.gqa import GQAAttention, AutoRegGQAAttention 
from attention_mechanisms.mqa.mqa import MQAttention, AutoRegMQAttention
from attention_mechanisms.mla.mla import MLAttention, AutoRegMLAttention
from attention_mechanisms.configs.deepseekv2mini_config import BaseConfig
from attention_mechanisms.utils.rms_norm import RMSNorm
from attention_mechanisms.utils.moe import GlobalRouter, ExpertMLP, MoEBlock
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Dict, Any, Optional, Tuple, Union
from flax import linen as nn


class TransformerBlock(nn.Module):
    """
    A configurable Transformer block that can use different attention mechanisms and MoE FFN.
    
    Supports both training and inference modes with standard or autoregressive
    attention variants. Uses RMSNorm for layer normalization.
    
    Attributes:
        config: Configuration containing model hyperparameters
        attn_type: The type of attention mechanism to use (MHSAttention, GQAAttention, etc.)
        autoregressive: Whether to use the autoregressive variant of the attention mechanism
    """
    config: BaseConfig
    attn_type: str = "mha"  # Options: "mha", "gqa", "mqa", "mla"
    autoregressive: bool = False
    
    def setup(self):
        attn_classes = {
            "mha": (MHSAttention, AutoRegMHSAttention),
            "gqa": (GQAAttention, AutoRegGQAAttention),
            "mqa": (MQAttention, AutoRegMQAttention),
            "mla": (MLAttention, AutoRegMLAttention)
        }
        
        if self.attn_type not in attn_classes:
            raise ValueError(f"Unknown attention type: {self.attn_type}. " 
                           f"Available types: {list(attn_classes.keys())}")
        
        attn_class = attn_classes[self.attn_type][1 if self.autoregressive else 0]
        
        self.attn_ln = RMSNorm(epsilon=self.config.rms_norm_epsilon)
        self.attn = attn_class(self.config)
        
        self.ffn_ln = RMSNorm(epsilon=self.config.rms_norm_epsilon)
        self.ffn = MoEBlock(self.config)
    
    def __call__(self, 
                x: Float[Array, "batch seq_len hidden"],
                mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None,
                memory_states: Optional[Float[Array, "batch mem_len hidden"]] = None,
                **kwargs) -> Union[
                    Float[Array, "batch seq_len hidden"],
                    Tuple[Float[Array, "batch seq_len hidden"], Dict[str, Any]]
                ]:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden]
            mask: Optional attention mask
            **kwargs: Additional arguments passed to the attention mechanism
                      (e.g., past_key, past_value for autoregressive attention)
        
        Returns:
            For normal attention: The output tensor
            For autoregressive attention: Tuple of (output tensor, cache dict)
        """
        # Residual connection with pre-layer normalization
        attn_input = self.attn_ln(x)
        
        # Choose memory states or self-attention input
        memory = memory_states if memory_states is not None else attn_input
        
        # Handle different return types for standard vs autoregressive attention
        if self.autoregressive:
            # Handle MQAttention differently as it requires memory states
            if self.attn_type == "mqa":
                attn_output, *cache_values = self.attn(attn_input, memory, mask=mask, **kwargs)
            elif self.attn_type == "mla":
                attn_output, *cache_values = self.attn(attn_input, mask=mask, **kwargs)
            else:  # mha, gqa
                attn_output, *cache_values = self.attn(attn_input, mask=mask, **kwargs)
            
            ffn_input = x + attn_output
            
            ffn_normed = self.ffn_ln(ffn_input)
            ffn_output, aux_loss = self.ffn(ffn_normed)
            
            output = ffn_input + ffn_output
            
            if self.attn_type == "mla":
                cache = {
                    "cached_cKV": cache_values[0],
                    "cached_kR": cache_values[1],
                    "aux_loss": aux_loss
                }
            else:  # mha, gqa, mqa
                cache = {
                    "past_key": cache_values[0],
                    "past_value": cache_values[1],
                    "aux_loss": aux_loss
                }
            
            return output, cache
        else:
            if self.attn_type == "mqa":
                attn_output = self.attn(attn_input, memory, mask=mask)
            else:  # mha, gqa, mla
                attn_output = self.attn(attn_input, mask=mask)
            
            ffn_input = x + attn_output
            
            ffn_normed = self.ffn_ln(ffn_input)
            ffn_output, aux_loss = self.ffn(ffn_normed)
            
            output = ffn_input + ffn_output
            
            return output, aux_loss
        
class TransformerModel(nn.Module):
    """
    Full transformer model with multiple transformer blocks.
    
    Attributes:
        config: Configuration containing model hyperparameters
        num_layers: Number of transformer blocks
        attn_type: The type of attention mechanism to use
        autoregressive: Whether to use the model in autoregressive mode
    """
    config: BaseConfig
    num_layers: int
    attn_type: str = "mha"
    autoregressive: bool = False
    
    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        
        self.layers = [
            TransformerBlock(
                config=self.config,
                attn_type=self.attn_type,
                autoregressive=self.autoregressive
            )
            for _ in range(self.num_layers)
        ]
        
        self.final_ln = RMSNorm(epsilon=self.config.rms_norm_epsilon)
        
        if not self.autoregressive:
            self.lm_head = nn.Dense(
                features=self.config.vocab_size,
                use_bias=False,
                kernel_init=nn.initializers.normal(stddev=0.02)
            )
    
    def __call__(self,
                input_ids: Float[Array, "batch seq_len"],
                mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None,
                memory_ids: Optional[Float[Array, "batch mem_len"]] = None,
                **kwargs) -> Union[
                    Float[Array, "batch seq_len vocab_size"],
                    Tuple[Float[Array, "batch seq_len hidden"], Dict[str, Any]]
                ]:
        """
        Forward pass through the transformer model.
        
        Args:
            input_ids: Token IDs of shape [batch, seq_len]
            mask: Optional attention mask
            memory_ids: Optional memory token IDs for MQAttention
            **kwargs: Additional arguments for autoregressive generation
        
        Returns:
            For normal mode: Logits of shape [batch, seq_len, vocab_size]
            For autoregressive mode: Tuple of (hidden states, cache dict)
        """
        x = self.embed(input_ids)
        
        memory_states = None
        if memory_ids is not None:
            memory_states = self.embed(memory_ids)
        
        total_aux_loss = 0.0
        
        if self.autoregressive:
            cache_dict = {}
            
            for i, layer in enumerate(self.layers):
                layer_prefix = f"layer_{i}"
                layer_kwargs = {k.replace(f"{layer_prefix}_", ""): v 
                              for k, v in kwargs.items() 
                              if k.startswith(layer_prefix)}
                
                x, layer_cache = layer(
                    x, 
                    mask=mask, 
                    memory_states=memory_states, 
                    **layer_kwargs
                )
                
                cache_dict.update({f"{layer_prefix}_{k}": v for k, v in layer_cache.items()})
                total_aux_loss += layer_cache.get("aux_loss", 0.0)
            
            x = self.final_ln(x)
            cache_dict["aux_loss"] = total_aux_loss
            
            return x, cache_dict
        else:
            for layer in self.layers:
                x, aux_loss = layer(
                    x, 
                    mask=mask, 
                    memory_states=memory_states, 
                )
                total_aux_loss += aux_loss
            
            x = self.final_ln(x)
            logits = self.lm_head(x)
            
            return logits, total_aux_loss