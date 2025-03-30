<img width="748" alt="image" src="https://github.com/user-attachments/assets/15f87919-d96b-4694-9ff3-2c41b5b954c9" />

# A Survey on State-of-the-Art Attention Mechanisms

This repository presents a comprehensive experiment and empirical analysis of various attention mechanisms in transformer-based language models. While the Multi-Head Self-Attention (MHA) mechanism introduced in the original transformer architecture has been foundational to the success of modern language models, several alternative attention mechanisms have emerged to address its computational challenges. We implement four primary attention variants in **[Flax](https://flax.readthedocs.io/)**: standard Multi-Head Self-Attention (MHA), Group-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-Linear Attention (MLA).

- [MHAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/mha/mha.py#L10), [AutoRegMHAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/mha/mha.py#L70) (w/ KV Caching)
- [GQAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/gqa/gqa.py#L19), [AutoRegGQAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/gqa/gqa.py#L120) (w/ KV Caching)
- [MLAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/mla/mla.py#L10), [AutoRegMLAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/mla/mla.py#L144) (w/ KV Caching)
- [MQAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/mqa/mqa.py#L10), [AutoRegMQAttention](https://github.com/ghubnerr/attention-mechanisms/blob/d86e49cc99f1c9fb557b18e68cd64504286c7b8c/attention_mechanisms/mqa/mqa.py#L80) (w/ KV Caching)

We also add scripts to assist in the convertion between standard and incremental (auto-regressive) attention mechanisms for converting between training and decoder-only (i.e. inference) environments. We also make Mixture of Experts FFW Layers with Top-K routing, and Rotary Position Embedding implementations and a Transformer Block architecture that combines all of the above, normalizing with RMSNorm.

### Directory Structure
```
.
├── attention_mechanisms
│   ├── configs                # Model Configurations 
│   ├── core                   
│   │   ├── converter.py       # Converter from a Standard (training) to an AutoRegressive-based Transformer (inference)
│   │   └── transformer.py     # Transformer Block and Transformer Modules
│   ├── gqa                    # Group-Query Attention and Auto Regressive Group-Query Attention Modules
│   │   ├── converter.py
│   │   ├── gqa.py
│   ├── mha                    # Multi-Head Self-Attention and Auto Regressive Multi-Head Self-Attention Modules
│   │   ├── converter.py
│   │   ├── mha.py
│   ├── mla                    # Multi-Latent Attention and Auto Regressive Multi-Latent Attention Modules
│   │   ├── converter.py
│   │   ├── mla.py
│   ├── mqa                    # Multi-Query Attention and Auto Regressive Multi-Query Attention Modules
│   │   ├── converter.py
│   │   ├── mqa.py
│   └── utils
│       ├── moe.py             # Mixture of Experts with Top-K Routing
│       ├── rms_norm.py        # RMS Normalization
│       └── rope.py            # RoPE (Rotary Positional Embedding)
├── notebooks
└── tests  
    ├── attentions              # Tests: Standard and Incremental Attention Mechanisms
    │   ├── test_gqa.py
    │   ├── test_mha.py
    │   ├── test_mla.py
    │   └── test_mqa.py
    ├── core                    # Tests: Transformers
    │   └── test_transformer.py
    └── utils                   
        ├── test_moe.py         # Tests: Mixture of Experts Block with Routing
        └── test_rope.py        # Tests: Rotatory Positional Encoding
```
### Example: Standard Multi-Latent Attention
<img width="734" alt="image" src="https://github.com/user-attachments/assets/7f173d8b-7eae-4055-bb69-5913b9cccceb" />

```python
class MLAttention(nn.Module):
    config: BaseConfig

    def setup(self):
        self.config.num_heads = self.config.num_heads
        self.config.head_dim = self.config.head_dim
        self.config.hidden_size = self.config.hidden_size
        self.config.compressed_dim_kv = self.config.compressed_dim_kv
        self.config.compressed_dim_q = self.config.compressed_dim_q
        self.config.rope_head_dim = self.config.rope_head_dim
        self.rope = RotaryPositionEmbedding(config=self.config)
        self.scale = 1.0 / jnp.sqrt(self.config.head_dim + self.config.rope_head_dim)

        self.W_DKV = nn.Dense(self.config.compressed_dim_kv, use_bias=False, kernel_init=xavier_uniform, name="W_DKV")
        self.W_UK = nn.Dense(self.config.head_dim * self.config.num_heads, use_bias=False, kernel_init=xavier_uniform, name="W_UK")
        self.W_UV = nn.Dense(self.config.head_dim * self.config.num_heads, use_bias=False, kernel_init=xavier_uniform, name="W_UV")
        self.W_DQ = nn.Dense(self.config.compressed_dim_q, use_bias=False, kernel_init=xavier_uniform, name="W_DQ")
        self.W_UQ = nn.Dense(self.config.head_dim * self.config.num_heads, use_bias=False, kernel_init=xavier_uniform, name="W_UQ")
        self.W_QR = nn.Dense(self.config.num_heads * self.config.rope_head_dim, use_bias=False, kernel_init=xavier_uniform, name="W_QR")
        self.W_KR = nn.Dense(self.config.rope_head_dim, use_bias=False, kernel_init=xavier_uniform, name="W_KR")
        self.W_O = nn.Dense(self.config.hidden_size, use_bias=False, kernel_init=xavier_uniform, name="W_O")

    def __call__(self,
                hidden_states: Float[Array, "batch seq_len hidden_size"],
                mask: Optional[Float[Array, "batch 1 seq_len seq_len"]] = None
            ) -> Float[Array, "batch seq_len hidden_size"]:
        batch_size, seq_len, hidden_dims = hidden_states.shape
        assert hidden_dims == self.config.hidden_size, "Input hidden size does not match config"

        c_KV = self.W_DKV(hidden_states)
        k_C = self.W_UK(c_KV)
        v_C = self.W_UV(c_KV)

        c_Q = self.W_DQ(hidden_states)
        q_C = self.W_UQ(c_Q)

        q_R = self.W_QR(c_Q).reshape(batch_size, seq_len, self.config.num_heads, self.config.rope_head_dim)
        k_R = self.W_KR(hidden_states).reshape(batch_size, seq_len, 1, self.config.rope_head_dim)

        q_R, k_R = self.rope(q_R, k_R)
        k_R = jnp.broadcast_to(k_R, (batch_size, seq_len, self.config.num_heads, self.config.rope_head_dim))

        q_C = q_C.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        k_C = k_C.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        q = jnp.concatenate([q_C, q_R], axis=-1)
        k = jnp.concatenate([k_C, k_R], axis=-1)

        v = v_C.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_scores = jax.lax.dot_general(q, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))) * self.scale

        if mask is not None:
            seq_len_total = k.shape[2]
            mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len_total))
            mask = jnp.repeat(mask, self.config.num_heads, axis=1)
            attn_scores += mask * -1e9

        attn_probs = nn.softmax(attn_scores, axis=-1)

        attn_output = jax.lax.dot_general(attn_probs, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))))

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = attn_output.reshape(batch_size, seq_len, self.config.num_heads * self.config.head_dim)

        return self.W_O(attn_output)
```


### Original Papers
- **Multi-Latent Attention**
```
@article{liu2024deepseek,
  title={Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model},
  author={Liu, Aixin and Feng, Bei and Wang, Bin and Wang, Bingxuan and Liu, Bo and Zhao, Chenggang and Dengr, Chengqi and Ruan, Chong and Dai, Damai and Guo, Daya and others},
  journal={arXiv preprint arXiv:2405.04434},
  year={2024}
}
```
- **Group-Query Attention**
```
@article{ainslie2023gqa,
  title={Gqa: Training generalized multi-query transformer models from multi-head checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and De Jong, Michiel and Zemlyanskiy, Yury and Lebr{\'o}n, Federico and Sanghai, Sumit},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023}
}
```
- **Multi-Query Attention**
```
@article{shazeer2019fast,
  title={Fast transformer decoding: One write-head is all you need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019}
}
```
- **Multi-Head Self-Attention**
```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

#### Tasks:

- [X] Translate Notebook into Repo
- [X] Notebook transformer block
- [ ] Model sharding -> /sharding branch
- [ ] Rename module to attention
- [ ] Make `__init__.py` imports better
- [ ] Dataset
- [ ] Evals
- [ ] Separate into Flax and Torch
- [ ] Append kernels into modules / or separate module
- [ ] Package it to PyPi
