import argparse
import math
from functools import partial
from pathlib import Path
import os

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.jax_utils as flax_utils
from flax.training import checkpoints
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm
import wandb

from attention_mechanisms.core.transformer import TransformerModel
from attention_mechanisms.configs.gpt2_config import GPT2ModelConfig
from attention_mechanisms.data import load_dataset, shard_batch, batch_dataset


@partial(jax.pmap,
         axis_name="batch",
         static_broadcasted_argnums=(3, 4))  # clip_norm, model_config_static
def train_step(state, batch, rng, clip_norm, model_config_static: GPT2ModelConfig):
    dropout_key = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    seq_len = batch['input_ids'].shape[-1]
    causal_mask = nn.make_causal_mask(
        jnp.ones((1, seq_len), dtype="bool"), dtype="bool")

    def loss_fn(params):
        current_memory_ids = batch.get('memory_ids')

        x = batch["input_ids"]
        logits, aux_loss_moe = state.apply_fn(
            {'params': params},
            x,
            mask=causal_mask,
            memory_ids=current_memory_ids,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        # Logits: (num_devices, per_device_batch, seq_len, vocab_size)
        # Labels: (num_devices, per_device_batch, seq_len)
        main_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :], batch['input_ids'][:, 1:]
        ))

        total_loss = main_loss
        # Add MoE auxiliary loss if MoE is active in the config
        if model_config_static.num_experts and model_config_static.num_experts > 0 and \
           model_config_static.top_k and model_config_static.top_k > 0:
            total_loss += aux_loss_moe * model_config_static.aux_loss_coef

        return total_loss, (main_loss, aux_loss_moe)

    (loss_val, (main_loss_val, aux_loss_moe_val)), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)

    # Synchronize gradients and losses across devices
    averaged_grads = jax.lax.pmean(grads, axis_name='batch')
    averaged_total_loss = jax.lax.pmean(loss_val, axis_name='batch')
    averaged_main_loss = jax.lax.pmean(main_loss_val, axis_name='batch')
    averaged_aux_loss_moe = jax.lax.pmean(aux_loss_moe_val, axis_name='batch')

    # Calculate gradient norms (before potential clipping by optimizer chain)
    raw_grad_norm = optax.global_norm(averaged_grads)

    # Emulate what Adam receives after grad clipping for metrics
    # The actual clipping is part of the optax.chain
    clipper = optax.clip_by_global_norm(clip_norm)
    clipped_grads_for_metric, _ = clipper.update(
        averaged_grads, optax.EmptyState(), None)
    clipped_grad_norm_for_metric = optax.global_norm(clipped_grads_for_metric)

    new_state = state.apply_gradients(grads=averaged_grads)

    metrics = {
        "train_loss_total": averaged_total_loss,
        "train_loss_main": averaged_main_loss,
        "train_loss_aux_moe": averaged_aux_loss_moe if (model_config_static.num_experts and model_config_static.num_experts > 0) else jnp.array(0.0),
        "grad_norm_raw": raw_grad_norm,
        "grad_norm_clipped_metric": clipped_grad_norm_for_metric
    }
    return new_state, metrics


def build_scheduler(total_steps, config):  # Use wandb config for scheduler params
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=config.peak_lr,
        transition_steps=config.warmup_steps,
    )
    plateau_schedule = optax.constant_schedule(value=config.peak_lr)

    # Ensure decay_steps is not negative
    decay_steps = total_steps - config.warmup_steps - config.plateau_steps
    decay_steps = max(0, decay_steps)  # Prevent negative decay_steps

    if decay_steps > 0:
        decay_schedule = optax.cosine_decay_schedule(
            init_value=config.peak_lr,
            decay_steps=decay_steps,
            alpha=(config.peak_lr * config.end_lr_factor) /
            config.peak_lr,
        )
        return optax.join_schedules(
            schedules=[warmup_schedule, plateau_schedule, decay_schedule],
            boundaries=[config.warmup_steps,
                        config.warmup_steps + config.plateau_steps],
        )
    elif config.warmup_steps + config.plateau_steps < total_steps:  # Only warmup and plateau
        return optax.join_schedules(
            schedules=[warmup_schedule, plateau_schedule],
            boundaries=[config.warmup_steps],
        )
    else:  # Only warmup if total_steps is very small
        return warmup_schedule


def main():
    parser = argparse.ArgumentParser(
        description="JAX/Flax Attention Mechanisms Pre-training Script")
    # Required
    parser.add_argument("--checkpoint-root-path", type=Path, required=True,
                        help="Root directory to store msgpack checkpoints (subdirs for each attn type)")
    parser.add_argument("--resume", action="store_true", required=False,
                        help="Resume training from latest checkpoint for each attention type")
    # With defaults
    parser.add_argument("--dataset-path", type=str, required=False, default=None,
                        help="Dataset path (e.g., path for load_from_disk or custom loading)")
    parser.add_argument("--num-chkpts", type=int, default=5,
                        help="Number of checkpoints to save during training for each attention type")
    parser.add_argument("--project-name", type=str, default="attention-mechanisms-eval",
                        help="Weights & Biases project name")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size-per-device", type=int,
                        default=4, help="Batch size per device")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--peak-lr", type=float,
                        default=5e-5, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int,
                        default=2000, help="Warmup steps for LR scheduler")
    parser.add_argument("--plateau-steps", type=int,
                        default=1000, help="Plateau steps for LR scheduler")
    parser.add_argument("--weight-decay", type=float,
                        default=0.01, help="AdamW weight decay")
    parser.add_argument("--max-norm", type=float, default=2.0,
                        help="Gradient clipping global norm")

    args = parser.parse_args()
    args.checkpoint_root_path.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_path)
    print(f"Dataset loaded. Number of examples: {len(dataset)}")

    # attention_types_to_eval = ["mha", "gqa", "mqa", "mla"]
    attention_types_to_eval = ["mha"]  # For quicker testing

    for current_attn_type in attention_types_to_eval:
        print(
            f"\n\n{'='*20} Training with Attention Type: {current_attn_type.upper()} {'='*20}")

        wandb.login()
        run_name = f"attn_{current_attn_type}_bs{args.batch_size_per_device*jax.device_count()}_lr{args.peak_lr}"
        wandb.init(
            project=args.project_name,
            name=run_name,
            config=vars(args),  # Log all command-line args
            reinit=True  # Important for multiple wandb.init calls in one script
        )
        # Add model-specific W&B config here if needed
        wandb.config.update({"attention_type": current_attn_type})

        # Define W&B metrics
        wandb.define_metric("examples_seen")
        wandb.define_metric("train_loss_total", step_metric="examples_seen")
        wandb.define_metric("train_loss_main", step_metric="examples_seen")
        wandb.define_metric("train_loss_aux_moe", step_metric="examples_seen")
        wandb.define_metric("perplexity_main", step_metric="examples_seen")
        wandb.define_metric("grad_norm_raw", step_metric="examples_seen")
        wandb.define_metric("grad_norm_clipped_metric",
                            step_metric="examples_seen")
        wandb.define_metric("learning_rate", step_metric="examples_seen")
        config = wandb.config

        transformer_config_obj = GPT2ModelConfig()
        model_hyperparams = {
            **transformer_config_obj.to_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size_per_device * jax.device_count(),
            "accum_steps": args.accum_steps,
            "max_norm": args.max_norm,
            "peak_lr": args.peak_lr,
            "warmup_steps": args.warmup_steps,
            "plateau_steps": args.plateau_steps,
            "weight_decay": args.weight_decay,
            "adam_eps": 1e-8,
            "adam_b1": 0.9,
            "adam_b2": 0.95,
            "end_lr_factor": 0.1,
        }
        wandb.config.update(model_hyperparams, allow_val_change=True)

        model = TransformerModel(
            config=transformer_config_obj,
            num_layers=transformer_config_obj.num_layers,
            attn_type=current_attn_type,
            autoregressive=False
        )

        # --- Optimizer and TrainState ---
        effective_batch_size = config.batch_size_per_device * \
            jax.device_count() * config.accum_steps
        steps_per_epoch = math.ceil(len(dataset) / effective_batch_size)
        total_steps = config.epochs * steps_per_epoch
        print(f"Total training steps for {current_attn_type}: {total_steps}")

        lr_schedule = build_scheduler(total_steps, config)

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_norm),
            optax.MultiSteps(
                optax.adamw(
                    learning_rate=lr_schedule,
                    b1=config.adam_b1,
                    b2=config.adam_b2,
                    eps=config.adam_eps,
                    weight_decay=config.weight_decay,
                ),
                every_k_schedule=config.accum_steps,
            )
        )

        # Use a fixed seed for reproducibility across attention types
        rng_key = jax.random.PRNGKey(0)
        init_rng, training_rng = jax.random.split(rng_key)

        dummy_input_ids = jnp.zeros(
            (1, transformer_config_obj.max_seq_len), dtype=jnp.int32)
        dummy_memory_ids_init = None
        if current_attn_type == "mqa":
            dummy_memory_ids_init = jnp.zeros(
                (1, transformer_config_obj.max_seq_len), dtype=jnp.int32)

        params = model.init(init_rng, input_ids=dummy_input_ids,
                            memory_ids=dummy_memory_ids_init)["params"]
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        current_checkpoint_path = args.checkpoint_root_path / current_attn_type
        current_checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_prefix = f"attn_{current_attn_type}_checkpoint_"

        if args.resume:
            try:
                state = checkpoints.restore_checkpoint(
                    ckpt_dir=str(current_checkpoint_path),
                    target=state,
                    prefix=checkpoint_prefix
                )
                print(
                    f"Restored model for {current_attn_type} from checkpoint, step: {state.step}")
            except FileNotFoundError:
                print(
                    f"No checkpoint found for {current_attn_type} at {current_checkpoint_path}. Starting from scratch.")
            except Exception as e:
                print(
                    f"Could not restore checkpoint for {current_attn_type}: {e}. Starting from scratch.")

        state = flax_utils.replicate(state)

        print(f"Available devices: {jax.device_count()}")
        param_cnt = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(
            f"Total parameters for {current_attn_type}: {param_cnt:,} ({param_cnt/1e6:.2f} M)")

        try:
            example_param = jax.tree_util.tree_leaves(
                flax_utils.unreplicate(state).params['layers_0']['attn'])[0]
            if isinstance(example_param, dict) and 'kernel' in example_param:  # For Dense layers
                print("Model dtype (example): ", example_param['kernel'].dtype)
            else:  # For directly defined params like in MLA
                print("Model dtype (example): ", example_param.dtype)
        except (KeyError, IndexError, TypeError) as e:
            print(f"Could not determine model dtype from a standard path: {e}")

        save_interval = max(
            1, total_steps // args.num_chkpts if args.num_chkpts > 0 else total_steps + 1)
        print(
            f"Total Steps: {total_steps}, saving every {save_interval} steps for {current_attn_type}")

        # --- Training Loop ---
        for epoch in range(1, config.epochs + 1):
            progress_bar = tqdm(
                total=steps_per_epoch, desc=f"Epoch {epoch} [{current_attn_type.upper()}]", unit="step")

            # Create a new data iterator for each epoch if shuffle=True
            epoch_rng_key = jax.random.fold_in(training_rng, epoch)
            # Global batch size for dataset iteration
            global_batch_size = config.batch_size_per_device * jax.device_count()
            batches = (shard_batch(b) for b in batch_dataset(
                dataset, global_batch_size
            ))
            prefetch_iter = flax_utils.prefetch_to_device(batches, size=3)

            for batch_data in prefetch_iter:
                # Ensure training_rng is updated to avoid reusing keys across steps if not careful
                curr_step_rng, training_rng = jax.random.split(training_rng)
                device_rngs = jax.random.split(
                    curr_step_rng, jax.device_count())

                state, metrics = train_step(
                    state, batch_data, device_rngs, config.max_norm, transformer_config_obj
                )

                update_step = int(flax_utils.unreplicate(state).step)

                if (update_step > 0 and update_step % save_interval == 0) or (update_step == total_steps):
                    if jax.process_index() == 0:  # Save only from process 0 in multi-host setup
                        chkpt_state = flax_utils.unreplicate(state)
                        checkpoints.save_checkpoint(
                            str(current_checkpoint_path),
                            target=chkpt_state,
                            step=update_step,
                            overwrite=True,  # Overwrite previous checkpoint at the same step number
                            keep=args.num_chkpts,
                            prefix=checkpoint_prefix
                        )
                        print(
                            f"Saved model checkpoint for {current_attn_type} at step {update_step}")

                if update_step % 10 == 0:  # Log less frequently to reduce noise
                    log_metrics = {
                        "examples_seen": update_step * effective_batch_size,  # total examples seen
                        "train_loss_total": float(metrics['train_loss_total'][0]),
                        "train_loss_main": float(metrics['train_loss_main'][0]),
                        "perplexity_main": math.exp(float(metrics['train_loss_main'][0])),
                        "grad_norm_raw": float(metrics['grad_norm_raw'][0]),
                        "grad_norm_clipped_metric": float(metrics['grad_norm_clipped_metric'][0]),
                        "learning_rate": float(lr_schedule(update_step)),
                    }
                    if transformer_config_obj.num_experts and transformer_config_obj.num_experts > 0:
                        log_metrics["train_loss_aux_moe"] = float(
                            metrics['train_loss_aux_moe'][0])

                    wandb.log(log_metrics, step=update_step)
                    progress_bar.set_postfix(
                        loss=log_metrics['train_loss_total'])

                progress_bar.update(1)
                if update_step >= total_steps:
                    break

            progress_bar.close()
            print(
                f"Epoch {epoch} for {current_attn_type} completed. Final batch total loss: {metrics['train_loss_total'][0]:.4f}")
            if update_step >= total_steps:
                print(
                    f"Reached total_steps ({total_steps}), finishing training for {current_attn_type}.")
                break

        wandb.finish()

    print("\nAll attention mechanism evaluations complete.")


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    main()
