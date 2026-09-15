import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.common import load_jsonl_spec
from finetuning.dataset import MossTTSNanoSFTDataset
from finetuning.reward_evaluator import MossTTSRewardEvaluator
from finetuning.group_generator import MossTTSGroupGenerator
from finetuning.grpo_utils import (
    compute_sequence_log_probs,
    compute_grpo_loss,
    calculate_group_advantages,
)

# ==============================================================================
# 2. Main Training Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO Fine-Tuning for MOSS-TTS-Nano")
    parser.add_argument("--model-path", type=str, default="models/MOSS-TTS-Nano")
    parser.add_argument("--codec-path", type=str, default="models/MOSS-Audio-Tokenizer-Nano")
    parser.add_argument("--train-jsonl", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--output-dir", type=str, default="output/moss_tts_nano_grpo")
    parser.add_argument("--group-size", type=int, default=4, help="Number of generations G per prompt")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16")
    device = accelerator.device

    # 1. Load Tokenizer & Base Models
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy Model (Trainable)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # Reference Model (Frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # 2. Initialize Evaluator & Group Generator
    reward_evaluator = MossTTSRewardEvaluator(
        device=str(device),
        cohere_model_name="CohereLabs/cohere-transcribe-03-2026"
    )

    group_generator = MossTTSGroupGenerator(
        model=model,
        tokenizer=tokenizer,
        codec_model_path=args.codec_path,
        device=str(device),
        group_size=args.group_size,
        sample_rate=48000
    )

    # 3. Load Dataset (supports id, audio, text, ref_audio)
    _, records = load_jsonl_spec(args.train_jsonl)
    dataset = MossTTSNanoSFTDataset(records, tokenizer=tokenizer, model_config=model.config, max_length=1024)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.per_device_batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # 4. GRPO Optimization Loop
    accelerator.print(f">>> Starting GRPO Fine-Tuning across {len(records)} samples...")
    
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            prompt_input_ids = batch["input_ids"]
            prompt_attention_mask = batch["attention_mask"]

            # Dataset field mapping
            target_texts = batch.get("text", [""] * len(prompt_input_ids))
            gt_audio_paths = batch.get("audio", None)
            ref_audio_paths = batch.get("ref_audio", None)
            sample_ids = batch.get("id", [f"step_{step}_{i}" for i in range(len(prompt_input_ids))])

            # A) Generate group samples (G per prompt)
            gen_out = group_generator.generate_group_samples(prompt_input_ids, prompt_attention_mask)
            gen_sequences = gen_out["gen_sequences"]
            gen_wavs = gen_out["generated_wavs"]  # 48kHz Waveforms
            prompt_len = gen_out["prompt_len"]

            # B) Expand target attributes to match group size G
            expanded_target_texts = [text for text in target_texts for _ in range(args.group_size)]
            
            # Target audio preference: ground truth audio > reference prompt audio
            target_ref_paths = gt_audio_paths if gt_audio_paths is not None else ref_audio_paths
            if target_ref_paths is not None:
                expanded_ref_wavs = [path for path in target_ref_paths for _ in range(args.group_size)]
            else:
                expanded_ref_wavs = gen_wavs

            # C) Evaluate Rewards (ASR Accuracy + Speaker Similarity + Duration Penalty)
            eval_res = reward_evaluator.evaluate_batch(
                gen_wavs=gen_wavs,
                ref_wavs=expanded_ref_wavs,
                target_texts=expanded_target_texts,
                sample_rate=48000
            )
            rewards = eval_res["total_rewards"]

            # D) Group Advantage Estimation
            advantages = calculate_group_advantages(rewards, group_size=args.group_size)

            # E) Sequence Log Probabilities (Policy vs Reference)
            expanded_attention_mask = torch.ones_like(gen_sequences[:, :, 0], dtype=torch.bool)
            
            policy_log_probs = compute_sequence_log_probs(
                model, gen_sequences[:, :, 0], expanded_attention_mask, gen_sequences, prompt_len
            )

            with torch.no_grad():
                ref_log_probs = compute_sequence_log_probs(
                    ref_model, gen_sequences[:, :, 0], expanded_attention_mask, gen_sequences, prompt_len
                )

            # F) GRPO Loss Calculation & Optimization Step
            loss = compute_grpo_loss(
                policy_log_probs=policy_log_probs,
                old_log_probs=policy_log_probs.detach(),
                ref_log_probs=ref_log_probs,
                advantages=advantages,
                clip_eps=args.clip_eps,
                kl_beta=args.kl_beta
            )

            accelerator.backward(loss)
            optimizer.step()

            # G) Metric Logging
            if step % 5 == 0:
                mean_r = rewards.mean().item()
                mean_wer = np.mean(eval_res.get("r_wer", [0]))
                mean_sim = np.mean(eval_res.get("r_sim", [0]))
                
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | ID: {sample_ids[0]} | "
                    f"Loss: {loss.item():.4f} | Reward: {mean_r:.4f} | "
                    f"WER-Score: {mean_wer:.3f} | Speaker-Sim: {mean_sim:.3f}"
                )

    # 5. Save Final Model
    output_path = Path(args.output_dir) / "checkpoint-final"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        accelerator.print(f"\n>>> Training Complete. Saved model checkpoint to: {output_path}")


if __name__ == "__main__":
    main()