import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator
from typing import Dict, Any, List

from reward_evaluator import MossTTSRewardEvaluator
from group_generator import MossTTSGroupGenerator


def compute_sequence_log_probs(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    labels: torch.LongTensor,
    prompt_len: int
) -> torch.Tensor:
    """
    labels Shape: (Batch_Size, Seq_Len, n_vq + 1)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True
    )
    global_hidden_states = outputs.global_hidden_states
    base_model = model.module if hasattr(model, "module") else model
    
    batch_size, seq_len, hidden_size = global_hidden_states.shape
    n_vq = int(base_model.config.n_vq)
    
    flat_hidden = global_hidden_states.reshape(batch_size * seq_len, hidden_size)
    local_dtype = base_model.local_transformer.ln_f.weight.dtype
    flat_hidden = flat_hidden.to(dtype=local_dtype)

    flat_labels = labels.reshape(batch_size * seq_len, n_vq + 1)
    local_inputs = torch.zeros(
        (batch_size * seq_len, n_vq + 1, hidden_size),
        dtype=local_dtype,
        device=flat_hidden.device,
    )
    local_inputs[:, 0, :] = flat_hidden

    text_targets = flat_labels[:, 0]
    safe_text_targets = text_targets.masked_fill(text_targets.lt(0), int(base_model.config.pad_token_id))
    local_inputs[:, 1, :] = base_model.transformer.wte(safe_text_targets)

    audio_targets = flat_labels[:, 1:]
    for ch in range(n_vq - 1):
        teacher_ids = audio_targets[:, ch]
        valid_mask = (teacher_ids >= 0) & (teacher_ids < base_model.audio_embeddings[ch].num_embeddings)
        safe_ids = teacher_ids.masked_fill(~valid_mask, 0)
        channel_embeds = base_model.audio_embeddings[ch](safe_ids) * valid_mask.unsqueeze(-1)
        local_inputs[:, ch + 2, :] = channel_embeds.to(dtype=local_dtype)

    local_attention_mask = torch.ones(
        (batch_size * seq_len, n_vq + 1),
        dtype=torch.bool,
        device=flat_hidden.device,
    )
    
    local_outputs = base_model.local_transformer(
        input_ids=None,
        attention_mask=local_attention_mask,
        inputs_embeds=local_inputs,
        use_cache=False,
        return_dict=True
    )
    local_hidden = local_outputs.last_hidden_state

    seq_log_probs = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.float32)

    for ch in range(n_vq):
        ch_logits = base_model.audio_lm_heads[ch](local_hidden[:, ch + 1, :]) # (batch*seq, vocab)
        ch_logits = ch_logits.view(batch_size, seq_len, -1)
        ch_targets = labels[:, :, ch + 1]
        
        log_p = F.log_softmax(ch_logits, dim=-1)
        
        mask = (ch_targets != -100)
        safe_targets = ch_targets.masked_fill(~mask, 0)
        target_log_p = torch.gather(log_p, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        target_log_p = target_log_p * mask.float()
        
        seq_log_probs += target_log_p.sum(dim=-1, keepdim=False) if target_log_p.ndim > 2 else target_log_p

    gen_log_probs = seq_log_probs[:, prompt_len:].sum(dim=-1) # (batch_size,)
    return gen_log_probs


def compute_grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01
) -> torch.Tensor:
    """
    حساب GRPO Loss مع Clipping و KL Divergence Penalty
    """
    # 1. Ratio
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # 2. Clipped Surrogate Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3. KL Penalty vs Reference Model
    kl_div = (policy_log_probs - ref_log_probs).mean()
    
    total_loss = policy_loss + kl_beta * kl_div
    return total_loss


def calculate_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """
    rewards Shape: (batch_size * group_size,)
    """
    rewards = rewards.view(-1, group_size) # (num_prompts, G)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    
    advantages = (rewards - mean) / (std + eps)
    return advantages.view(-1) # (batch_size * group_size,)