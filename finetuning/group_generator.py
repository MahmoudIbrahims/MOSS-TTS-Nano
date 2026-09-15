import torch
from typing import Dict, Any, Optional
from transformers import AutoModel


class MossTTSGroupGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        codec_model_path: str,
        device: str = "cuda",
        group_size: int = 4,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_frames: int = 375,  
        sample_rate: int = 48000
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.group_size = group_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_frames = max_new_frames
        self.sample_rate = sample_rate

        print(f"Loading MOSS Audio Codec from: {codec_model_path}...")
        self.codec = AutoModel.from_pretrained(
            codec_model_path,
            trust_remote_code=True
        ).to(self.device)
        self.codec.eval()

    @torch.no_grad()
    def generate_group_samples(
        self,
        prompt_input_ids: torch.LongTensor,
        prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        if prompt_input_ids.ndim > 2:
            prompt_input_ids = prompt_input_ids.view(-1, prompt_input_ids.shape[-1])

        batch_size, seq_len = prompt_input_ids.shape

        expanded_input_ids = prompt_input_ids.repeat_interleave(self.group_size, dim=0).to(self.device)

        expanded_attention_mask = None
        if prompt_attention_mask is not None:
            if prompt_attention_mask.ndim > 2:
                prompt_attention_mask = prompt_attention_mask.view(-1, prompt_attention_mask.shape[-1])
            expanded_attention_mask = prompt_attention_mask.repeat_interleave(self.group_size, dim=0).to(self.device)

        gen_sequences = self.model.generate(
            input_ids=expanded_input_ids,
            # attention_mask=expanded_attention_mask,
            max_new_frames=self.max_new_frames,
            do_sample=True,
            text_temperature=self.temperature,
            audio_temperature=self.temperature,
            text_top_p=self.top_p,
            audio_top_p=self.top_p,
        )

        if hasattr(gen_sequences, "sequences"):
            gen_sequences = gen_sequences.sequences
        elif isinstance(gen_sequences, (tuple, list)):
            gen_sequences = gen_sequences[0]

        prompt_len = seq_len
        gen_audio_tokens = gen_sequences[:, prompt_len:, 1:]

        generated_wavs = []
        for i in range(gen_audio_tokens.shape[0]):
            single_audio_tokens = gen_audio_tokens[i]  # (gen_len, n_vq)
            
            tokens_for_codec = single_audio_tokens.transpose(0, 1).unsqueeze(0)
            
            try:
                wav = self.codec.decode(tokens_for_codec)
                if isinstance(wav, tuple):
                    wav = wav[0]
                wav = wav.squeeze(0).cpu()  # Waveform (1, samples)
            except Exception as e:
                wav = torch.zeros((1, self.sample_rate), dtype=torch.float32)

            generated_wavs.append(wav)

        return {
            "expanded_input_ids": expanded_input_ids,
            "expanded_attention_mask": expanded_attention_mask,
            "gen_sequences": gen_sequences,
            "gen_audio_tokens": gen_audio_tokens,
            "generated_wavs": generated_wavs, 
            "prompt_len": prompt_len
        }