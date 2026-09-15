import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from typing import List, Dict, Any, Tuple
import jiwer
from transformers import AutoProcessor, CohereAsrForConditionalGeneration


class MossTTSRewardEvaluator:
    def __init__(
        self,
        device: str = "cuda",
        cohere_model_name: str = "CohereLabs/cohere-transcribe-03-2026",
        speaker_model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        weights: Dict[str, float] = None
    ):
        self.device = device
        
        self.weights = weights or {
            "wer": 0.35,
            "speaker_sim": 0.30,
            "quality": 0.15,
            "duration_stability": 0.20
        }

        print(f"Loading Cohere ASR model from {cohere_model_name}...")
        self.asr_processor = AutoProcessor.from_pretrained(cohere_model_name)
        self.asr_model = CohereAsrForConditionalGeneration.from_pretrained(
            cohere_model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self.asr_model.eval()

        print(f"Loading Speaker Encoder model ({speaker_model_name})...")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.spk_encoder = EncoderClassifier.from_hparams(
                source=speaker_model_name,
                run_opts={"device": self.device}
            )
        except ImportError:
            print("Warning: SpeechBrain not installed. Speaker Similarity will fallback to 0.5")
            self.spk_encoder = None

        print("Loading UTMOS Audio Quality Model...")
        try:
            self.utmos_model = torch.hub.load(
                repo_or_dir="sarulab-speech/UTMOS22",
                model="utmos22_strong",
                trust_repo=True
            ).to(self.device)
            self.utmos_model.eval()
        except Exception as e:
            print(f"Warning: UTMOS failed to load ({e}). Quality reward fallback enabled.")
            self.utmos_model = None

    @torch.no_grad()
    def compute_asr_wer_reward(self, audio_wavs: List[torch.Tensor], target_texts: List[str], sample_rate: int = 48000) -> List[float]:
        rewards = []
        for wav, target_text in zip(audio_wavs, target_texts):
            if sample_rate != 16000:
                wav_16k = torchaudio.functional.resample(wav.cpu(), sample_rate, 16000)
            else:
                wav_16k = wav.cpu()

            inputs = self.asr_processor(wav_16k.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
            generated_ids = self.asr_model.generate(**inputs)
            transcription = self.asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            try:
                wer = jiwer.wer(target_text.lower(), transcription.lower())
                wer_reward = max(0.0, 1.0 - wer)
            except Exception:
                wer_reward = 0.0
            
            rewards.append(float(wer_reward))
        return rewards

    @torch.no_grad()
    def compute_speaker_similarity(self, gen_wavs: List[torch.Tensor], ref_wavs: List[torch.Tensor], sample_rate: int = 48000) -> List[float]:
        if self.spk_encoder is None:
            return [0.5] * len(gen_wavs)

        rewards = []
        for gen_w, ref_w in zip(gen_wavs, ref_wavs):
            gen_16k = torchaudio.functional.resample(gen_w.to(self.device), sample_rate, 16000)
            ref_16k = torchaudio.functional.resample(ref_w.to(self.device), sample_rate, 16000)

            emb1 = self.spk_encoder.encode_batch(gen_16k)
            emb2 = self.spk_encoder.encode_batch(ref_16k)

            sim = F.cosine_similarity(emb1, emb2, dim=-1).mean().item()
            norm_sim = max(0.0, (sim + 1.0) / 2.0)
            rewards.append(float(norm_sim))
        return rewards

    @torch.no_grad()
    def compute_audio_quality(self, gen_wavs: List[torch.Tensor], sample_rate: int = 48000) -> List[float]:
        if self.utmos_model is None:
            return [0.7] * len(gen_wavs)

        rewards = []
        for gen_w in gen_wavs:
            gen_16k = torchaudio.functional.resample(gen_w.to(self.device), sample_rate, 16000)
            if gen_16k.ndim == 1:
                gen_16k = gen_16k.unsqueeze(0)
            score = self.utmos_model(gen_16k, 16000).item()
            # UTMOS scores typically range between 1.0 and 5.0
            norm_score = np.clip((score - 1.0) / 4.0, 0.0, 1.0)
            rewards.append(float(norm_score))
        return rewards

    def compute_duration_stability(
        self, 
        gen_wavs: List[torch.Tensor], 
        target_texts: List[str], 
        sample_rate: int = 48000,
        expected_cps: float = 12.0  
    ) -> List[float]:
        rewards = []
        for gen_w, text in zip(gen_wavs, target_texts):
            actual_duration = gen_w.shape[-1] / float(sample_rate)
            expected_duration = max(len(text) / expected_cps, 0.5)

            ratio = actual_duration / expected_duration
            
            if 0.7 <= ratio <= 1.4:
                reward = 1.0
            else:
                reward = np.exp(-2.0 * abs(ratio - 1.0))
            
            rewards.append(float(np.clip(reward, 0.0, 1.0)))
        return rewards

    def evaluate_batch(
        self, 
        gen_wavs: List[torch.Tensor], 
        ref_wavs: List[torch.Tensor], 
        target_texts: List[str], 
        sample_rate: int = 48000
    ) -> Dict[str, Any]:
        
        r_wer = self.compute_asr_wer_reward(gen_wavs, target_texts, sample_rate)
        r_sim = self.compute_speaker_similarity(gen_wavs, ref_wavs, sample_rate)
        r_qual = self.compute_audio_quality(gen_wavs, sample_rate)
        r_dur = self.compute_duration_stability(gen_wavs, target_texts, sample_rate)

        total_rewards = []
        for i in range(len(gen_wavs)):
            total_r = (
                self.weights["wer"] * r_wer[i] +
                self.weights["speaker_sim"] * r_sim[i] +
                self.weights["quality"] * r_qual[i] +
                self.weights["duration_stability"] * r_dur[i]
            )
            total_rewards.append(total_r)

        return {
            "total_rewards": torch.tensor(total_rewards, device=self.device, dtype=torch.float32),
            "r_wer": r_wer,
            "r_sim": r_sim,
            "r_qual": r_qual,
            "r_dur": r_dur
        }