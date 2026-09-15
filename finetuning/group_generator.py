# import torch
# from typing import Dict, Any, Optional
# from transformers import AutoModel


# class MossTTSGroupGenerator:
#     def __init__(
#         self,
#         model,
#         tokenizer,
#         codec_model_path: str,
#         device: str = "cuda",
#         group_size: int = 4,
#         temperature: float = 0.8,
#         top_p: float = 0.95,
#         max_new_frames: int = 375,  
#         sample_rate: int = 48000
#     ):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device
#         self.group_size = group_size
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_new_frames = max_new_frames
#         self.sample_rate = sample_rate

#         print(f"Loading MOSS Audio Codec from: {codec_model_path}...")
#         self.codec = AutoModel.from_pretrained(
#             codec_model_path,
#             trust_remote_code=True
#         ).to(self.device)
#         self.codec.eval()

#     @torch.no_grad()
#     def generate_group_samples(
#         self,
#         prompt_input_ids: torch.LongTensor,
#         prompt_attention_mask: Optional[torch.Tensor] = None,
#     ) -> Dict[str, Any]:

#         if prompt_input_ids.ndim > 2:
#             prompt_input_ids = prompt_input_ids.view(-1, prompt_input_ids.shape[-1])

#         batch_size, seq_len = prompt_input_ids.shape

#         expanded_input_ids = prompt_input_ids.repeat_interleave(self.group_size, dim=0).to(self.device)

#         expanded_attention_mask = None
#         if prompt_attention_mask is not None:
#             if prompt_attention_mask.ndim > 2:
#                 prompt_attention_mask = prompt_attention_mask.view(-1, prompt_attention_mask.shape[-1])
#             expanded_attention_mask = prompt_attention_mask.repeat_interleave(self.group_size, dim=0).to(self.device)

#         gen_sequences = self.model.generate(
#             input_ids=expanded_input_ids,
#             max_new_frames=self.max_new_frames,
#             do_sample=True,
#             text_temperature=self.temperature,
#             audio_temperature=self.temperature,
#             text_top_p=self.top_p,
#             audio_top_p=self.top_p,
#         )

#         while not isinstance(gen_sequences, torch.Tensor):
#             if hasattr(gen_sequences, "sequences"):
#                 gen_sequences = gen_sequences.sequences
#             elif isinstance(gen_sequences, (tuple, list)):
#                 gen_sequences = gen_sequences[0]
#             else:
#                 break

#         prompt_len = seq_len
#         gen_audio_tokens = gen_sequences[:, prompt_len:, 1:]

#         generated_wavs = []
#         for i in range(gen_audio_tokens.shape[0]):
#             single_audio_tokens = gen_audio_tokens[i]  # (gen_len, n_vq)
            
#             tokens_for_codec = single_audio_tokens.transpose(0, 1).unsqueeze(0)
            
#             try:
#                 wav = self.codec.decode(tokens_for_codec)
#                 if isinstance(wav, tuple):
#                     wav = wav[0]
#                 wav = wav.squeeze(0).cpu()  # Waveform (1, samples)
#             except Exception as e:
#                 wav = torch.zeros((1, self.sample_rate), dtype=torch.float32)

#             generated_wavs.append(wav)

#         return {
#             "expanded_input_ids": expanded_input_ids,
#             "expanded_attention_mask": expanded_attention_mask,
#             "gen_sequences": gen_sequences,
#             "gen_audio_tokens": gen_audio_tokens,
#             "generated_wavs": generated_wavs, 
#             "prompt_len": prompt_len
#         }


#-----------------------
import logging
import traceback
from typing import Dict, Any, Optional

import torch
from transformers import AutoModel


# ============================================================
# Logger
# ============================================================

logger = logging.getLogger("MossTTSGroupGenerator")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


def describe_object(name: str, obj: Any):
    """
    Print detailed information about an object returned by HF generate().
    """

    logger.info("=" * 80)
    logger.info("OBJECT DEBUG: %s", name)
    logger.info("type       = %s", type(obj))

    # Tensor
    if torch.is_tensor(obj):
        logger.info("is_tensor  = True")
        logger.info("shape      = %s", tuple(obj.shape))
        logger.info("dtype      = %s", obj.dtype)
        logger.info("device     = %s", obj.device)

        if obj.numel() > 0:
            try:
                logger.info(
                    "min/max    = %.4f / %.4f",
                    obj.float().min().item(),
                    obj.float().max().item(),
                )
            except Exception:
                pass

        logger.info("=" * 80)
        return

    logger.info("is_tensor  = False")

    # ModelOutput / object
    if hasattr(obj, "keys"):
        try:
            logger.info("keys       = %s", list(obj.keys()))
        except Exception as e:
            logger.warning("Could not inspect keys: %s", e)

    if hasattr(obj, "sequences"):
        sequences = getattr(obj, "sequences")
        logger.info(
            "has sequences = True | type=%s | shape=%s",
            type(sequences),
            getattr(sequences, "shape", None),
        )
    else:
        logger.info("has sequences = False")

    # Tuple / List
    if isinstance(obj, (tuple, list)):
        logger.info("container length = %d", len(obj))

        for i, item in enumerate(obj):
            logger.info(
                "[%d] type=%s shape=%s",
                i,
                type(item),
                getattr(item, "shape", None),
            )

    # Generic attributes
    try:
        logger.info(
            "has logits      = %s",
            hasattr(obj, "logits")
        )
    except Exception:
        pass

    logger.info("=" * 80)


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
        sample_rate: int = 48000,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.group_size = group_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_frames = max_new_frames
        self.sample_rate = sample_rate

        logger.info("=" * 80)
        logger.info("Initializing MossTTSGroupGenerator")
        logger.info("device          = %s", device)
        logger.info("group_size      = %s", group_size)
        logger.info("temperature     = %s", temperature)
        logger.info("top_p           = %s", top_p)
        logger.info("max_new_frames  = %s", max_new_frames)
        logger.info("sample_rate     = %s", sample_rate)
        logger.info("=" * 80)

        # --------------------------------------------------------
        # Model information
        # --------------------------------------------------------

        logger.info(
            "Main model type: %s",
            type(self.model)
        )

        try:
            logger.info(
                "Main model device: %s",
                next(self.model.parameters()).device
            )
        except Exception:
            logger.warning(
                "Could not determine main model device"
            )

        # --------------------------------------------------------
        # Codec
        # --------------------------------------------------------

        print(f"Loading MOSS Audio Codec from: {codec_model_path}...")

        logger.info(
            "Loading codec from: %s",
            codec_model_path
        )

        self.codec = AutoModel.from_pretrained(
            codec_model_path,
            trust_remote_code=True,
        ).to(self.device)

        self.codec.eval()

        logger.info(
            "Codec loaded successfully: %s",
            type(self.codec)
        )

        try:
            logger.info(
                "Codec device: %s",
                next(self.codec.parameters()).device
            )
        except Exception:
            logger.warning(
                "Could not determine codec device"
            )

    @torch.no_grad()
    def generate_group_samples(
        self,
        prompt_input_ids: torch.LongTensor,
        prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        logger.info("")
        logger.info("=" * 100)
        logger.info("START generate_group_samples()")
        logger.info("=" * 100)

        # ========================================================
        # 1. Input validation
        # ========================================================

        logger.info(
            "prompt_input_ids BEFORE processing: type=%s shape=%s dtype=%s device=%s",
            type(prompt_input_ids),
            getattr(prompt_input_ids, "shape", None),
            getattr(prompt_input_ids, "dtype", None),
            getattr(prompt_input_ids, "device", None),
        )

        if prompt_input_ids is None:
            raise ValueError(
                "prompt_input_ids is None"
            )

        if not torch.is_tensor(prompt_input_ids):
            raise TypeError(
                f"prompt_input_ids must be Tensor, got {type(prompt_input_ids)}"
            )

        # if prompt_input_ids.ndim > 2:
    
        #     logger.warning(
        #         "prompt_input_ids ndim=%d > 2, reshaping...",
        #         prompt_input_ids.ndim
        #     )

        #     prompt_input_ids = prompt_input_ids.view(
        #         -1,
        #         prompt_input_ids.shape[-1]
        #     )

        # batch_size, seq_len = prompt_input_ids.shape

        # logger.info(
        #     "Original batch_size=%d | seq_len=%d",
        #     batch_size,
        #     seq_len
        # )

        # # ========================================================
        # # 2. Expand group
        # # ========================================================

        # expanded_input_ids = (
        #     prompt_input_ids
        #     .repeat_interleave(self.group_size, dim=0)
        #     .to(self.device)
        # )

        # logger.info(
        #     "expanded_input_ids: shape=%s dtype=%s device=%s",
        #     tuple(expanded_input_ids.shape),
        #     expanded_input_ids.dtype,
        #     expanded_input_ids.device,
        # )

        # MOSS input shape: [batch, frames, channels]
        if prompt_input_ids.ndim != 3:
            raise ValueError(
                f"Expected prompt_input_ids to have 3 dimensions "
                f"[batch, frames, channels], got shape={prompt_input_ids.shape}"
            )

        batch_size, seq_len, num_channels = prompt_input_ids.shape

        print(
            f"MOSS input: batch={batch_size}, "
            f"frames={seq_len}, "
            f"channels={num_channels}"
        )

        expanded_input_ids = prompt_input_ids.repeat_interleave(
            self.group_size,
            dim=0
        ).to(self.device)

        print(
            f"expanded_input_ids shape={expanded_input_ids.shape}, "
            f"device={expanded_input_ids.device}"
            )

        expanded_attention_mask = None

        if prompt_attention_mask is not None:

            logger.info(
                "prompt_attention_mask BEFORE processing: "
                "type=%s shape=%s dtype=%s device=%s",
                type(prompt_attention_mask),
                getattr(prompt_attention_mask, "shape", None),
                getattr(prompt_attention_mask, "dtype", None),
                getattr(prompt_attention_mask, "device", None),
            )

            if prompt_attention_mask.ndim > 2:

                logger.warning(
                    "prompt_attention_mask ndim=%d > 2, reshaping...",
                    prompt_attention_mask.ndim
                )

                prompt_attention_mask = prompt_attention_mask.view(
                    -1,
                    prompt_attention_mask.shape[-1]
                )

            expanded_attention_mask = (
                prompt_attention_mask
                .repeat_interleave(
                    self.group_size,
                    dim=0
                )
                .to(self.device)
            )

            logger.info(
                "expanded_attention_mask: shape=%s dtype=%s device=%s",
                tuple(expanded_attention_mask.shape),
                expanded_attention_mask.dtype,
                expanded_attention_mask.device,
            )

        # ========================================================
        # 3. Generate
        # ========================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("CALLING model.generate()")
        logger.info("-" * 80)

        logger.info(
            "generate arguments:"
        )

        logger.info(
            "input_ids shape = %s",
            tuple(expanded_input_ids.shape)
        )

        logger.info(
            "max_new_frames = %s",
            self.max_new_frames
        )

        logger.info(
            "do_sample = True"
        )

        logger.info(
            "text_temperature = %s",
            self.temperature
        )

        logger.info(
            "audio_temperature = %s",
            self.temperature
        )

        logger.info(
            "text_top_p = %s",
            self.top_p
        )

        logger.info(
            "audio_top_p = %s",
            self.top_p
        )

        try:

            gen_sequences = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_frames=self.max_new_frames,
                do_sample=True,
                text_temperature=self.temperature,
                audio_temperature=self.temperature,
                text_top_p=self.top_p,
                audio_top_p=self.top_p,
            )

        except Exception as e:

            logger.error("=" * 80)
            logger.error("MODEL.GENERATE() FAILED")
            logger.error("=" * 80)

            logger.error(
                "Exception type: %s",
                type(e)
            )

            logger.error(
                "Exception: %s",
                str(e)
            )

            logger.error(
                "Traceback:\n%s",
                traceback.format_exc()
            )

            raise

        # ========================================================
        # 4. VERY IMPORTANT DEBUG
        # ========================================================

        describe_object(
            "RAW OUTPUT FROM model.generate()",
            gen_sequences
        )

        # ========================================================
        # 5. Extract sequences safely
        # ========================================================

        logger.info(
            "Attempting to extract actual generated sequence..."
        )

        # original_output = gen_sequences

        # # --------------------------------------------------------
        # # Case 1: Tensor
        # # --------------------------------------------------------

        # if torch.is_tensor(gen_sequences):

        #     logger.info(
        #         "generate() returned Tensor directly."
        #     )


        original_output = gen_sequences

        # --------------------------------------------------------
        # Case 1: Tensor
        # --------------------------------------------------------

        if torch.is_tensor(gen_sequences):

            logger.info(
                "generate() returned Tensor directly."
            )

        # --------------------------------------------------------
        # Case 2: MOSS custom generation output
        # --------------------------------------------------------

        elif hasattr(gen_sequences, "audio_token_ids"):

            logger.info(
                "MOSS generation output detected."
            )

            logger.info(
                "audio_token_ids type=%s shape=%s",
                type(gen_sequences.audio_token_ids),
                getattr(gen_sequences.audio_token_ids, "shape", None),
            )

            logger.info(
                "prompt_input_ids type=%s shape=%s",
                type(gen_sequences.prompt_input_ids),
                getattr(gen_sequences.prompt_input_ids, "shape", None),
            )

            gen_sequences = gen_sequences.audio_token_ids

            describe_object(
                "EXTRACTED MOSS audio_token_ids",
                gen_sequences
            )

        # --------------------------------------------------------
        # Case 3: ModelOutput with .sequences
        # --------------------------------------------------------

        elif hasattr(gen_sequences, "sequences"):

            logger.info(
                "generate() returned object with `.sequences`."
            )

            gen_sequences = gen_sequences.sequences

            describe_object(
                "EXTRACTED .sequences",
                gen_sequences
            )
        # --------------------------------------------------------
        # Case 2: ModelOutput with .sequences
        # --------------------------------------------------------

        elif hasattr(gen_sequences, "sequences"):

            logger.info(
                "generate() returned object with `.sequences`."
            )

            gen_sequences = gen_sequences.sequences

            describe_object(
                "EXTRACTED .sequences",
                gen_sequences
            )

        # --------------------------------------------------------
        # Case 3: Tuple/List
        # --------------------------------------------------------

        elif isinstance(gen_sequences, (tuple, list)):

            logger.warning(
                "generate() returned %s with length=%d",
                type(gen_sequences),
                len(gen_sequences)
            )

            if len(gen_sequences) == 0:

                raise RuntimeError(
                    "model.generate() returned an EMPTY tuple/list."
                )

            # Inspect every item before choosing
            for i, item in enumerate(gen_sequences):

                describe_object(
                    f"generate_output[{i}]",
                    item
                )

            tensor_candidates = [
                item
                for item in gen_sequences
                if torch.is_tensor(item)
            ]

            if len(tensor_candidates) == 1:

                logger.info(
                    "Exactly one Tensor found in tuple/list. Using it."
                )

                gen_sequences = tensor_candidates[0]

            elif len(tensor_candidates) > 1:

                raise RuntimeError(
                    "model.generate() returned multiple Tensor candidates. "
                    "Cannot safely determine which one is the generated sequence."
                )

            else:

                raise RuntimeError(
                    "model.generate() returned tuple/list but no Tensor "
                    "could be identified."
                )

        # --------------------------------------------------------
        # Unknown output
        # --------------------------------------------------------

        else:

            raise TypeError(
                "Unsupported output type from model.generate(): "
                f"{type(original_output)}"
            )

        # ========================================================
        # 6. Validate final sequence tensor
        # ========================================================

        describe_object(
            "FINAL gen_sequences",
            gen_sequences
        )

        if not torch.is_tensor(gen_sequences):

            raise TypeError(
                "After extraction, gen_sequences is still not a Tensor. "
                f"Got {type(gen_sequences)}"
            )

        if gen_sequences.ndim != 3:

            raise RuntimeError(
                "Unexpected generated sequence dimensions. "
                f"Expected 3D tensor [batch, sequence, channels], "
                f"got shape={tuple(gen_sequences.shape)}"
            )

        expected_batch = batch_size * self.group_size

        if gen_sequences.shape[0] != expected_batch:

            raise RuntimeError(
                "Generated batch size mismatch. "
                f"Expected {expected_batch}, "
                f"got {gen_sequences.shape[0]}"
            )

        # ========================================================
        # 7. Extract audio tokens
        # ========================================================

        logger.info(
            "Extracting audio tokens..."
        )

        logger.info(
            "gen_sequences shape = %s",
            tuple(gen_sequences.shape)
        )

        # MOSS `audio_token_ids` already contains GENERATED audio only.
        #
        # Shape:
        #   [batch, generated_frames, num_codebooks]
        #
        # Example:
        #   [4, 15, 16]

        if gen_sequences.ndim != 3:
            raise RuntimeError(
                "Expected MOSS audio_token_ids to be 3D. "
                f"Got shape={tuple(gen_sequences.shape)}"
            )

        generated_frames = gen_sequences.shape[1]
        num_codebooks = gen_sequences.shape[2]

        logger.info(
            "generated_frames = %d",
            generated_frames
        )

        logger.info(
            "num_codebooks = %d",
            num_codebooks
        )

        if generated_frames == 0:
            raise RuntimeError(
                "MOSS generated zero audio frames."
            )

        if num_codebooks != 16:
            raise RuntimeError(
                "Unexpected number of audio codebooks. "
                f"Expected 16, got {num_codebooks}. "
                f"shape={tuple(gen_sequences.shape)}"
            )

        # IMPORTANT:
        # No prompt slicing.
        # No channel slicing.
        # audio_token_ids is already generated audio.

        gen_audio_tokens = gen_sequences

        logger.info(
            "gen_audio_tokens shape = %s",
            tuple(gen_audio_tokens.shape)
        )

        logger.info(
            "gen_audio_tokens dtype = %s",
            gen_audio_tokens.dtype
        )

        logger.info(
            "gen_audio_tokens device = %s",
            gen_audio_tokens.device
        )

        # For MOSS audio_token_ids, there is no prompt
        # inside this tensor.
        prompt_len = 0

        # ========================================================
        # 8. Decode audio
        # ========================================================

        generated_wavs = []

        logger.info(
            "Starting codec decoding for %d samples...",
            gen_audio_tokens.shape[0]
        )

        for i in range(gen_audio_tokens.shape[0]):

            logger.info("")
            logger.info(
                "Decoding sample %d/%d",
                i + 1,
                gen_audio_tokens.shape[0]
            )

            single_audio_tokens = gen_audio_tokens[i]

            logger.info(
                "single_audio_tokens shape = %s",
                tuple(single_audio_tokens.shape)
            )

            tokens_for_codec = (
                single_audio_tokens
                .transpose(0, 1)
                .unsqueeze(0)
            )

            logger.info(
                "tokens_for_codec shape = %s",
                tuple(tokens_for_codec.shape)
            )

            try:

                wav = self.codec.decode(
                    tokens_for_codec
                )

                logger.info(
                    "Raw codec output type = %s",
                    type(wav)
                )

                if isinstance(wav, tuple):

                    logger.info(
                        "Codec returned tuple. Taking first element."
                    )

                    wav = wav[0]

                logger.info(
                    "Raw waveform shape = %s",
                    getattr(wav, "shape", None)
                )

                wav = wav.squeeze(0).cpu()

                logger.info(
                    "Final waveform shape = %s",
                    tuple(wav.shape)
                )

                generated_wavs.append(wav)

            except Exception as e:

                logger.error("=" * 80)
                logger.error(
                    "CODEC DECODE FAILED FOR SAMPLE %d",
                    i
                )
                logger.error(
                    "Exception: %s",
                    str(e)
                )
                logger.error(
                    "Traceback:\n%s",
                    traceback.format_exc()
                )
                logger.error("=" * 80)

                # Keep pipeline alive
                wav = torch.zeros(
                    (1, self.sample_rate),
                    dtype=torch.float32
                )

                generated_wavs.append(wav)

        # ========================================================
        # 9. Return
        # ========================================================

        logger.info("")
        logger.info("=" * 100)
        logger.info("generate_group_samples() COMPLETED")
        logger.info(
            "generated_wavs = %d",
            len(generated_wavs)
        )
        logger.info(
            "gen_sequences shape = %s",
            tuple(gen_sequences.shape)
        )
        logger.info(
            "gen_audio_tokens shape = %s",
            tuple(gen_audio_tokens.shape)
        )
        logger.info("=" * 100)

        return {
            "expanded_input_ids": expanded_input_ids,
            "expanded_attention_mask": expanded_attention_mask,
            "gen_sequences": gen_sequences,
            "gen_audio_tokens": gen_audio_tokens,
            "generated_wavs": generated_wavs,
            "prompt_len": prompt_len,
        }