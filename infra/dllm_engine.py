"""Masked diffusion language model engine for LLaDA-style decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .llada_loader import load_llada_model


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype=torch.long)
    cols = torch.arange(steps, device=block_mask_index.device).unsqueeze(0)
    return num_transfer_tokens + (cols < rem.unsqueeze(1)).to(torch.long)


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None,
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be provided when threshold is None")
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)
    batch_size, seq_len = confidence.shape
    cols = torch.arange(seq_len, device=confidence.device).unsqueeze(0).expand(batch_size, seq_len)
    select_sorted = cols < num_transfer_tokens.unsqueeze(1).expand(batch_size, seq_len)
    transfer_int = torch.zeros(batch_size, seq_len, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index


@torch.no_grad()
def masked_diffusion_generate(
    model,
    prompt_ids: torch.LongTensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = None,
):
    prompt_ids = prompt_ids.to(model.device)
    x = torch.full(
        (prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_ids.shape[1]] = prompt_ids

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        start = prompt_ids.shape[1] + num_block * block_length
        end = start + block_length
        block_mask_index = x[:, start:end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        i = 0
        while True:
            logits = model(x).logits
            mask_index = x == mask_id
            mask_index[:, end:] = False
            quota = None if threshold is not None else num_transfer_tokens[:, i]
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                quota,
                threshold,
            )
            x = torch.where(transfer_index, x0, x)
            nfe += 1
            i += 1
            if (x[:, start:end] == mask_id).sum() == 0:
                break
    return x, nfe


@torch.no_grad()
def masked_diffusion_generate_with_prefix_cache(
    model,
    prompt_ids: torch.LongTensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = None,
):
    prompt_ids = prompt_ids.to(model.device)
    x = torch.full(
        (prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_ids.shape[1]] = prompt_ids

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        output = model(x, use_cache=True)
        past_key_values = [
            tuple(cache_tensor[:, :, :current_block_start].contiguous() for cache_tensor in layer_cache)
            for layer_cache in output.past_key_values
        ]
        mask_index = x == mask_id
        mask_index[:, current_block_end:] = False
        quota = None if threshold is not None else num_transfer_tokens[:, 0]
        x0, transfer_index = get_transfer_index(
            output.logits,
            temperature,
            remasking,
            mask_index,
            x,
            quota,
            threshold,
        )
        x = torch.where(transfer_index, x0, x)
        nfe += 1

        step_idx = 1
        while step_idx < steps_per_block:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = x[:, current_block_start:] == mask_id
            mask_index[:, block_length:] = False
            logits = model(
                x[:, current_block_start:],
                past_key_values=past_key_values,
                use_cache=True,
            ).logits
            quota = None if threshold is not None else num_transfer_tokens[:, step_idx]
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x[:, current_block_start:],
                quota,
                threshold,
            )
            updated_suffix = torch.where(transfer_index, x0, x[:, current_block_start:])
            x = torch.cat((x[:, :current_block_start], updated_suffix), dim=1)
            nfe += 1
            step_idx += 1
    return x, nfe


@torch.no_grad()
def masked_diffusion_generate_with_dual_cache(
    model,
    prompt_ids: torch.LongTensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = None,
):
    prompt_ids = prompt_ids.to(model.device)
    x = torch.full(
        (prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_ids.shape[1]] = prompt_ids

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = True

        mask_index = x == mask_id
        mask_index[:, current_block_end:] = False
        quota = None if threshold is not None else num_transfer_tokens[:, 0]
        x0, transfer_index = get_transfer_index(
            output.logits,
            temperature,
            remasking,
            mask_index,
            x,
            quota,
            threshold,
        )
        x = torch.where(transfer_index, x0, x)
        nfe += 1

        step_idx = 1
        while step_idx < steps_per_block:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            output = model(
                x[:, current_block_start:current_block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            )
            mask_index = x[:, current_block_start:current_block_end] == mask_id
            quota = None if threshold is not None else num_transfer_tokens[:, step_idx]
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x[:, current_block_start:current_block_end],
                quota,
                threshold,
            )
            updated_block = torch.where(transfer_index, x0, x[:, current_block_start:current_block_end])
            x = torch.cat((x[:, :current_block_start], updated_block, x[:, current_block_end:]), dim=1)
            nfe += 1
            step_idx += 1
    return x, nfe


@dataclass
class DLLMSamplingParams:
    gen_length: int = 128
    steps: int = 128
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    threshold: float | None = None
    decode_mode: str = "vanilla"


@dataclass
class DLLMOutput:
    prompt: str | list[int]
    generated_text: str
    token_ids: list[int]
    nfe: int


class LLaDAEngine:
    def __init__(
        self,
        model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
        *,
        tensor_parallel_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        seed: int = 0,
    ):
        if tensor_parallel_size != 1:
            raise ValueError("LLaDAEngine currently supports tensor_parallel_size=1 only.")
        torch.manual_seed(seed)
        self.model_name = model_name
        self.model = load_llada_model(
            model_name, device=device, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.mask_token_id = self.model.config.mask_token_id

    def _encode_prompt(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            return self.tokenizer(text)["input_ids"]
        return prompt

    def generate(self, prompts, sampling_params: DLLMSamplingParams | list[DLLMSamplingParams]):
        if isinstance(sampling_params, DLLMSamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        prompt_ids = [self._encode_prompt(p) for p in prompts]
        max_prompt_len = max(len(p) for p in prompt_ids)
        assert all(len(p) == max_prompt_len for p in prompt_ids), (
            "LLaDAEngine first-pass batching expects equal prompt lengths; "
            "use repeated prompts or pre-tokenized equal-length prompts."
        )
        assert all(sp == sampling_params[0] for sp in sampling_params), (
            "LLaDAEngine first-pass batching expects identical sampling params across the batch."
        )

        sp = sampling_params[0]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.model.device)
        decode_fns = {
            "vanilla": masked_diffusion_generate,
            "prefix": masked_diffusion_generate_with_prefix_cache,
            "dual": masked_diffusion_generate_with_dual_cache,
        }
        if sp.decode_mode not in decode_fns:
            raise ValueError(f"Unknown decode_mode: {sp.decode_mode}")
        full_ids, nfe = decode_fns[sp.decode_mode](
            self.model,
            prompt_tensor,
            steps=sp.steps,
            gen_length=sp.gen_length,
            block_length=sp.block_length,
            temperature=sp.temperature,
            remasking=sp.remasking,
            mask_id=self.mask_token_id,
            threshold=sp.threshold,
        )

        outputs = []
        for prompt, ids, seq in zip(prompts, prompt_ids, full_ids):
            generated = seq[len(ids):].tolist()
            outputs.append(
                DLLMOutput(
                    prompt=prompt,
                    generated_text=self.tokenizer.decode(generated, skip_special_tokens=True),
                    token_ids=generated,
                    nfe=nfe,
                )
            )
        return outputs
