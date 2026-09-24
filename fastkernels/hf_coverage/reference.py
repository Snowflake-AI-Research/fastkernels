"""Pinned Hugging Face preparation/execution; never imports audit model code."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys

from .runner import (Workload, configure_torch, digest, git_value, guarded_worker,
                     measure, symbol, to_device, versions, worker_metadata)


def verify_reference_pin(expected: str) -> dict:
    import transformers

    directory = Path(transformers.__file__).resolve().parent
    root = git_value(directory, "rev-parse", "--show-toplevel")
    if root:
        revision = git_value(directory, "rev-parse", "HEAD")
        changed = git_value(Path(root), "diff", "--name-only", "HEAD", "--", str(directory))
        if changed:
            raise RuntimeError(f"pinned Transformers source has tracked modifications: {changed}")
    else:
        text = importlib.metadata.distribution("transformers").read_text("direct_url.json")
        revision = json.loads(text or "{}").get("vcs_info", {}).get("commit_id")
    if revision != expected:
        raise RuntimeError(f"reference Transformers revision {revision!r} does not match {expected!r}")
    return {"revision": revision, "package_path": str(directory),
            "imported_version": transformers.__version__}


def resolve_generation_config(reference):
    """Read optional pinned token metadata without loading checkpoint weights."""
    source = reference.get("generation_config_source")
    inline = reference.get("generation_config")
    if source is None:
        return inline, None
    if inline is not None:
        raise ValueError("Specify generation_config or generation_config_source, not both")
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(source["repo"], "generation_config.json", revision=source["revision"]))
    return json.loads(path.read_text()), {**source, "sha256": digest(path)}


def fix_dinat_reference_layout():
    """Correct the pinned DiNAT caller's Q/K/V axes at its NATTEN boundary.

    HF passes [batch, width, height, heads, dim]; NATTEN requires
    [batch, heads, height, width, dim]. This authorized reference-only repair
    preserves native attention arithmetic and leaves the pinned checkout intact.
    """
    from transformers.models.dinat import modeling_dinat

    if getattr(modeling_dinat, "_hf_coverage_layout_fixed", False):
        return
    native_qk = modeling_dinat.natten2dqkrpb
    native_av = modeling_dinat.natten2dav

    def qk(query, key, bias, kernel_size, dilation):
        return native_qk(query.permute(0, 3, 2, 1, 4),
                         key.permute(0, 3, 2, 1, 4), bias, kernel_size, dilation)

    def av(attention, value, kernel_size, dilation):
        return native_av(attention, value.permute(0, 3, 2, 1, 4), kernel_size, dilation)

    modeling_dinat.natten2dqkrpb = qk
    modeling_dinat.natten2dav = av
    modeling_dinat._hf_coverage_layout_fixed = True


def load_reference_model(model_class, config, weights, dtype, *, load_with_base_class=False,
                         reference_backend=None, load_device="cpu", generation_config=None,
                         adapter_config=None):
    """Apply pinned HF loading rules to common weights without a checkpoint download."""
    loader = model_class.from_pretrained
    if load_with_base_class:
        from functools import partial
        from transformers import PreTrainedModel

        # TimmBackbone's override interprets the first argument as a timm model
        # name and discards the supplied config. Use HF's common-state loader.
        loader = partial(PreTrainedModel.from_pretrained.__func__, model_class)
    generation_options = {}
    if generation_config is not None:
        from transformers import GenerationConfig

        generation_options["generation_config"] = GenerationConfig(**generation_config)
    base_weights = dict(weights)
    if adapter_config is not None:
        # Common state uses native PEFT module names. Load the base through HF
        # first, then let its adapter loader construct and populate LoRA layers.
        adapter_weights = {name.replace(".default.", "."): value
                           for name, value in weights.items() if ".lora_" in name}
        base_weights = {name.replace(".base_layer.", "."): value
                        for name, value in weights.items() if ".lora_" not in name}
    model, info = loader(
        None, config=config, state_dict=base_weights, dtype=dtype,
        device_map=load_device, attn_implementation=reference_backend,
        local_files_only=True, output_loading_info=True, **generation_options,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"reference weight loading was incomplete: {info}")
    if adapter_config is not None:
        from peft import LoraConfig

        adapter_info = model.load_adapter(
            peft_config=LoraConfig(**adapter_config), adapter_state_dict=adapter_weights,
            adapter_name="default",
        )
        if adapter_info.missing_keys or adapter_info.unexpected_keys:
            raise RuntimeError(f"reference adapter loading was incomplete: {adapter_info}")
        if model.state_dict().keys() != weights.keys():
            raise RuntimeError("reference adapter state does not match the supplied common state")
        info["adapter_loading"] = {"missing_keys": [], "unexpected_keys": []}
    if config.model_type == "dinat":
        fix_dinat_reference_layout()
        info["reference_corrections"] = ["dinat_qkv_layout"]
    return model.eval(), {key: sorted(value) if isinstance(value, set) else value
                          for key, value in info.items()}


def merge_config_values(base, updates):
    """Resize nested tower configurations without discarding checkpoint settings."""
    result = dict(base)
    for name, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            value = merge_config_values(result[name], value)
        result[name] = value
    return result


def prepare(job: dict, directory: Path) -> dict:
    import torch

    pin = verify_reference_pin(job["transformers_revision"])
    configure_torch(job["seed"], "float32", gpu=False)
    case = job["case"]
    reuse_from = job.get("reuse_from")
    if reuse_from:
        source_directory = Path(reuse_from)
        source_job = json.loads((source_directory / "job.json").read_text())
        for field in ("model", "case_sha256", "dtype", "seed", "transformers_revision"):
            if source_job[field] != job[field]:
                raise ValueError(f"reuse source has a different {field}")
        source = source_directory / "prepared.pt"
        prepared = torch.load(source, map_location="cpu", weights_only=True)
        if prepared["model"] != job["model"] or prepared["case_sha256"] != job["case_sha256"]:
            raise ValueError("reuse source prepared data does not match its job")
    elif job["upcast_from"]:
        source = Path(job["upcast_from"]) / "prepared.pt"
        prepared = torch.load(source, map_location="cpu", weights_only=True)
        if prepared["model"] != job["model"] or prepared["case_sha256"] != job["case_sha256"]:
            raise ValueError("upcast source must use the same model and case settings")
        if prepared["source_dtype"] not in ("float16", "bfloat16"):
            raise ValueError("upcast source must contain low-precision values")
        from torch.utils._pytree import tree_map

        for group in ("weights", "inputs"):
            prepared[group] = tree_map(
                lambda value: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value,
                prepared[group],
            )
        prepared["upcast_source_sha256"] = digest(source)
    else:
        ref = case["reference"]
        config_cls = symbol(ref["config_class"])
        source = ref["source"]
        overrides = merge_config_values(case.get("config_overrides", {}), case.get("dimension_overrides", {}))
        if source["kind"] == "constructor_defaults":
            config = config_cls(**overrides)
        elif source["kind"] == "pinned_recipe":
            # Explicit constructor arguments from a named recipe in pinned HF
            # source, rather than an inaccessible checkpoint configuration.
            config = config_cls(**merge_config_values(source["parameters"], overrides))
        elif source["kind"] == "example_checkpoint":
            if not source.get("checkpoint") or not source.get("revision"):
                raise ValueError("example_checkpoint requires a checkpoint and pinned config revision")
            raw, _ = config_cls.get_config_dict(
                source["checkpoint"], revision=source["revision"],
                subfolder=source.get("subfolder", ""),
            )
            if source.get("config_converter"):
                # Official checkpoint formats can predate the pinned native
                # model. Use HF's own conversion before development resizing.
                raw = symbol(source["config_converter"])(raw).to_dict()
            if source.get("config_key"):
                for name in source["config_key"].split("."):
                    raw = raw[name]
            if source.get("config_property"):
                # Some native composite configs derive a child from multiple
                # serialized sections, rather than storing it under one key.
                parent = symbol(source["config_class"]).from_dict(raw)
                raw = getattr(parent, source["config_property"]).to_dict()
            config = config_cls(**merge_config_values(raw, overrides))
        elif source["kind"] == "composite_checkpoints":
            from transformers import PretrainedConfig

            raw = {}
            for name, component in source["components"].items():
                if not component.get("checkpoint") or not component.get("revision"):
                    raise ValueError("each component requires a checkpoint and pinned config revision")
                raw[name], _ = PretrainedConfig.get_config_dict(
                    component["checkpoint"], revision=component["revision"],
                )
            config = config_cls(**merge_config_values(raw, overrides))
        else:
            raise ValueError(f"unsupported configuration source {source['kind']!r}")
        config._attn_implementation = case.get("reference_backend", "eager")
        model_class = symbol(ref["model_class"])
        dtype = getattr(torch, job["dtype"])
        source_weights = job.get("state_dict")
        if ref.get("adapter_config") is not None and not source_weights:
            raise ValueError("This adapter case requires an explicit common state_dict including LoRA weights")
        serialized_suffixes = ref.get("serialized_weight_suffixes", [])
        if serialized_suffixes and not source_weights:
            raise ValueError("This quantized case requires an explicit serialized common state_dict")
        if source_weights:
            weights = torch.load(source_weights, map_location="cpu", weights_only=True)
            source_weights_sha256 = digest(Path(source_weights))
            initialization = "Explicit common HF-format state_dict converted by pinned HF loading rules"
        else:
            model = model_class(config).eval()
            weights = dict(model.state_dict())
            source_weights_sha256 = None
            initialization = "HF FP32 initialization converted by pinned HF from_pretrained loading rules"
            # Declared subtrees use unit fan-in variance when native random
            # initialization hides computation or produces invalid generation.
            # Supplied weights and non-matrix parameters are never changed.
            fan_in_stds = {}
            embedding_weights = {layer.weight.data_ptr() for layer in model.modules()
                                 if isinstance(layer, torch.nn.Embedding)}
            weight_generator = torch.Generator().manual_seed(job["seed"] + 3)
            for prefix in ref.get("fan_in_normal_modules", []):
                for name, layer in model.get_submodule(prefix).named_modules():
                    if isinstance(layer, torch.nn.Linear):
                        fan_in = layer.in_features
                    elif isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d)):
                        fan_in = layer.in_channels * math.prod(layer.kernel_size) / layer.groups
                        if isinstance(layer, torch.nn.ConvTranspose1d):
                            fan_in /= layer.stride[0]
                    else:
                        continue
                    if layer.weight.data_ptr() in embedding_weights:
                        continue  # Preserve embeddings and their tied output heads.
                    key = ".".join(part for part in (prefix, name, "weight") if part)
                    std = fan_in ** -0.5
                    weights[key].normal_(std=std, generator=weight_generator)
                    fan_in_stds[key] = std
            if fan_in_stds:
                initialization += "; matrix fan-in normal, seed+3: " + json.dumps(fan_in_stds, sort_keys=True)
            del model
            # Some HF constructors zero gates and hide otherwise enabled branches.
            # Randomize only explicitly named zero parameters, identically on both sides.
            names = ref.get("randomize_zero_parameters", [])
            gate_generator = torch.Generator().manual_seed(job["seed"] + 2)
            for name in names:
                if torch.count_nonzero(weights[name]).item():
                    raise ValueError(f"Expected zero-initialized parameter: {name}")
                weights[name].normal_(mean=0.0, std=0.2, generator=gate_generator)
            if names:
                initialization += "; named zero parameters use shared N(0, 0.2^2), seed+2: " + ", ".join(names)
            if ref.get("csm_equal_codebook_embeddings"):
                # Official CSM conversion copies equal audio embeddings into
                # separate tables. Inference requires equal values, not aliasing.
                source = "backbone_model.embed_tokens.embed_audio_tokens.weight"
                target = "depth_decoder.model.embed_tokens.weight"
                weights[target].copy_(weights[source])
                spec = ref["randomize_zero_buffers"]
                buffer_names = [name for name in weights
                                if name.startswith(spec["prefix"]) and name.endswith(spec["suffix"])]
                if len(buffer_names) != config.num_codebooks:
                    raise ValueError("CSM preparation requires every codec centroid buffer")
                buffer_generator = torch.Generator().manual_seed(job["seed"] + spec["seed_offset"])
                for name in buffer_names:
                    if torch.count_nonzero(weights[name]).item():
                        raise ValueError(f"Expected zero-initialized codec buffer: {name}")
                    weights[name].normal_(std=spec["std"], generator=buffer_generator)
                initialization += "; CSM converter equal-value audio embedding copy; codec buffers: " + json.dumps({
                    "names": buffer_names, "normal_std": spec["std"],
                    "seed": job["seed"] + spec["seed_offset"],
                }, sort_keys=True)
        generation_config, generation_record = resolve_generation_config(ref)
        model, loading_info = load_reference_model(
            model_class, config, weights, dtype,
            load_with_base_class=ref.get("load_with_base_class", False),
            reference_backend=case.get("reference_backend", "eager"),
            load_device=ref.get("load_device", "cpu"),
            generation_config=generation_config,
            adapter_config=ref.get("adapter_config"),
        )
        if generation_record is not None:
            loading_info["generation_config_source"] = generation_record
        serialized_weights = {key: value for key, value in weights.items()
                              if any(key.endswith(suffix) for suffix in serialized_suffixes)}
        # Native MXFP4 deserialization stores packed tensors outside Parameters;
        # retain their original serialized form for the second native load.
        weights = to_device(dict(model.state_dict()), "cpu")
        for key, value in serialized_weights.items():
            if key not in weights:
                weights[key] = value.cpu()
        del model
        spec = case["input"]
        def input_dtype(name):
            return getattr(torch, spec.get("dtypes", {}).get(name, job["dtype"]))

        generator = torch.Generator().manual_seed(job["seed"] + 1)
        source_inputs = job.get("input_dict")
        source_inputs_sha256 = None
        if source_inputs:
            from torch.utils._pytree import tree_map

            inputs = torch.load(source_inputs, map_location="cpu", weights_only=True)
            if not isinstance(inputs, dict) or not inputs:
                raise ValueError("--input-dict must contain a nonempty dictionary of model inputs")
            inputs = {name: tree_map(
                lambda value: value.to(input_dtype(name))
                if isinstance(value, torch.Tensor) and value.is_floating_point() else value,
                entry,
            ) for name, entry in inputs.items()}
            source_inputs_sha256 = digest(Path(source_inputs))
        elif spec["kind"] == "supplied":
            raise ValueError("this workload requires processor-prepared inputs via --input-dict")
        elif spec["kind"] == "musicgen_forward":
            batch, length = spec.get("batch_size", 1), spec["sequence_length"]
            ids = torch.randint(config.text_encoder.vocab_size, (batch, length), generator=generator)
            decoder_ids = torch.randint(config.decoder.vocab_size,
                                        (batch * config.decoder.num_codebooks, spec["decoder_sequence_length"]),
                                        generator=generator)
            decoder_ids[:, 0] = config.decoder.bos_token_id
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                      "decoder_input_ids": decoder_ids}
        elif spec["kind"] == "moshi_forward":
            batch, length = spec.get("batch_size", 1), spec["sequence_length"]
            ids = torch.randint(config.vocab_size, (batch, length), generator=generator)
            ids[:, 0] = config.vocab_size
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            for name in ("user_audio_codes", "moshi_audio_codes"):
                codes = torch.randint(config.audio_vocab_size, (batch, config.num_codebooks, length),
                                      generator=generator)
                codes[:, :, 0] = config.audio_vocab_size
                inputs[name] = codes
        elif spec["kind"] == "pi0":
            camera_mask = torch.tensor(spec["camera_mask"], dtype=torch.bool)
            batch, cameras = camera_mask.shape
            vision, text = config.vlm_config.vision_config, config.vlm_config.text_config
            ids = torch.randint(3, text.vocab_size, (batch, spec["sequence_length"]), generator=generator)
            patches = (vision.image_size // vision.patch_size) ** 2
            for row, mask in enumerate(camera_mask):
                image_tokens = int(mask.sum()) * patches
                if image_tokens >= ids.shape[1]:
                    raise ValueError("PI0 input must include image positions and instruction tokens")
                ids[row, :image_tokens] = config.vlm_config.image_token_index
            inputs = {
                "input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_attention_mask": camera_mask,
                "pixel_values": torch.randn(batch, cameras, 3, vision.image_size, vision.image_size,
                                              generator=generator).to(input_dtype("pixel_values")),
                "state": torch.randn(batch, config.max_state_dim, generator=generator).to(input_dtype("state")),
                "noise": torch.randn(batch, config.chunk_size, config.max_action_dim,
                                     generator=generator).to(input_dtype("noise")),
            }
        elif spec["kind"] == "tokens":
            shape = (spec.get("batch_size", 1), spec["sequence_length"])
            vocabulary_size = spec["vocab_size"] if "vocab_size" in spec else config.vocab_size
            if "input_ids" in spec:
                ids = torch.tensor(spec["input_ids"], dtype=torch.long)
                if tuple(ids.shape) != shape:
                    raise ValueError("declared token IDs do not match the input shape")
            else:
                ids = torch.randint(vocabulary_size, shape, generator=generator)
            inputs = {"input_ids": ids}
            if "attention_mask_values" in spec:
                mask = torch.tensor(spec["attention_mask_values"], dtype=torch.long)
                if tuple(mask.shape) != shape or not torch.all((mask == 0) | (mask == 1)):
                    raise ValueError("declared attention mask must be binary and match the token shape")
                inputs["attention_mask"] = mask
            if "visual_region_count" in spec:
                regions = spec["visual_region_count"]
                inputs["visual_feats"] = torch.randn(
                    shape[0], regions, config.visual_feat_dim, generator=generator,
                ).to(input_dtype("visual_feats"))
                if config.visual_pos_dim != 4:
                    raise ValueError("visual regions require four normalized box coordinates")
                corners = torch.rand(shape[0], regions, 2, generator=generator)
                extent = torch.rand(shape[0], regions, 2, generator=generator) * (1 - corners)
                inputs["visual_pos"] = torch.cat((corners, corners + extent), dim=-1).to(
                    input_dtype("visual_pos"))
            for name, vocabulary_key in spec.get("auxiliary_fields", {}).items():
                inputs[name] = torch.randint(getattr(config, vocabulary_key), shape,
                                             generator=generator)
            if "entity_mentions" in spec:
                mentions = spec["entity_mentions"]
                positions = torch.full((shape[0], len(mentions), max(map(len, mentions))), -1)
                for index, mention in enumerate(mentions):
                    if not mention or min(mention) < 0 or max(mention) >= shape[1]:
                        raise ValueError("entity mentions must name valid input token positions")
                    positions[:, index, :len(mention)] = torch.tensor(mention)
                inputs["entity_position_ids"] = positions
                inputs["entity_ids"] = torch.randint(1, config.entity_vocab_size,
                    (shape[0], len(mentions)), generator=generator)
            if "xpath_depth" in spec:
                lengths = torch.randint(1, spec["xpath_depth"] + 1, shape, generator=generator)
                padding = torch.arange(config.max_depth) >= lengths[..., None]
                for name, size, pad in (
                    ("xpath_tags_seq", config.max_xpath_tag_unit_embeddings, config.tag_pad_id),
                    ("xpath_subs_seq", config.max_xpath_subs_unit_embeddings, config.subs_pad_id),
                ):
                    ids = torch.randint(size - 1, (*shape, config.max_depth), generator=generator)
                    ids += ids >= pad
                    inputs[name] = ids.masked_fill(padding, pad)
            if "table_columns" in spec:
                prefix = spec["query_length"]
                columns = spec["table_columns"]
                cells = torch.arange(shape[1] - prefix) // spec["tokens_per_cell"]
                rows = cells // columns + 1
                types = torch.zeros((*shape, 7), dtype=torch.long)
                types[:, prefix:, 0] = 1
                types[:, prefix:, 1] = cells % columns + 1
                types[:, prefix:, 2] = rows
                types[:, prefix:, 4] = rows
                types[:, prefix:, 5] = rows.max() - rows + 1
                if any(int(types[..., i].max()) >= size for i, size in enumerate(config.type_vocab_sizes)):
                    raise ValueError("table input exceeds the configured token-type vocabularies")
                inputs["token_type_ids"] = types
            if "encoder_sequence_length" in spec:
                width = getattr(config, "cross_attention_hidden_size", None) or config.d_model
                inputs["encoder_hidden_states"] = torch.randn(
                    shape[0], spec["encoder_sequence_length"], width,
                    generator=generator, dtype=torch.float32,
                ).to(input_dtype("encoder_hidden_states"))
            if "visual_sequence_length" in spec:
                inputs["visual_embeds"] = torch.randn(
                    shape[0], spec["visual_sequence_length"], config.visual_embedding_dim,
                    generator=generator, dtype=torch.float32,
                ).to(input_dtype("visual_embeds"))
            if case["workload"] == "causal_lm" and shape[1] < 2:
                raise ValueError("causal workload requires a prompt and at least one decode token")
        elif spec["kind"] == "trajectory":
            batch, length = spec["batch_size"], spec["sequence_length"]
            inputs = {
                name: torch.randn(batch, length, width, generator=generator).to(input_dtype(name))
                for name, width in (("states", config.state_dim), ("actions", config.act_dim),
                                    ("returns_to_go", 1))
            }
            inputs["timesteps"] = torch.arange(length).expand(batch, -1).clone()
        elif spec["kind"] == "document":
            shape = (spec["batch_size"], spec["sequence_length"])
            origins = torch.randint(901, (*shape, 2), generator=generator)
            sizes = torch.randint(1, 101, (*shape, 2), generator=generator)
            inputs = {
                "input_ids": torch.randint(config.vocab_size, shape, generator=generator),
                "bbox": torch.cat((origins, origins + sizes), dim=-1),
            }
            if spec.get("attention_mask"):
                inputs["attention_mask"] = torch.ones(shape, dtype=torch.long)
            if spec.get("normalized_bbox"):
                inputs["bbox"] = (inputs["bbox"].float() / 1000).to(input_dtype("bbox"))
            if spec.get("image_shape"):
                image_shape = (spec["batch_size"], *spec["image_shape"])
                if "image_value_range" in spec:
                    low, high = spec["image_value_range"]
                    pixels = torch.rand(image_shape, generator=generator) * (high - low) + low
                else:
                    pixels = torch.randn(image_shape, generator=generator)
                image_name = spec.get("image_input_name", "pixel_values")
                inputs[image_name] = pixels.to(input_dtype(image_name))
            if spec.get("decoder_length"):
                start = spec.get("decoder_start_token_id", config.decoder_start_token_id)
                decoder_ids = torch.randint(
                    config.vocab_size, (spec["batch_size"], spec["decoder_length"]), generator=generator,
                )
                decoder_ids[:, 0] = start
                inputs["decoder_input_ids"] = decoder_ids
        elif spec["kind"] == "seq2seq_tokens":
            decoder_start = spec.get("decoder_start_token_id", getattr(config, "decoder_start_token_id", None))
            if decoder_start is None:
                raise ValueError("sequence-to-sequence case requires a resolved decoder start token")
            bos_token_id = getattr(config, "bos_token_id", None)
            prefix = spec.get("encoder_prefix_token_ids", [] if bos_token_id is None else [bos_token_id])
            suffix = spec.get("encoder_suffix_token_ids", [] if config.eos_token_id is None else [config.eos_token_id])
            special_ids = {config.pad_token_id, bos_token_id,
                           config.eos_token_id, decoder_start, *prefix, *suffix}
            special_ids.update(spec.get("fixed_token_ids", {}).values())
            special_ids.update(spec.get("decoder_fixed_token_ids", {}).values())
            if "image_token_positions" in spec:
                special_ids.add(config.image_token_index)
            inputs = {}
            for name, length, tower in (("input_ids", "encoder_sequence_length", "encoder"),
                                        ("decoder_input_ids", "decoder_sequence_length", "decoder")):
                vocabulary_key = "src_vocab_size" if tower == "encoder" else "tgt_vocab_size"
                if tower + "_vocab_size" in spec:
                    vocabulary_size = spec[tower + "_vocab_size"]
                elif hasattr(config, vocabulary_key):
                    vocabulary_size = getattr(config, vocabulary_key)
                else:
                    vocabulary_size = (config.vocab_size if hasattr(config, "vocab_size")
                                       else getattr(config, tower).vocab_size)
                vocabulary = torch.tensor([
                    token for token in range(spec.get("content_token_id_min", 0), vocabulary_size)
                    if token not in special_ids
                ])
                shape = (spec["batch_size"], spec[length])
                if tower == "decoder" and "decoder_channels" in spec:
                    shape += (spec["decoder_channels"],)
                inputs[name] = vocabulary[torch.randint(
                    vocabulary.numel(), shape, generator=generator,
                )]
            if prefix:
                inputs["input_ids"][:, :len(prefix)] = torch.tensor(prefix)
            if suffix:
                inputs["input_ids"][:, -len(suffix):] = torch.tensor(suffix)
            inputs["decoder_input_ids"][:, 0] = decoder_start
            for field, name in (("fixed_token_ids", "input_ids"),
                                ("decoder_fixed_token_ids", "decoder_input_ids")):
                for position, token in spec.get(field, {}).items():
                    inputs[name][:, int(position)] = token
            if "image_shape" in spec:
                inputs["pixel_values"] = torch.randn(
                    spec["batch_size"], *spec["image_shape"], generator=generator,
                ).to(input_dtype("pixel_values"))
                inputs["input_ids"][:, spec["image_token_positions"]] = config.image_token_index
            if spec.get("attention_mask"):
                inputs["attention_mask"] = (inputs["input_ids"] != config.pad_token_id).long()
            if spec.get("global_first_token"):
                inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
                inputs["global_attention_mask"][:, 0] = 1
        elif spec["kind"] in ("text_image", "text_audio"):
            text_config = getattr(config, "text_config", None)
            if text_config is None:
                text_config = config.get_text_config()
            inputs = {
                "input_ids": torch.randint(
                    text_config.vocab_size,
                    (spec["text_batch_size"], spec["sequence_length"]), generator=generator,
                ),
                spec.get("image_input_name", "pixel_values"): torch.randn(
                    spec["image_batch_size"], *spec["shape"], generator=generator,
                    dtype=torch.float32,
                ).to(input_dtype(spec.get("image_input_name", "pixel_values"))),
            }
            if "high_res_shape" in spec:
                inputs["high_res_pixel_values"] = torch.randn(
                    spec["image_batch_size"], *spec["high_res_shape"], generator=generator,
                ).to(input_dtype("high_res_pixel_values"))
            if "qformer_sequence_length" in spec:
                inputs["qformer_input_ids"] = torch.randint(
                    1, config.qformer_config.vocab_size,
                    (spec["text_batch_size"], spec["qformer_sequence_length"]), generator=generator,
                )
            if "audio_shape" in spec:
                inputs["input_features"] = torch.randn(
                    spec["text_batch_size"], *spec["audio_shape"], generator=generator,
                ).to(input_dtype("input_features"))
            if "video_second_per_grid" in spec:
                inputs["video_second_per_grid"] = torch.tensor(spec["video_second_per_grid"], dtype=torch.float32)
            for lengths, mask_name in (("input_features_lengths", "input_features_mask"),
                                       ("feature_attention_lengths", "feature_attention_mask")):
                if lengths in spec:
                    inputs[mask_name] = (
                        torch.arange(spec.get("audio_shape", spec["shape"])[spec.get("audio_time_axis", -1)])[None, :]
                        < torch.tensor(spec[lengths])[:, None]
                    ).to(getattr(torch, spec.get("dtypes", {}).get(mask_name, "long")))
            if "video_shape" in spec:
                inputs["pixel_values_videos"] = torch.randn(
                    spec["image_batch_size"], *spec["video_shape"], generator=generator,
                    dtype=torch.float32,
                ).to(input_dtype("pixel_values_videos"))
            if "image_sizes" in spec:
                inputs["image_sizes"] = torch.tensor(spec["image_sizes"], dtype=torch.long)
            if "image_attention_mask" in spec:
                inputs["image_attention_mask"] = torch.tensor(spec["image_attention_mask"], dtype=torch.bool)
            for name in ("image_grid_thw", "image_merge_sizes", "video_grid_thw",
                         "video_merge_sizes", "video_compression_mask", "image_position_ids",
                         "target_sizes", "target_sizes_videos", "moe_mm_token_type_ids",
                         "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"):
                if name in spec:
                    inputs[name] = torch.tensor(spec[name], dtype=(torch.bool
                        if name == "video_compression_mask" else torch.long))
            if "second_per_grid_ts" in spec:
                inputs["second_per_grid_ts"] = torch.tensor(spec["second_per_grid_ts"], dtype=torch.float32)
            if spec.get("flatten_pixel_batch"):
                image_name = spec.get("image_input_name", "pixel_values")
                inputs[image_name] = inputs[image_name].flatten(0, 1)
                if "pixel_values_videos" in inputs:
                    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].flatten(0, 1)
            if "spatial_shapes" in spec:
                spatial_shapes = torch.tensor(spec["spatial_shapes"], dtype=torch.long)
                inputs["spatial_shapes"] = spatial_shapes
                inputs["pixel_attention_mask"] = (
                    torch.arange(spec["shape"][0])[None, :] < spatial_shapes.prod(dim=-1)[:, None]
                ).long()
            if any(field in spec for field in ("image_token_positions", "video_token_positions", "audio_token_positions")):
                image_token = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
                if image_token is None and hasattr(config, "vlm_config"):
                    image_token = getattr(config.vlm_config, "image_token_id",
                                          getattr(config.vlm_config, "image_token_index", None))
                video_token = getattr(config, "video_token_id", getattr(config, "video_token_index", None))
                audio_token = getattr(config, "audio_token_id", None)
                ids = inputs["input_ids"]
                prompt_length = ids.shape[1] - (case["workload"] == "causal_lm")
                vocabulary = torch.tensor([token for token in range(text_config.vocab_size)
                                           if token not in {image_token, video_token, audio_token}])
                ids.copy_(vocabulary[torch.randint(vocabulary.numel(), ids.shape, generator=generator)])
                for field, token in (("image_token_positions", image_token), ("video_token_positions", video_token),
                                     ("audio_token_positions", audio_token)):
                    if field in spec:
                        positions = spec[field]
                        if token is None or any(position < 0 or position >= prompt_length for position in positions):
                            raise ValueError("modality placeholders require a configured token and valid prompt positions")
                        ids[:, positions] = token
            for position, token in spec.get("fixed_token_ids", {}).items():
                inputs["input_ids"][:, int(position)] = token
            if spec.get("image_token_type_ids"):
                inputs["token_type_ids"] = (inputs["input_ids"] == config.image_token_id).long()
            if spec.get("mm_token_type_ids"):
                token_types = torch.zeros_like(inputs["input_ids"])
                for name, value in (("image_token_positions", 1), ("video_token_positions", 2)):
                    token_types[:, spec.get(name, [])] = value
                inputs["mm_token_type_ids"] = token_types
            if "token_type_id" in spec:
                inputs["token_type_ids"] = torch.full_like(inputs["input_ids"], spec["token_type_id"])
            if "image_embeds_positions" in spec:
                mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
                mask[:, spec["image_embeds_positions"]] = True
                inputs["image_embeds_position_mask"] = mask
            if "eos_positions" in spec:
                text = text_config
                ids = inputs["input_ids"]
                positions = spec["eos_positions"]
                bos = spec.get("bos_token_id", getattr(text, "bos_token_id", None))
                pad = spec.get("pad_token_id", getattr(text, "pad_token_id", None))
                eos = spec.get("eos_token_id", text.eos_token_id)
                if len(positions) != ids.shape[0] or any(not 0 < p < ids.shape[1] for p in positions):
                    raise ValueError("eos_positions must specify one interior or final position per text row")
                special_ids = {bos, eos, pad}
                vocabulary = torch.tensor([token for token in range(text.vocab_size)
                                           if token not in special_ids])
                ids.copy_(vocabulary[torch.randint(vocabulary.numel(), ids.shape, generator=generator)])
                for row, position in enumerate(positions):
                    if bos is not None:
                        ids[row, 0] = bos
                    ids[row, position] = eos
                    if pad is not None:
                        ids[row, position + 1:] = pad
            if "input_ids" in spec:
                # Tokenized task examples can exercise punctuation-dependent
                # attention blocks that random token IDs would miss.
                explicit_ids = torch.tensor(spec["input_ids"], dtype=torch.long)
                if explicit_ids.shape != inputs["input_ids"].shape:
                    raise ValueError("explicit input_ids must match the declared text shape")
                inputs["input_ids"] = explicit_ids
            if spec.get("attention_mask"):
                ids = inputs["input_ids"]
                if "eos_positions" in spec:
                    # Some processors use EOS itself as padding. The first
                    # EOS belongs to the sequence; subsequent padding does not.
                    inputs["attention_mask"] = (
                        torch.arange(ids.shape[1])[None, :] <= torch.tensor(spec["eos_positions"])[:, None]
                    ).long()
                else:
                    pad = spec.get("pad_token_id", getattr(text_config, "pad_token_id", None))
                    if pad is None:
                        raise ValueError("text attention masks require a padding token or explicit EOS positions")
                    inputs["attention_mask"] = (ids != pad).long()
        elif spec["kind"] == "timeseries_list":
            inputs = {"past_values": [torch.randn(length, generator=generator).to(input_dtype("past_values"))
                                      for length in spec["lengths"]]}
            if "frequencies" in spec:
                if len(spec["frequencies"]) != len(spec["lengths"]):
                    raise ValueError("each time series requires one frequency category")
                inputs["freq"] = torch.tensor(spec["frequencies"], dtype=torch.long)
        elif spec["kind"] in ("image", "video", "spectrogram", "waveform", "continuous"):
            name = spec.get("name", "pixel_values" if spec["kind"] in ("image", "video") else "input_values")
            inputs = {name: torch.randn(spec.get("batch_size", 1), *spec["shape"],
                                       generator=generator, dtype=torch.float32).to(input_dtype(name))}
            if spec.get("segmentation_prompt"):
                inputs["prompt_pixel_values"] = torch.randn(
                    *inputs[name].shape, generator=generator,
                ).to(input_dtype("prompt_pixel_values"))
                # Synthetic foreground/background prompt, normalized as the
                # native SegGPT processor normalizes a repeated RGB mask.
                mask = torch.zeros(inputs[name].shape, dtype=torch.float32)
                height, width = mask.shape[-2:]
                mask[:, :, height // 4:3 * height // 4, width // 4:3 * width // 4] = 1
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                inputs["prompt_masks"] = ((mask - mean) / std).to(input_dtype("prompt_masks"))
            if "task_inputs" in spec:
                inputs["task_inputs"] = torch.tensor(spec["task_inputs"], dtype=input_dtype("task_inputs"))
            if "attention_mask_length" in spec:
                inputs["attention_mask"] = torch.ones(
                    spec.get("batch_size", 1), spec["attention_mask_length"], dtype=torch.long,
                )
            if "prompt_depth_shape" in spec:
                inputs["prompt_depth"] = (torch.rand(
                    spec.get("batch_size", 1), *spec["prompt_depth_shape"], generator=generator,
                ) * 10).to(input_dtype("prompt_depth"))
            if "input_points" in spec:
                inputs["input_points"] = torch.tensor(spec["input_points"], dtype=input_dtype("input_points"))
            if "input_labels" in spec:
                inputs["input_labels"] = torch.tensor(spec["input_labels"], dtype=torch.long)
            if "noise_shape" in spec:
                inputs["noise"] = torch.rand(*spec["noise_shape"], generator=generator)
            if "noise_sequence_shape" in spec:
                inputs["noise_sequence"] = torch.randn(
                    *spec["noise_sequence_shape"], generator=generator,
                ).to(input_dtype("noise_sequence"))
            if "decoder_sequence_length" in spec:
                decoder = getattr(config, "decoder", getattr(config, "text_config", config))
                start = spec.get("decoder_start_token_id", getattr(config, "decoder_start_token_id", None))
                if start is None:
                    raise ValueError("decoder inputs require a resolved start token")
                special_ids = {start, decoder.pad_token_id, decoder.bos_token_id, decoder.eos_token_id}
                vocabulary = torch.tensor([token for token in range(decoder.vocab_size)
                                           if token not in special_ids])
                decoder_ids = vocabulary[torch.randint(
                    vocabulary.numel(), (spec.get("batch_size", 1), spec["decoder_sequence_length"]),
                    generator=generator,
                )]
                decoder_ids[:, 0] = start
                inputs["decoder_input_ids"] = decoder_ids
        else:
            raise ValueError(f"input kind {spec['kind']!r} is not implemented")
        if "patch_grid_columns" in spec and not source_inputs:
            # Document processors prepend one-based grid coordinates to each
            # flattened image patch. They are positions, not image values.
            patches = inputs["flattened_patches"]
            indices = torch.arange(patches.shape[-2])
            patches[..., 0] = indices // spec["patch_grid_columns"] + 1
            patches[..., 1] = indices % spec["patch_grid_columns"] + 1
        for name, dtype_name in spec.get("dtypes", {}).items():
            from torch.utils._pytree import tree_map

            inputs[name] = tree_map(lambda value: value.to(getattr(torch, dtype_name)), inputs[name])
        prepared = {"model": job["model"], "case_sha256": job["case_sha256"],
                    "config": config.to_dict(), "weights": weights, "inputs": inputs,
                    "source_dtype": job["dtype"],
                    "initialization": initialization, "source_weights_sha256": source_weights_sha256,
                    "source_inputs_sha256": source_inputs_sha256,
                    "loading_info": loading_info}
    if reuse_from:
        # Preparation files are immutable. A hard link retains the exact bytes
        # without duplicating large weights; both run directories must share a filesystem.
        (directory / "prepared.pt").hardlink_to(source)
    else:
        torch.save(prepared, directory / "prepared.pt")
    return {"status": "completed", "reference_pin": pin, "versions": versions(),
            "config": prepared["config"], "initialization": prepared["initialization"],
            "loading_info": prepared.get("loading_info"),
            "source_weights_sha256": prepared.get("source_weights_sha256"),
            "source_inputs_sha256": prepared.get("source_inputs_sha256"),
            "source_dtype": prepared["source_dtype"], "prepared_sha256": digest(directory / "prepared.pt"),
            "upcast_source_sha256": prepared.get("upcast_source_sha256"),
            "reused_prepared_from": str(source) if reuse_from else None}


def reference_workloads(model, inputs, case) -> dict[str, Workload]:
    import torch

    keys = case.get("outputs", ["logits"])
    extra = dict(case["reference"].get("forward_kwargs", {}))

    def select(output, output_keys=None, *, retained_cache=None):
        selected = {}

        def flatten(name, value):
            if isinstance(value, dict):
                for child, tensor in value.items():
                    flatten(f"{name}.{child}", tensor)
            elif isinstance(value, (tuple, list)):
                for index, tensor in enumerate(value):
                    flatten(f"{name}.{index}", tensor)
            else:
                selected[name] = value

        for key in keys if output_keys is None else output_keys:
            if retained_cache is not None and key == case["reference"].get("cache_output", "past_key_values"):
                value = retained_cache
            else:
                value = output
                for part in key.split("."):
                    value = getattr(value, part)
            if key.split(".")[-1].endswith(("past_key_values", "cache_params")):
                from transformers.cache_utils import (CacheLayerMixin, DynamicCache, EncoderDecoderCache,
                                                       LinearAttentionCacheLayerMixin, StaticCache)

                if isinstance(value, EncoderDecoderCache):
                    for kind, cache in (("self", value.self_attention_cache),
                                        ("cross", value.cross_attention_cache)):
                        for index, layer in enumerate(cache.layers):
                            for name, tensor in (("key", layer.keys), ("value", layer.values)):
                                if tensor is not None:
                                    selected[f"{key}.{index}.{kind}.{name}"] = tensor
                elif isinstance(value, (DynamicCache, StaticCache)):
                    for index, layer in enumerate(value.layers):
                        if isinstance(layer, LinearAttentionCacheLayerMixin):
                            for state_name in ("conv_states", "recurrent_states"):
                                state = getattr(layer, state_name)
                                if state is not None:
                                    history = case["reference"].get("conv_cache_history")
                                    if state_name == "conv_states" and history is not None:
                                        # Native Mamba retains one extra oldest sample that
                                        # its next update discards before convolution.
                                        if not 0 < history <= state.shape[-1]:
                                            raise ValueError("invalid logical convolution history length")
                                        state = state[..., -history:]
                                    selected[f"{key}.{index}.{state_name}"] = state
                        if isinstance(layer, CacheLayerMixin):
                            # A hybrid layer can carry both recurrent state and
                            # attention K/V. Empty attention slots have no tensors.
                            for name, tensor in (("key", layer.keys), ("value", layer.values)):
                                if tensor is not None:
                                    selected[f"{key}.{index}.{name}"] = tensor
                    # MiniMax stores recurrent matrices beside its K/V layers.
                    # Empty list entries mark layers without recurrent state.
                    for index, state in enumerate(getattr(value, "linear_cache", ())):
                        if isinstance(state, torch.Tensor):
                            selected[f"{key}.{index}.recurrent_states"] = state
                else:
                    raise TypeError("cache selection requires a dynamic, static, or encoder-decoder K/V cache")
            else:
                flatten(key, value)
        return selected

    if case["workload"] in ("sam2_video", "edgetam_video", "sam3_tracker_video"):
        if case["workload"] == "edgetam_video":
            from transformers.models.edgetam_video.modeling_edgetam_video import EdgeTamVideoInferenceSession as VideoSession
        elif case["workload"] == "sam3_tracker_video":
            from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoInferenceSession as VideoSession
        else:
            from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession as VideoSession

        state = {}
        video = inputs["video"]

        def prepare_video():
            session = VideoSession(
                video=video, video_height=video.shape[-2], video_width=video.shape[-1],
                inference_device=video.device, inference_state_device=video.device,
                video_storage_device=video.device, dtype=video.dtype,
            )
            object_index = session.obj_id_to_idx(1)
            session.add_point_inputs(object_index, 0, {
                "point_coords": inputs["input_points"], "point_labels": inputs["input_labels"],
            })
            session.obj_with_new_inputs.append(1)
            state["session"] = session

        def run_video():
            session, outputs = state["session"], {}
            for index in range(video.shape[0]):
                output = model(inference_session=session, frame_idx=index)
                if output.object_ids != [1] or output.frame_idx != index:
                    raise ValueError("SAM2 video returned unexpected object or frame identities")
                for name in ("pred_masks", "object_score_logits"):
                    outputs[f"frame{index}.{name}"] = getattr(output, name)
                bucket = "cond_frame_outputs" if index == 0 else "non_cond_frame_outputs"
                frame_state = session.output_dict_per_obj[0][bucket][index]
                state_names = ["pred_masks", "object_pointer", "object_score_logits",
                               "maskmem_features", "maskmem_pos_enc"]
                if case["workload"] in ("sam2_video", "sam3_tracker_video"):
                    state_names.append("high_res_masks")
                for name in state_names:
                    outputs[f"frame{index}.state.{name}"] = frame_state[name]
            return outputs

        return {"forward": Workload(run=run_video, prepare=prepare_video)}
    if case["workload"] == "sample_actions":
        return {"sample_actions": Workload(run=lambda: {"actions": model.sample_actions(**inputs)})}
    if case["workload"] == "generate":
        if model.config.model_type == "csm":
            def generate_csm():
                depth_logits = []
                handle = model.depth_decoder.register_forward_hook(
                    lambda module, args, output: depth_logits.append(output.logits))
                try:
                    output = model.generate(**inputs, **case["generation_kwargs"],
                                            return_dict_in_generate=True, output_logits=True)
                finally:
                    handle.remove()
                codebooks = model.config.num_codebooks - 1
                if len(depth_logits) != output.sequences.shape[1] * codebooks:
                    raise ValueError("CSM depth decoder did not execute every codebook step")
                result = {
                    "sequences": output.sequences,
                    "logits": torch.stack(output.logits, dim=1),
                    "depth_logits": torch.stack([
                        torch.cat(depth_logits[start:start + codebooks], dim=1)
                        for start in range(0, len(depth_logits), codebooks)
                    ], dim=1),
                    "audio_values": output.audio[0][None, None],
                }
                for index, layer in enumerate(output.past_key_values.layers):
                    result[f"past_key_values.{index}.key"] = layer.keys
                    result[f"past_key_values.{index}.value"] = layer.values
                return result
            return {"generate": Workload(run=generate_csm)}
        output_names = case["reference"].get("generation_output_names")
        if output_names is not None:
            def generate_tuple():
                values = model.generate(**inputs, **case["generation_kwargs"])
                if not isinstance(values, tuple) or len(values) != len(output_names):
                    raise ValueError("native generation outputs do not match the declared tuple")
                return dict(zip(output_names, values, strict=True))
            return {"generate": Workload(run=generate_tuple)}
        output_name = case["reference"].get("generation_output_name")
        if output_name is not None:
            return {"generate": Workload(run=lambda: {
                output_name: model.generate(**inputs, **case["generation_kwargs"]),
            })}
        return {"generate": Workload(run=lambda: select(model.generate(
            **inputs, **case["generation_kwargs"],
            return_dict_in_generate=True, output_logits=True,
        )))}
    if case["workload"] in ("causal_lm_continuation", "seq2seq_continuation", "memory_continuation"):
        encoder_decoder = case["workload"] == "seq2seq_continuation"
        memory_continuation = case["workload"] == "memory_continuation"
        if encoder_decoder:
            from transformers.modeling_outputs import BaseModelOutput
            encoder_output_type = BaseModelOutput
            if model.config.model_type == "moonshine":
                from transformers.models.moonshine.modeling_moonshine import MoonshineEncoderModelOutput
                if "attention_mask" in inputs:
                    raise ValueError("Moonshine continuation currently requires an unpadded waveform")
                # Moonshine reads the encoder's downsampled attention mask.
                # The declared unpadded workload has no such mask.
                encoder_output_type = MoonshineEncoderModelOutput
            elif model.config.model_type == "switch_transformers":
                from transformers.modeling_outputs import MoEModelOutputWithPastAndCrossAttentions
                # Its forward also reads router_logits, which defaults to None
                # when the selected configuration does not request them.
                encoder_output_type = MoEModelOutputWithPastAndCrossAttentions
        ids = inputs["decoder_input_ids" if encoder_decoder else "input_ids"]
        if ids.shape[1] < 3:
            raise ValueError("continuation requires a prefix and two new tokens")
        prefix_length = ids.shape[1] - 2
        ref = case["reference"]
        reuse_encoder = encoder_decoder and ref.get("reuse_encoder", True)
        cache_argument = ref.get("cache_argument", "past_key_values")
        cache_output = ref.get("cache_output", "past_key_values")
        parameters = inspect.signature(model.forward).parameters
        state = {}
        positions = [torch.tensor([prefix_length + index], device=ids.device) for index in range(2)]

        def initial():
            kwargs = dict(inputs, **extra)
            kwargs["decoder_input_ids" if encoder_decoder else "input_ids"] = ids[:, :prefix_length]
            for name in ref.get("prefill_sequence_input_names", ()):
                kwargs[name] = inputs[name][:, :prefix_length]
            mask_name = "decoder_attention_mask" if encoder_decoder else "attention_mask"
            if mask_name in kwargs:
                kwargs[mask_name] = kwargs[mask_name][:, :prefix_length]
            if not memory_continuation and not ref.get("native_cache_defaults", False):
                kwargs["use_cache"] = True
            if ref.get("caller_retained_cache"):
                from transformers.cache_utils import DynamicCache

                # RecurrentGemma updates caller-owned attention state and its
                # own recurrent buffers, but omits them from the LM output.
                model._setup_cache(model.config, ids.shape[0], ids.device,
                                   model.get_input_embeddings().weight.dtype)
                state["cache"] = DynamicCache(config=model.config)
                kwargs[cache_argument] = state["cache"]
            if ref.get("encoder_input_names"):
                # Some encoders return a transformed mask that the public LM
                # output omits. Retain their native output for continuation;
                # encoder computation still runs inside the timed initial call.
                encoder_inputs = {name: kwargs.pop(name) for name in ref["encoder_input_names"]}
                state["encoder"] = model.get_encoder()(**encoder_inputs)
                kwargs["encoder_outputs"] = state["encoder"]
            return model(**kwargs)

        def advance(index):
            token = ids[:, prefix_length + index:prefix_length + index + 1]
            if ref.get("decode_input") == "full_history":
                # CPMAnt builds positions and masks from the complete history,
                # then selects the uncached tokens inside its forward method.
                token = ids[:, :prefix_length + index + 1]
            kwargs = dict(extra)
            kwargs.update(state.get("decode_inputs", {}))
            for name in ref.get("decode_sequence_input_names", ()):
                kwargs[name] = inputs[name][:, prefix_length + index:prefix_length + index + 1]
            if not memory_continuation and not ref.get("native_cache_defaults", False):
                kwargs["use_cache"] = True
            kwargs[cache_argument] = state["cache"]
            if encoder_decoder:
                if ref.get("decoder_full_history"):
                    # FSMT derives positions from the history, then internally
                    # selects the newest token for its cached forward.
                    token = ids[:, :prefix_length + index + 1]
                kwargs["decoder_input_ids"] = token
                if reuse_encoder:
                    kwargs["encoder_outputs"] = (state["encoder"] if ref.get("encoder_input_names")
                                                 else encoder_output_type(last_hidden_state=state["encoder"]))
                if reuse_encoder and "attention_mask" in inputs:
                    kwargs["attention_mask"] = inputs["attention_mask"]
                if "decoder_attention_mask" in inputs:
                    kwargs["decoder_attention_mask"] = inputs["decoder_attention_mask"][:, :prefix_length + index + 1]
            else:
                kwargs["input_ids"] = token
                for name in ref.get("continuation_input_names", ()):
                    kwargs[name] = inputs[name]
                if "attention_mask" in inputs:
                    # XLNet adds its previous memory mask internally; its input
                    # mask describes only the tokens in this call.
                    kwargs["attention_mask"] = (inputs["attention_mask"][:, prefix_length + index:prefix_length + index + 1]
                                                if memory_continuation else
                                                inputs["attention_mask"][:, :prefix_length + index + 1])
                position = positions[index]
                if "cache_position" in parameters:
                    kwargs["cache_position"] = position
                if "position_ids" in parameters and not ref.get("native_position_ids"):
                    kwargs["position_ids"] = position.unsqueeze(0)
            output = model(**kwargs)
            if not ref.get("caller_retained_cache"):
                state["cache"] = getattr(output, cache_output)
            return output

        def prepare_step(index):
            output = initial()
            # Cross-attention models retain the image encoding from prefill;
            # continuation consumes it without recomputing the vision tower.
            state["decode_inputs"] = {
                name: getattr(output, field)
                for name, field in ref.get("decode_from_prefill_outputs", {}).items()
            }
            if not ref.get("caller_retained_cache"):
                state["cache"] = getattr(output, cache_output)
            if state["cache"] is None:
                raise RuntimeError("HF did not return a cache for the continuation workload")
            if reuse_encoder and not ref.get("encoder_input_names"):
                state["encoder"] = output.encoder_last_hidden_state
            for previous in range(index):
                advance(previous)

        def retain(output):
            state["output"] = output
            # Base decoders such as ImageGPT return hidden states, without an
            # LM head. Collection below still checks every declared output.
            name = ("last_hidden_state" if memory_continuation or not hasattr(output, "logits")
                    else "logits")
            return {name: getattr(output, name)}

        def collect(output, output_keys=None):
            retained_cache = state["cache"] if ref.get("caller_retained_cache") else None
            selected = select(state.pop("output"), output_keys, retained_cache=retained_cache)
            if retained_cache is not None:
                for index, layer in enumerate(model.model.layers):
                    block = layer.temporal_block
                    if hasattr(block, "rg_lru"):
                        selected[f"{cache_output}.{index}.conv_states"] = block.conv1d_state
                        selected[f"{cache_output}.{index}.recurrent_states"] = block.rg_lru.recurrent_states
            return selected

        initial_phase = "forward" if memory_continuation else "prefill"
        next_phase = "continuation" if memory_continuation else "decode"
        workloads = {initial_phase: Workload(run=lambda: retain(initial()), collect=collect)}
        for index in range(2):
            workloads[f"{next_phase}_{index + 1}"] = Workload(
                run=lambda index=index: retain(advance(index)),
                prepare=lambda index=index: prepare_step(index),
                collect=lambda output: collect(output, ref.get("continuation_outputs")),
            )
        return workloads

    if case["workload"] in ("forward", "masked_lm", "image_classification"):
        return {"forward": Workload(run=lambda: select(model(**inputs, **extra)))}
    if case["workload"] == "seq2seq_cached":
        decoder_ids = inputs["decoder_input_ids"]
        if decoder_ids.shape[1] != 2:
            raise ValueError("seq2seq_cached evaluates an initial token and one continuation")
        state = {}

        def initial():
            return model(**dict(inputs, decoder_input_ids=decoder_ids[:, :1]), **extra)

        def prepare_continuation():
            output = initial()
            state["cache"] = output.past_key_values
            state["encoder"] = (output.encoder_last_hidden_state,)
            if state["cache"] is None:
                raise RuntimeError("HF did not return the requested encoder-decoder cache")

        def continuation():
            return select(model(decoder_input_ids=decoder_ids, encoder_outputs=state["encoder"],
                                past_key_values=state["cache"], **extra))

        return {"prefill": Workload(run=lambda: select(initial())),
                "decode": Workload(run=continuation, prepare=prepare_continuation)}
    if case["workload"] != "causal_lm":
        raise ValueError(f"workload {case['workload']!r} is not implemented")
    ids = inputs["input_ids"]
    prompt, token = ids[:, :-1], ids[:, -1:]
    ref = case["reference"]
    cache_argument = ref.get("cache_argument", "past_key_values")
    cache_output = ref.get("cache_output", "past_key_values")
    parameters = inspect.signature(model.forward).parameters
    prompt_positions = torch.arange(prompt.shape[1], device=ids.device)
    decode_position = torch.tensor([prompt.shape[1]], device=ids.device)
    state = {}

    def prefill():
        kwargs = dict(extra, input_ids=prompt, use_cache=True)
        if ref.get("caller_retained_cache"):
            from transformers.cache_utils import DynamicCache

            # RecurrentGemma accepts a caller cache but omits it from outputs.
            # Preserve its own recurrent-state initialization and cache class.
            model._setup_cache(model.config, ids.shape[0], ids.device,
                               model.get_input_embeddings().weight.dtype)
            state["cache"] = DynamicCache(config=model.config)
            kwargs[cache_argument] = state["cache"]
        for name in ref.get("prefill_input_names", []):
            kwargs[name] = inputs[name]
        for name in ref.get("prefill_sequence_input_names", []):
            kwargs[name] = inputs[name][:, :-1]
        if "cache_position" in parameters:
            kwargs["cache_position"] = prompt_positions
        if "position_ids" in parameters and not ref.get("native_position_ids"):
            kwargs["position_ids"] = prompt_positions.unsqueeze(0)
        return model(**kwargs)

    def prepare_decode():
        output = prefill()
        state["decode_inputs"] = {
            name: getattr(output, field)
            for name, field in ref.get("decode_from_prefill_outputs", {}).items()
        }
        if not ref.get("caller_retained_cache"):
            state["cache"] = getattr(output, cache_output)
        if state["cache"] is None:
            raise RuntimeError("HF did not return the requested cache")

    def decode():
        decode_ids = ids if ref.get("decode_input") == "full_history" else token
        kwargs = dict(extra, input_ids=decode_ids, use_cache=True)
        kwargs.update(state.get("decode_inputs", {}))
        for name in ref.get("decode_sequence_input_names", []):
            kwargs[name] = inputs[name][:, -1:]
        kwargs[cache_argument] = state["cache"]
        if "cache_position" in parameters:
            kwargs["cache_position"] = decode_position
        if "position_ids" in parameters and not ref.get("native_position_ids"):
            kwargs["position_ids"] = decode_position.unsqueeze(0)
        return select(model(**kwargs), case.get("decode_outputs", keys))

    return {"prefill": Workload(run=lambda: select(prefill())),
            "decode": Workload(run=decode, prepare=prepare_decode)}


def execute(job: dict, directory: Path) -> dict:
    import torch

    pin = verify_reference_pin(job["transformers_revision"])
    configure_torch(job["seed"], "float32", gpu=True,
                    cudnn_deterministic=job["case"].get("cudnn_deterministic", False))
    prepared = torch.load(directory / "prepared.pt", map_location="cpu", weights_only=True)
    ref = job["case"]["reference"]
    config = symbol(ref["config_class"]).from_dict(prepared["config"])
    config._attn_implementation = job["case"].get("reference_backend", "eager")
    generation_config, generation_record = resolve_generation_config(ref)
    model, loading_info = load_reference_model(symbol(ref["model_class"]), config,
                                               prepared["weights"], getattr(torch, job["dtype"]),
                                               load_with_base_class=ref.get("load_with_base_class", False),
                                               reference_backend=job["case"].get("reference_backend", "eager"),
                                               load_device=ref.get("load_device", "cpu"),
                                               generation_config=generation_config,
                                               adapter_config=ref.get("adapter_config"))
    if generation_record is not None:
        loading_info["generation_config_source"] = generation_record
    if job["case"].get("reference_experts_backend") is not None:
        model.set_experts_implementation(job["case"]["reference_experts_backend"])
    model.to(device="cuda:0")
    speakers_record = None
    if "speakers" in ref:
        from huggingface_hub import hf_hub_download

        speakers = ref["speakers"]
        path = hf_hub_download(speakers["repo"], speakers["filename"], revision=speakers["revision"])
        model.load_speakers(path)
        speakers_record = {**speakers, "sha256": digest(Path(path))}
    inputs = to_device(prepared["inputs"], "cuda:0")
    workloads = reference_workloads(model, inputs, job["case"])
    measurements, outputs = measure(workloads, warmup=job["warmup"], iterations=job["iterations"],
                                    generation_seed=job["case"].get("generation_seed"))
    torch.save(outputs, directory / "reference_outputs.pt")
    return {"status": "completed", **worker_metadata(), "reference_pin": pin,
            "loading_method": "pinned HF from_pretrained with common state_dict and requested dtype; device transfer only",
            "base_class_loading": ref.get("load_with_base_class", False),
            "load_device": ref.get("load_device", "cpu"),
            "serialized_weight_suffixes": ref.get("serialized_weight_suffixes", []),
            "speakers": speakers_record,
            "loading_info": loading_info,
            "reference_backend": model.config._attn_implementation,
            "reference_experts_backend": getattr(model.config, "_experts_implementation", None),
            "workloads": measurements}


if __name__ == "__main__":
    phase, path = sys.argv[1:]
    functions = {"prepare": prepare, "reference": execute}
    if phase not in functions:
        raise SystemExit("reference worker accepts prepare or reference phases")
    raise SystemExit(guarded_worker(Path(path), phase, functions[phase]))
