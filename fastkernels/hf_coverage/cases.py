"""Selected audit workloads: HF sources, sizes, inputs, and checked outputs.

Each entry is complete; no size or historical-stage overrides run afterward.
Registration is not review approval. See review.csv for outstanding work.
"""

CASES = {
    'afmoe': {
        'reference': {
            'config_class': 'transformers:AfmoeConfig',
            'model_class': 'transformers:AfmoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'arcee-ai/Trinity-Mini',
                'revision': 'cdb81e8130815fec299cccc8f8700004b8366e1e',
                'url': 'https://huggingface.co/arcee-ai/Trinity-Mini/blob/cdb81e8130815fec299cccc8f8700004b8366e1e/config.json',
                'description': ('Pinned HF model documentation checkpoint; retain dual norms, output gate, first two '
                    'dense layers and local/global attention schedule.'),
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 5, 'num_attention_heads': 8, 'num_key_value_heads': 1, 'head_dim': 64,
            'vocab_size': 1024, 'sliding_window': 128,
            'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention'],
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve attention width2x hidden,8:1 GQA, dense2 then routed3, local/full/local schedule, '
            'original128 experts/top8/shared1, muP embedding scale and FP32 normalization arithmetic. '
            'Reduce sliding window128 and use270 tokens to cross boundary.'),
    },

    'aimv2': {
        'reference': {
            'config_class': 'transformers:Aimv2Config',
            'model_class': 'transformers:Aimv2Model',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'apple/aimv2-large-patch14-224-lit',
                'revision': 'b17c109df4f9dbb941074073ad3771c28df5c826',
                'description': ('Pinned Aimv2Model.forward paired image/text processor example selects the lit '
                    'checkpoint, learned vision positions and attention pooling. Processor supplies '
                    'attention mask, enabling causal text attention.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/aimv2/modeling_aimv2.py#L687-L710',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 1, 'sequence_length': 77,
            'shape': [3, 224, 224], 'eos_positions': [25, 76], 'attention_mask': True, 'pad_token_id': 49407,
            'bos_token_id': 49406, 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_image', 'logits_per_text', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'albert': {
        'reference': {
            'config_class': 'transformers:AlbertConfig',
            'model_class': 'transformers:AlbertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned HF AlbertForMaskedLM example explicitly loads albert/albert-base-v2; retain one'
                    ' shared hidden group, one inner layer, gelu_new, and the narrow embedding projection.'),
                'checkpoint': 'albert/albert-base-v2',
                'revision': '8e2f239c5f8a2c0f253781ca60135db913e5c80c',
                'url': 'https://huggingface.co/albert/albert-base-v2/blob/8e2f239c5f8a2c0f253781ca60135db913e5c80c/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'align': {
        'reference': {
            'config_class': 'transformers:AlignConfig',
            'model_class': 'transformers:AlignModel',
            'source': {
                'checkpoint': 'kakaobrain/align-base',
                'revision': 'e96a37facc7b1f59090ece82293226b817afd6ba',
                'url': 'https://huggingface.co/kakaobrain/align-base/resolve/e96a37facc7b1f59090ece82293226b817afd6ba/config.json',
                'kind': 'example_checkpoint',
                'description': ('The pinned paired forward example names this checkpoint. Retain raw CLS text '
                    'projection, the separately executed text pooler, and final MBConv feature pooling '
                    'without an EfficientNet top projection.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 1, 'sequence_length': 64,
            'shape': [3, 289, 289], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'altclip': {
        'reference': {
            'config_class': 'transformers:AltCLIPConfig',
            'model_class': 'transformers:AltCLIPModel',
            'source': {
                'checkpoint': 'BAAI/AltCLIP',
                'revision': '17788d7d45af0e41d0f10682fe91d655ac96435c',
                'url': 'https://huggingface.co/BAAI/AltCLIP/resolve/17788d7d45af0e41d0f10682fe91d655ac96435c/config.json',
                'kind': 'example_checkpoint',
                'description': ('The pinned full-forward example mistakenly names a ChineseCLIP checkpoint. The same '
                    'AltCLIP class feature methods and both tower examples name this architecture-matching '
                    'checkpoint. Retain the full projected text sequence and both original nested poolers.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 1, 'sequence_length': 64,
            'shape': [3, 224, 224], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'apertus': {
        'reference': {
            'config_class': 'transformers:ApertusConfig',
            'model_class': 'transformers:ApertusForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'swiss-ai/Apertus-8B-Instruct-2509',
                'revision': 'b946d40447b2b597999b9c86d44bee0b452c919f',
                'description': 'Pinned public generation example; checkpoint has use_cache=False, xIELU, learned Q/K normalization, 4:1 GQA, Llama3 RoPE.',
            },
            'generation_config': {
                '_from_model_config': True, 'bos_token_id': 1, 'eos_token_id': [2, 68, 72],
                'transformers_version': '4.54.0.dev0',
            },
        },
        'input': {
            'kind': 'supplied',
            'description': 'Requires --input-dict; single 64-token prompt and supplied attention mask.',
            'batch_size': 1,
        },
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 4, 'do_sample': False, 'use_cache': False},
        'outputs': ['sequences', 'logits'],
        'reference_backend': None,
    },

    'arcee': {
        'workload': 'forward',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:ArceeConfig',
            'model_class': 'transformers:ArceeForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'arcee-ai/AFM-4.5B',
                'revision': 'e84d19711eca5817c1608cb344d931bf05c966dd',
                'description': ('Documented example after unavailable task placeholder. Config and generation_config '
                    'both disable cache; preserves ordinary full-sequence forward.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 257},
        'outputs': ['logits'],
    },

    'aria': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:AriaConfig',
            'model_class': 'transformers:AriaForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Rhymes-AI/Aria',
                'revision': '53a09c77895d7025efa6239ea71cf1a8991064e8',
                'description': 'Pinned HF Aria conditional-generation example checkpoint.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/aria/modeling_aria.py',
            },
            'prefill_input_names': ['pixel_values'],
        },
        'dimension_overrides': {
            'vision_config': {
                'image_size': 70, 'hidden_size': 128, 'intermediate_size': 512, 'num_hidden_layers': 2,
                'num_attention_heads': 4,
            },
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 128, 'num_hidden_layers': 2,
                'num_attention_heads': 2, 'num_key_value_heads': 2, 'moe_num_experts': 8,
            },
            'projector_patch_to_query_dict': {'25': 8},
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 70, 70],
            'sequence_length': 24, 'image_token_positions': list(range(3, 11)),
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve Idefics3 tanh-GELU vision, both stacked projector attention projections and learned '
            'queries, shared experts plus routed top6 of8, native text head_dim128 and RoPE/cache. Reduced '
            'depth,width,image size and query/expert count only. Pinned AriaForConditionalGeneration does '
            'not populate declared image_hidden_states; complete frontend is still executed and separately '
            'verified.'),
    },

    'audio_spectrogram_transformer': {
        'reference': {
            'config_class': 'transformers:ASTConfig',
            'model_class': 'transformers:ASTModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/audio_spectrogram_transformer/configuration_audio_spectrogram_transformer.py#L32-L46',
                'description': ('Pinned HF base-model example constructs ASTModel(ASTConfig()). Preserve overlapping '
                    'patches and both special tokens.'),
            },
        },
        'input': {'kind': 'spectrogram', 'batch_size': 1, 'shape': [1024, 128]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'audioflamingo3': {
        'reference': {
            'config_class': 'transformers:AudioFlamingo3Config',
            'model_class': 'transformers:AudioFlamingo3ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'nvidia/audio-flamingo-3-hf',
                'revision': '7d4bae64ee29878af6504ae6f6bb3e40492838ad',
                'description': 'Official pinned HF conditional-generation example and published default configuration.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/audioflamingo3/modeling_audioflamingo3.py',
            },
        },
        'dimension_overrides': {
            'audio_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'max_source_positions': 32,
            },
            'text_config': {
                'hidden_size': 448, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 7, 'num_key_value_heads': 1,
                'layer_types': ['full_attention', 'full_attention'],
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [128, 64],
            'sequence_length': 43, 'audio_token_positions': list(range(3, 15)),
            'image_input_name': 'input_features', 'input_features_lengths': [49],
        },
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain 128 mel bands and 7:1 language grouped-query attention. Of 64 audio frames, 49 are '
            'valid; this exercises attention masking and selection of 12 valid pooled tokens out of 16. '
            'Keep the checkpoint default that disables language caching.'),
    },

    'bamba': {
        'reference': {
            'config_class': 'transformers:BambaConfig',
            'model_class': 'transformers:BambaForCausalLM',
            'source': {
                'checkpoint': 'ibm-ai-platform/Bamba-9B-v2',
                'revision': 'b42852dc9eb96c8ae3359dc8df0e4c3f5c37eb21',
                'url': 'https://huggingface.co/ibm-ai-platform/Bamba-9B-v2/blob/b42852dc9eb96c8ae3359dc8df0e4c3f5c37eb21/config.json',
                'kind': 'example_checkpoint',
                'description': ('Pinned HF docs/source/en/model_doc/bamba.md:62-63 causal-LM example, completed by '
                    'pinned constructor defaults. Preserve the first11-layer pattern with attention at9 and'
                    ' default full-head RoPE executed by pinned code despite partial_rotary_factor0.5.'),
            },
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 270},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
        'dimension_overrides': {
            'hidden_size': 512, 'intermediate_size': 1792, 'num_hidden_layers': 11, 'num_attention_heads': 4,
            'num_key_value_heads': 1, 'mamba_n_heads': 16, 'mamba_d_head': 64, 'mamba_d_state': 128,
            'vocab_size': 1024, 'attn_layer_indices': [9],
        },
        'dimension_purpose': ('Retain the first 11 blocks, including attention at 9 and recurrence afterward; 128-wide '
            'attention heads, 4:1 grouped queries, 3.5x feed-forward width, 64-wide recurrent heads, '
            '128-state recurrence and 256-token chunks retained.'),
        'configuration_scope': 'reviewed_scaled_v1',
    },

    'bark': {
        'reference': {
            'config_class': 'transformers:BarkConfig',
            'model_class': 'transformers:BarkModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'suno/bark',
                'revision': '70a8a7d34168586dc5d028fa9666aceade177992',
                'description': 'Native unprompted Bark.generate with official stochastic semantic/coarse/fine settings; full native EnCodec decode.',
            },
            'generation_config': {
                'coarse_acoustics_config': {
                    '_from_model_config': False,
                    'bad_words_ids': None,
                    'begin_suppress_tokens': None,
                    'bos_token_id': None,
                    'coarse_infer_token': 12050,
                    'coarse_rate_hz': 75,
                    'coarse_semantic_pad_token': 12048,
                    'constraints': None,
                    'decoder_start_token_id': None,
                    'diversity_penalty': 0.0,
                    'do_sample': True,
                    'early_stopping': False,
                    'encoder_no_repeat_ngram_size': 0,
                    'encoder_repetition_penalty': 1.0,
                    'eos_token_id': None,
                    'epsilon_cutoff': 0.0,
                    'eta_cutoff': 0.0,
                    'exponential_decay_length_penalty': None,
                    'force_words_ids': None,
                    'forced_bos_token_id': None,
                    'forced_decoder_ids': None,
                    'forced_eos_token_id': None,
                    'generation_kwargs': {},
                    'guidance_scale': None,
                    'length_penalty': 1.0,
                    'max_coarse_history': 630,
                    'max_coarse_input_length': 256,
                    'max_length': 20,
                    'max_new_tokens': None,
                    'max_time': None,
                    'min_length': 0,
                    'min_new_tokens': None,
                    'n_coarse_codebooks': 2,
                    'no_repeat_ngram_size': 0,
                    'num_beam_groups': 1,
                    'num_beams': 1,
                    'num_return_sequences': 1,
                    'output_attentions': False,
                    'output_hidden_states': False,
                    'output_scores': False,
                    'pad_token_id': None,
                    'penalty_alpha': None,
                    'remove_invalid_values': False,
                    'renormalize_logits': True,
                    'repetition_penalty': 1.0,
                    'return_dict_in_generate': False,
                    'sequence_bias': None,
                    'sliding_window_len': 60,
                    'suppress_tokens': None,
                    'temperature': 0.7,
                    'top_k': 50,
                    'top_p': 1.0,
                    'transformers_version': '4.31.0.dev0',
                    'typical_p': 1.0,
                    'use_cache': True,
                },
                'codebook_size': 1024,
                'fine_acoustics_config': {
                    '_from_model_config': False,
                    'bad_words_ids': None,
                    'begin_suppress_tokens': None,
                    'bos_token_id': None,
                    'constraints': None,
                    'decoder_start_token_id': None,
                    'diversity_penalty': 0.0,
                    'do_sample': False,
                    'early_stopping': False,
                    'encoder_no_repeat_ngram_size': 0,
                    'encoder_repetition_penalty': 1.0,
                    'eos_token_id': None,
                    'epsilon_cutoff': 0.0,
                    'eta_cutoff': 0.0,
                    'exponential_decay_length_penalty': None,
                    'force_words_ids': None,
                    'forced_bos_token_id': None,
                    'forced_decoder_ids': None,
                    'forced_eos_token_id': None,
                    'generation_kwargs': {},
                    'guidance_scale': None,
                    'length_penalty': 1.0,
                    'max_fine_history_length': 512,
                    'max_fine_input_length': 1024,
                    'max_length': 20,
                    'max_new_tokens': None,
                    'max_time': None,
                    'min_length': 0,
                    'min_new_tokens': None,
                    'n_fine_codebooks': 8,
                    'no_repeat_ngram_size': 0,
                    'num_beam_groups': 1,
                    'num_beams': 1,
                    'num_return_sequences': 1,
                    'output_attentions': False,
                    'output_hidden_states': False,
                    'output_scores': False,
                    'pad_token_id': None,
                    'penalty_alpha': None,
                    'remove_invalid_values': False,
                    'renormalize_logits': False,
                    'repetition_penalty': 1.0,
                    'return_dict_in_generate': False,
                    'sequence_bias': None,
                    'suppress_tokens': None,
                    'temperature': 0.5,
                    'top_k': 50,
                    'top_p': 1.0,
                    'transformers_version': '4.31.0.dev0',
                    'typical_p': 1.0,
                    'use_cache': True,
                },
                'model_type': 'bark',
                'sample_rate': 24000,
                'semantic_config': {
                    '_from_model_config': False,
                    'bad_words_ids': None,
                    'begin_suppress_tokens': None,
                    'bos_token_id': None,
                    'constraints': None,
                    'decoder_start_token_id': None,
                    'diversity_penalty': 0.0,
                    'do_sample': True,
                    'early_stopping': False,
                    'encoder_no_repeat_ngram_size': 0,
                    'encoder_repetition_penalty': 1.0,
                    'eos_token_id': 10000,
                    'epsilon_cutoff': 0.0,
                    'eta_cutoff': 0.0,
                    'exponential_decay_length_penalty': None,
                    'force_words_ids': None,
                    'forced_bos_token_id': None,
                    'forced_decoder_ids': None,
                    'forced_eos_token_id': None,
                    'generation_kwargs': {},
                    'guidance_scale': None,
                    'length_penalty': 1.0,
                    'max_input_semantic_length': 256,
                    'max_length': 20,
                    'max_new_tokens': 768,
                    'max_time': None,
                    'min_length': 0,
                    'min_new_tokens': None,
                    'no_repeat_ngram_size': 0,
                    'num_beam_groups': 1,
                    'num_beams': 1,
                    'num_return_sequences': 1,
                    'output_attentions': False,
                    'output_hidden_states': False,
                    'output_scores': False,
                    'pad_token_id': None,
                    'penalty_alpha': None,
                    'remove_invalid_values': False,
                    'renormalize_logits': True,
                    'repetition_penalty': 1.0,
                    'return_dict_in_generate': False,
                    'semantic_infer_token': 129599,
                    'semantic_pad_token': 10000,
                    'semantic_rate_hz': 49.9,
                    'semantic_vocab_size': 10000,
                    'sequence_bias': None,
                    'suppress_tokens': None,
                    'temperature': 0.7,
                    'text_encoding_offset': 10048,
                    'text_pad_token': 129595,
                    'top_k': 50,
                    'top_p': 1.0,
                    'transformers_version': '4.31.0.dev0',
                    'typical_p': 1.0,
                    'use_cache': True,
                },
            },
            'generation_output_name': 'waveform',
        },
        'dimension_overrides': {
            'semantic_config': {'hidden_size': 64, 'num_layers': 2, 'num_heads': 4},
            'coarse_acoustics_config': {'hidden_size': 64, 'num_layers': 2, 'num_heads': 4},
            'fine_acoustics_config': {'hidden_size': 64, 'num_layers': 2, 'num_heads': 4},
        },
        'reference_backend': 'eager',
        'input': {'kind': 'external', 'batch_size': 1, 'sequence_length': 256},
        'workload': 'generate',
        'generation_kwargs': {'semantic_max_new_tokens': 24},
        'generation_seed': 23,
        'outputs': ['waveform'],
        'dimension_purpose': ('Reduce only transformer widths64/layers2/heads4; preserve native '
            'vocabularies,256textprefix,coarsewindow60,fine1024bidirectionalcontext,all8codebooks,fullnativecodec.24sampledsemanticsteps'
            ' exercise coarse60tokenwindow transition and boundaudiolength; no optionalspeakerprompt.'),
    },

    'bart': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:BartConfig',
            'model_class': 'transformers:BartForConditionalGeneration',
            'forward_kwargs': {},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/bart-large-cnn',
                'revision': '37f520fa929c961707657b28798b30c003dd100b',
                'description': ('First pinned conditional-generation task example: summarization with BART-large-CNN; '
                    'later mask-filling example is a separate task example.'),
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'beit': {
        'reference': {
            'config_class': 'transformers.models.beit.configuration_beit:BeitConfig',
            'model_class': 'transformers.models.beit.modeling_beit:BeitModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs BeitConfig() and '
                    'BeitModel(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/beit/configuration_beit.py#L55',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'bert': {
        'reference': {
            'config_class': 'transformers:BertConfig',
            'model_class': 'transformers:BertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned HF configuration documentation names google-bert/bert-base-uncased; use its '
                    'checkpoint config, completed by pinned constructor defaults.'),
                'checkpoint': 'google-bert/bert-base-uncased',
                'revision': '86b5e0934494bd15c9632b12f734a8a67f723594',
                'url': 'https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'dimension_overrides': {
            'hidden_size': 256, 'num_attention_heads': 4, 'num_hidden_layers': 4, 'intermediate_size': 1024,
            'vocab_size': 1024,
        },
        'dimension_purpose': 'Four repeated encoder blocks; 64-wide heads and 4x feed-forward width; 512 tokens.',
        'configuration_scope': 'reviewed_scaled_v1',
    },

    'bert_generation': {
        'reference': {
            'config_class': 'transformers:BertGenerationConfig',
            'model_class': 'transformers:BertGenerationEncoder',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned standalone BertGenerationConfig example constructs BertGenerationEncoder; use '
                    'its named encoder checkpoint and preserve the bare bidirectional hidden-state output.'),
                'checkpoint': 'google/bert_for_seq_generation_L-24_bbc_encoder',
                'revision': 'c817d1fd1be2ffa69431227a1fe320544943d4db',
                'url': 'https://huggingface.co/google/bert_for_seq_generation_L-24_bbc_encoder/blob/c817d1fd1be2ffa69431227a1fe320544943d4db/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'big_bird': {
        'reference': {
            'config_class': 'transformers:BigBirdConfig',
            'model_class': 'transformers:BigBirdForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/bigbird-roberta-base',
                'revision': '5a145f7852cba9bd431386a58137bf8a29903b90',
                'url': 'https://huggingface.co/google/bigbird-roberta-base/blob/5a145f7852cba9bd431386a58137bf8a29903b90/config.json',
                'description': ('Pinned HF MLM example names this checkpoint; preserve block64/random3 sparse attention'
                    ' with padded length832 above dense fallback cutoff704.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 769},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'bigbird_pegasus': {
        'reference': {
            'config_class': 'transformers:BigBirdPegasusConfig',
            'model_class': 'transformers:BigBirdPegasusForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/bigbird-pegasus-large-arxiv',
                'revision': 'c5ae494f3d319933617d513f8491c5be87f5d35b',
                'description': ('Pinned conditional-generation example checkpoint preserves block-sparse attention, '
                    'learned positions, bias-free attention and GELU-new.'),
            },
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 769,
            'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'biogpt': {
        'reference': {
            'config_class': 'transformers:BioGptConfig',
            'model_class': 'transformers:BioGptForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/biogpt',
                'revision': 'eb0d815e95434dc9e3b78f464e52b899bee7d923',
                'url': 'https://huggingface.co/microsoft/biogpt/blob/eb0d815e95434dc9e3b78f464e52b899bee7d923/config.json',
                'description': ('Pinned HF task example or configuration documentation checkpoint; preserve checkpoint '
                    'computation flags, completed by pinned constructor defaults.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'bit': {
        'reference': {
            'config_class': 'transformers:BitConfig',
            'model_class': 'transformers:BitModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/bit/configuration_bit.py',
                'description': ('Pinned HF constructor defaults select this architecture and retain its public task; '
                    'ViT-MAE retains the original masked-image pretraining task including default loss.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['last_hidden_state', 'pooler_output'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'bitnet': {
        'reference': {
            'config_class': 'transformers:BitNetConfig',
            'model_class': 'transformers:BitNetForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/bitnet-b1.58-2B-4T',
                'revision': '04c3b9ad9361b824064a1f25ea60a8be9599b127',
                'description': 'Pinned causal-LM example; preserve offline ternary quantization with common explicitly prepared weights and scales.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 512, 'intermediate_size': 1024, 'num_hidden_layers': 2, 'num_attention_heads': 4,
            'num_key_value_heads': 1, 'head_dim': 128, 'vocab_size': 1024, 'bos_token_id': 1,
            'eos_token_id': 2,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Two residual layers, native128-wide heads,4:1GQA and269-token prompt plus cached continuation;'
            ' vocabulary/token IDs resized only. Common offline ternary state must be supplied; ordinary '
            'dense random initialization is not the quantized model.'),
    },

    'blenderbot': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:BlenderbotConfig',
            'model_class': 'transformers:BlenderbotForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'facebook/blenderbot-400M-distill',
                'revision': 'eaaf64e3be20ad1f1fb0bdf689565ba52c97eafe',
                'description': 'Pinned conditional-generation conversation example checkpoint.',
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 97,
            'decoder_sequence_length': 75, 'encoder_prefix_token_ids': [], 'encoder_suffix_token_ids': [2],
            'content_token_id_min': 4,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'blenderbot_small': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:BlenderbotSmallConfig',
            'model_class': 'transformers:BlenderbotSmallForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'facebook/blenderbot_small-90M',
                'revision': 'bbf60f5f68fd8789ac04bd1c20712233f3dc899f',
                'description': 'Pinned conditional-generation conversation example checkpoint.',
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 97,
            'decoder_sequence_length': 75, 'encoder_prefix_token_ids': [], 'encoder_suffix_token_ids': [],
            'content_token_id_min': 4,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'blip': {
        'reference': {
            'config_class': 'transformers:BlipConfig',
            'model_class': 'transformers:BlipModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Salesforce/blip-image-captioning-base',
                'revision': '82a37760796d32b1411fe092ab5d4e227313294b',
                'description': ('Pinned BlipModel.forward example selects this checkpoint for paired text/image '
                    'base-model inference; both towers and poolers are retained.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/blip/modeling_blip.py',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 2, 'sequence_length': 37,
            'shape': [3, 384, 384], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_image', 'logits_per_text', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'blip_2': {
        'reference': {
            'fan_in_normal_modules': ['vision_model'],
            'randomize_zero_parameters': ['query_tokens'],
            'config_class': 'transformers:Blip2Config',
            'model_class': 'transformers:Blip2ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Salesforce/blip2-opt-2.7b',
                'revision': '59a1ef6c1e5117b3f65523d1c6066825bcf315e3',
                'description': ('Pinned conditional-generation example selects Salesforce/blip2-opt-2.7b. Full '
                    'vision/query-former/OPT path, current image placeholder convention. Optional 8-bit '
                    'loading not enabled.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/blip_2/modeling_blip_2.py#L1730-L1781',
            },
        },
        'dimension_overrides': {
            'vision_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'image_size': 56,
            },
            'qformer_config': {
                'hidden_size': 256, 'encoder_hidden_size': 256, 'intermediate_size': 1024,
                'num_hidden_layers': 4, 'num_attention_heads': 4,
            },
            'text_config': {
                'hidden_size': 256, 'word_embed_proj_dim': 256, 'ffn_dim': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 2, 'sequence_length': 43,
            'shape': [3, 56, 56], 'image_token_positions': list(range(0, 32)),
        },
        'initialization_reason': (
            'Native vision weights make residual updates negligible; zero query tokens hide Qformer self-attention. '
            'Shared fan-in vision matrices and nonzero random query tokens activate these paths while preserving '
            'biases, normalization, dimensions and inputs.'),
        'workload': 'forward',
        'outputs': [
            'logits', 'language_model_outputs.logits', 'vision_outputs.last_hidden_state',
            'vision_outputs.pooler_output', 'qformer_outputs.last_hidden_state',
            'qformer_outputs.pooler_output', 'language_model_outputs.past_key_values',
        ],
        'reference_backend': {'': 'sdpa', 'vision_config': 'sdpa', 'text_config': 'sdpa', 'qformer_config': 'eager'},
        'dimension_purpose': ('Retain native32 query tokens and image placeholders, all vision blocks, alternating '
            'query-former cross/self-only layers, native query-former loading dtype, pre-norm OPT without '
            'width projection and tied native50304 vocabulary. Full conditioned forward with returned '
            'language caches; decode continuation checked separately.'),
    },

    'blip_text': {
        'reference': {
            'config_class': 'transformers:BlipTextConfig',
            'model_class': 'transformers.models.blip.modeling_blip_text:BlipTextModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('Pinned BlipTextConfig public example constructs BlipTextModel(BlipTextConfig()). '
                    'Default forward is_decoder=False despite config.is_decoder=True, no cross input; '
                    'pooler runs and is compared.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 193},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'bloom': {
        'reference': {
            'config_class': 'transformers:BloomConfig',
            'model_class': 'transformers:BloomForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'bigscience/bloom',
                'revision': '7f10a99ce7c08f03c7719a586cb2cbda1433ac05',
                'url': 'https://huggingface.co/bigscience/bloom/blob/7f10a99ce7c08f03c7719a586cb2cbda1433ac05/config.json',
                'description': ('Pinned BloomConfig checkpoint annotation names bigscience/bloom; the '
                    'causal-language-model forward has no checkpoint example and the model documentation '
                    'lists this checkpoint without selecting a closer task example. Preserve default '
                    'residual order, ALiBi, native BloomGELU and caching.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'blt': {
        'reference': {
            'config_class': 'transformers:BltConfig',
            'model_class': 'transformers:BltForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'itazap/blt-1b-hf',
                'revision': '91aa6b8e168046ad91e517d2191e1e974f50cc01',
                'description': ('Pinned public model example configuration; ordinary unpadded forward includes '
                    'entropy-based patching. Cached generation is not part of this workload.'),
            },
        },
        'reference_backend': None,
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 128},
        'workload': 'forward',
        'outputs': ['logits'],
    },

    'bridgetower': {
        'reference': {
            'config_class': 'transformers:BridgeTowerConfig',
            'model_class': 'transformers:BridgeTowerModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'BridgeTower/bridgetower-base',
                'revision': 'fd4811362b878484486d705eb13d9797361e6788',
                'description': ('Pinned BridgeTowerModel.forward example selects base checkpoint, full text/image/cross'
                    ' towers and additive bridge links.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/bridgetower/modeling_bridgetower.py#L1187-L1208',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 37,
            'shape': [3, 288, 288], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': ['text_features', 'image_features', 'pooler_output', 'hidden_states', 'attentions'],
        'reference_backend': None,
    },

    'bros': {
        'reference': {
            'config_class': 'transformers:BrosConfig',
            'model_class': 'transformers:BrosModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'jinho8345/bros-base-uncased',
                'revision': 'ca0d062e8f49e031f6eaf32fcc18be47301ce255',
                'url': 'https://huggingface.co/jinho8345/bros-base-uncased/blob/ca0d062e8f49e031f6eaf32fcc18be47301ce255/config.json',
                'description': 'Pinned HF BrosModel forward example checkpoint; supplied normalized bounding boxes.',
            },
        },
        'input': {'kind': 'document', 'batch_size': 1, 'sequence_length': 512, 'normalized_bbox': True},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'camembert': {
        'reference': {
            'config_class': 'transformers:CamembertConfig',
            'model_class': 'transformers:CamembertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'almanach/camembert-base',
                'revision': 'a75967561c78f2aa81cc41045378d3b4ee25af9e',
                'url': 'https://huggingface.co/almanach/camembert-base/blob/a75967561c78f2aa81cc41045378d3b4ee25af9e/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/camembert.md:64 masked-LM example; checkpoint '
                    'configuration completed by pinned constructor defaults.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'canine': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:CanineConfig',
            'model_class': 'transformers:CanineModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/canine/configuration_canine.py',
                'description': ('Pinned CanineConfig example constructs CanineModel(CanineConfig()); preserve its '
                    'constructor computation.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512, 'vocab_size': 1114112},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'chinese_clip': {
        'reference': {
            'config_class': 'transformers.models.chinese_clip.configuration_chinese_clip:ChineseCLIPConfig',
            'model_class': 'transformers.models.chinese_clip.modeling_chinese_clip:ChineseCLIPModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'OFA-Sys/chinese-clip-vit-base-patch16',
                'revision': '36e679e65c2a2fead755ae21162091293ad37834',
                'hf_source_revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned ChineseCLIPModel.forward paired text/image example names this checkpoint. '
                    'Preserve both towers, projections, normalized embeddings, all contrasts, and already '
                    'computed nested hidden/projected pooler outputs.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L931',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 3, 'image_batch_size': 2, 'sequence_length': 64,
            'shape': [3, 224, 224], 'batch_size': 1, 'eos_positions': [63, 63, 63], 'bos_token_id': 101,
            'eos_token_id': 102, 'pad_token_id': 0,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'chmv2': {
        'reference': {
            'config_class': 'transformers:CHMv2Config',
            'model_class': 'transformers:CHMv2ForDepthEstimation',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/chmv2/configuration_chmv2.py#L66',
                'description': ('Pinned public task constructor example. Named gated checkpoint config unavailable '
                    '(403); constructor example explicitly establishes this same depth task.'),
            },
        },
        'dimension_overrides': {},
        'reference_backend': 'sdpa',
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 416, 416]},
        'outputs': ['predicted_depth'],
        'workload': 'forward',
        'dimension_purpose': ('Full constructor dimensions and four backbone stages, CLS project readouts, four fusion '
            'stages,256mixeddepthbins;416image matches backbone default.'),
    },

    'clap': {
        'reference': {
            'config_class': 'transformers:ClapConfig',
            'model_class': 'transformers:ClapModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'laion/clap-htsat-unfused',
                'revision': '8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a',
                'url': 'https://huggingface.co/laion/clap-htsat-unfused/blob/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/config.json',
                'description': ('Public paired task forward uses unfused checkpoint; native feature extractor1001x64 '
                    'log-mel, then default bicubic reshape to256x256.'),
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 2, 'image_batch_size': 1, 'sequence_length': 20,
            'shape': [1, 1001, 64], 'image_input_name': 'input_features', 'attention_mask': True,
            'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_audio', 'logits_per_text', 'text_embeds', 'audio_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'audio_model_output.last_hidden_state', 'audio_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'clip': {
        'reference': {
            'config_class': 'transformers:CLIPConfig',
            'model_class': 'transformers:CLIPModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openai/clip-vit-base-patch32',
                'revision': '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                'url': 'https://huggingface.co/openai/clip-vit-base-patch32/blob/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/config.json',
                'description': ('Pinned CLIPModel.forward names this paired text/image checkpoint. Preserve both '
                    'towers, QuickGELU, causal text attention, legacy highest-token-ID pooling, vision CLS '
                    'pre/post normalization, learned projections, normalized embeddings and contrastive '
                    'scale.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 1, 'sequence_length': 77,
            'shape': [3, 224, 224], 'batch_size': 1, 'eos_positions': [76, 76], 'bos_token_id': 49406,
            'eos_token_id': 49407, 'pad_token_id': 49407,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'clipseg': {
        'reference': {
            'config_class': 'transformers:CLIPSegConfig',
            'model_class': 'transformers:CLIPSegForImageSegmentation',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'CIDAS/clipseg-rd64-refined',
                'revision': '999e0328d9e10b484360c477313983f9afdd7050',
                'description': 'Pinned HF public task example checkpoint.',
            },
        },
        'input': {
            'kind': 'text_image', 'shape': [3, 352, 352], 'text_batch_size': 3, 'image_batch_size': 3,
            'sequence_length': 77, 'batch_size': 1, 'eos_positions': [76, 76, 76], 'bos_token_id': 49406,
            'eos_token_id': 49407, 'pad_token_id': 49407,
        },
        'outputs': [
            'logits', 'conditional_embeddings', 'pooled_output', 'vision_model_output.last_hidden_state',
            'vision_model_output.pooler_output', 'vision_model_output.hidden_states', 'decoder_output.logits',
            'decoder_output.hidden_states',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'codegen': {
        'reference': {
            'config_class': 'transformers:CodeGenConfig',
            'model_class': 'transformers:CodeGenForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Salesforce/codegen-350M-mono',
                'revision': 'd9107f71cca463240db1143f4a75a927a27fcb27',
                'url': 'https://huggingface.co/Salesforce/codegen-350M-mono/blob/d9107f71cca463240db1143f4a75a927a27fcb27/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/codegen.md public causal-LM example; preserve '
                    'checkpoint computational settings and native default inference behavior.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'cohere': {
        'reference': {
            'config_class': 'transformers:CohereConfig',
            'model_class': 'transformers:CohereForCausalLM',
            'native_cache_defaults': True,
            'native_position_ids': True,
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/cohere/configuration_cohere.py',
                'description': ('Complete CohereConfig constructor fallback: the documented '
                    'CohereForAI/c4ai-command-r-v01 configuration is gated and unavailable to this '
                    'environment. No checkpoint configuration was loaded.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'cohere2': {
        'reference': {
            'config_class': 'transformers:Cohere2Config',
            'model_class': 'transformers:Cohere2ForCausalLM',
            'native_cache_defaults': True,
            'native_position_ids': True,
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/cohere2/configuration_cohere2.py',
                'description': ('Complete Cohere2Config constructor fallback: the documented '
                    'CohereLabs/c4ai-command-r7b-12-2024 configuration is gated and unavailable to this '
                    'environment. These are constructor dimensions, not the documented 7B checkpoint '
                    'dimensions.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 4099},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'cohere_asr': {'reference': {'config_class': 'transformers:CohereAsrConfig',
                   'model_class': 'transformers:CohereAsrForConditionalGeneration',
                   'source': {'kind': 'constructor_defaults',
                              'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                              'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/cohere_asr/configuration_cohere_asr.py#L27-L35',
                              'description': 'Explicit independent CohereAsrConfig() '
                                             'conditional-generation constructor example. This is not '
                                             'asserted equivalent to the gated checkpoint. Preserve '
                                             'Parakeet/Conformer audio encoding, learned-position ReLU '
                                             'decoder, encoder projection, untied full vocabulary '
                                             'logits and self/cross cache.'},
                   'encoder_input_names': ['input_features', 'attention_mask']},
     'dimension_overrides': {'encoder_config': {'hidden_size': 320,
                                                'num_attention_heads': 2,
                                                'intermediate_size': 1280,
                                                'num_hidden_layers': 2},
                             'hidden_size': 256,
                             'num_attention_heads': 2,
                             'num_key_value_heads': 2,
                             'intermediate_size': 1024,
                             'num_hidden_layers': 2},
     'dimension_purpose': 'Two homogeneous encoder and decoder layers retain encoder head width 160, decoder '
                          'head width 128, ordinary multihead grouping and 4x feed-forward ratios. Preserve '
                          '128 mel bands, factor-8 subsampling, 256 frontend channels, kernel-9 convolution and '
                          'the full 16,384-token vocabulary. 129 frames cross all subsampling stages with odd '
                          'boundaries; a 7-token prefix plus two continuations tests state growth. CPU '
                          'controls also cover a padded 105-frame sample and decoder key padding.',
     'input': {'kind': 'spectrogram',
               'name': 'input_features',
               'batch_size': 2,
               'shape': [129, 128],
               'attention_mask_length': 129,
               'decoder_sequence_length': 9,
               'decoder_start_token_id': 4},
     'workload': 'seq2seq_continuation',
     'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
     'reference_backend': 'sdpa'},

    'colmodernvbert': {
        'reference': {
            'config_class': 'transformers:ColModernVBertConfig',
            'model_class': 'transformers:ColModernVBertForRetrieval',
            'source': {
                'kind': 'composite_checkpoints',
                'components': {
                    'vlm_config': {
                        'checkpoint': 'ModernVBERT/colmodernvbert-merged',
                        'revision': 'a8b921f15ec3c8ba4264d40e1dbd7d77ea1336f0',
                        'url': 'https://huggingface.co/ModernVBERT/colmodernvbert-merged/blob/a8b921f15ec3c8ba4264d40e1dbd7d77ea1336f0/config.json',
                    },
                },
                'description': ('Native task documented checkpoint. Retrieval checkpoint serializes the base '
                    'modernvbert config; wrap it in native ColModernVBertConfig.vlm_config with constructor'
                    ' projection128.'),
            },
        },
        'dimension_overrides': {
            'vlm_config': {
                'text_config': {
                    'hidden_size': 64, 'intermediate_size': 128, 'num_attention_heads': 4, 'vocab_size': 1024,
                    'cls_token_id': 1019, 'sep_token_id': 1020, 'pad_token_id': 0, 'bos_token_id': 1019,
                    'eos_token_id': 1020,
                },
                'vision_config': {'hidden_size': 64, 'intermediate_size': 128, 'num_attention_heads': 4},
                'image_token_id': 1023,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 145,
            'shape': [1, 3, 512, 512], 'image_token_positions': list(range(1, 65)),
        },
        'workload': 'forward',
        'outputs': ['embeddings', 'image_hidden_states'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduce widths/vocab only, retain512image/1024patches/64merged image tokens/22text+12vision '
            'layers/128local window with145tokens; full nonpadded image+text sequence, optional text/image '
            'padding masks omitted.'),
    },

    'colpali': {
        'reference': {
            'config_class': 'transformers:ColPaliConfig',
            'model_class': 'transformers:ColPaliForRetrieval',
            'source': {
                'kind': 'composite_checkpoints',
                'components': {
                    'vlm_config': {
                        'checkpoint': 'vidore/colpaligemma-3b-pt-448-base',
                        'revision': '30ab955d073de4a91dc5a288e8c97226647e3e5a',
                        'url': 'https://huggingface.co/vidore/colpaligemma-3b-pt-448-base/blob/30ab955d073de4a91dc5a288e8c97226647e3e5a/config.json',
                    },
                },
                'description': ('Pinned documented vidore/colpali-v1.2 adapter@6e6cf485 names this accessible PaliGemma'
                    ' base; native ColPali wrapper projection128. Native processor document inputs provide '
                    'allzero token_type_ids, enabling bidirectional prefix.'),
            },
        },
        'dimension_overrides': {
            'vlm_config': {
                'projection_dim': 64,
                'hidden_size': 64,
                'vocab_size': 1024,
                'image_token_index': 1023,
                'text_config': {'hidden_size': 64, 'intermediate_size': 128, 'head_dim': 8, 'vocab_size': 1024},
                'vision_config': {'hidden_size': 64, 'intermediate_size': 128, 'projection_dim': 64},
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 1030,
            'shape': [3, 448, 448], 'image_token_positions': list(range(0, 1024)), 'attention_mask': True,
            'token_type_id': 0,
        },
        'workload': 'forward',
        'outputs': ['embeddings', 'image_hidden_states', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduce widths/vocabulary only; retain448image/1024patches/27vision+18Gemma layers,8Q:1KV '
            'grouped attention, native128retrieval projection,1030document tokens and fulldefaultK/V '
            'outputs.'),
    },

    'colqwen2': {
        'reference': {
            'config_class': 'transformers:ColQwen2Config',
            'model_class': 'transformers:ColQwen2ForRetrieval',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'vidore/colqwen2-v1.0-hf',
                'revision': 'ddc07d2317c80f75fc742b7362ee9ad1912908f9',
                'url': 'https://huggingface.co/vidore/colqwen2-v1.0-hf/blob/ddc07d2317c80f75fc742b7362ee9ad1912908f9/config.json',
                'description': 'Pinned HF documented configuration checkpoint; image retrieval with native sequential positions.',
            },
            'prefill_input_names': ['pixel_values', 'image_grid_thw'],
        },
        'dimension_overrides': {
            'vlm_config': {
                'image_token_id': 900,
                'video_token_id': 901,
                'vision_config': {'depth': 2, 'embed_dim': 128, 'hidden_size': 768, 'num_heads': 2},
                'text_config': {
                    'vocab_size': 1024, 'hidden_size': 768, 'intermediate_size': 1536, 'num_hidden_layers': 2,
                    'num_attention_heads': 6, 'num_key_value_heads': 1, 'max_position_embeddings': 2048,
                },
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'sequence_length': 24, 'image_batch_size': 1,
            'shape': [20, 1176], 'image_grid_thw': [[1, 4, 4]], 'image_token_positions': [4, 5, 6, 7],
        },
        'workload': 'causal_lm',
        'outputs': ['embeddings', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Image retrieval head128dims;quickGELU vision;6:1 GQA and128head width;native MRoPE sections. '
            'Padded patch container exercises unpadding. Two layers,prefill/decode,all returned caches.'),
    },

    'conditional_detr': {
        'reference': {
            'config_class': 'transformers:ConditionalDetrConfig',
            'model_class': 'transformers:ConditionalDetrForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/conditional-detr-resnet-50',
                'revision': '8f8795fb7c319c7862d4f4cd699e76bb09cf2593',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/conditional_detr/modeling_conditional_detr.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 800, 1066]},
        'outputs': ['logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'convbert': {
        'reference': {
            'config_class': 'transformers:ConvBertConfig',
            'model_class': 'transformers:ConvBertForMaskedLM',
            'source': {
                'checkpoint': 'YituTech/conv-bert-base',
                'revision': '5cb451936b5c4a96562d8b146de85f64f9cf2c22',
                'url': 'https://huggingface.co/YituTech/conv-bert-base/blob/5cb451936b5c4a96562d8b146de85f64f9cf2c22/config.json',
                'kind': 'example_checkpoint',
                'description': ('Preserve ConvBertForMaskedLM with the named base checkpoint: full-width embeddings, '
                    'ungrouped FFNs, reduced dense-attention heads, and the required kernel9 dynamic '
                    'span-convolution branch.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'convnext': {
        'reference': {
            'config_class': 'transformers.models.convnext.configuration_convnext:ConvNextConfig',
            'model_class': 'transformers.models.convnext.modeling_convnext:ConvNextModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs ConvNextModel from configuration '
                    'constructor defaults with random weights.'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/convnext/configuration_convnext.py#L31-L43',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'convnextv2': {
        'reference': {
            'config_class': 'transformers.models.convnextv2.configuration_convnextv2:ConvNextV2Config',
            'model_class': 'transformers.models.convnextv2.modeling_convnextv2:ConvNextV2Model',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs ConvNextV2Model from configuration '
                    'constructor defaults with random weights. The example misspells Config capitalization;'
                    ' use the actual documented ConvNextV2Config class.'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/convnextv2/configuration_convnextv2.py#L31-L43',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'cpmant': {
        'reference': {
            'config_class': 'transformers:CpmAntConfig',
            'model_class': 'transformers:CpmAntForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'openbmb/cpm-ant-10b',
                'revision': '3b53c0f95f625de6ae676f9edf2cc65930f8b4b8',
                'url': 'https://huggingface.co/openbmb/cpm-ant-10b/blob/3b53c0f95f625de6ae676f9edf2cc65930f8b4b8/config.json',
                'description': 'Pinned HF public forward example checkpoint.',
            },
            'decode_input': 'full_history',
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'ctrl': {
        'reference': {
            'config_class': 'transformers:CTRLConfig',
            'model_class': 'transformers:CTRLLMHeadModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'Salesforce/ctrl',
                'revision': '3ff4f697e86bb0b95f7a70e39822af96671727d9',
                'url': 'https://huggingface.co/Salesforce/ctrl/blob/3ff4f697e86bb0b95f7a70e39822af96671727d9/config.json',
                'description': 'Pinned HF public forward example checkpoint.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'cvt': {
        'reference': {
            'config_class': 'transformers.models.cvt.configuration_cvt:CvtConfig',
            'model_class': 'transformers.models.cvt.modeling_cvt:CvtModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned CvtConfig example explicitly constructs CvtModel(CvtConfig()). Retain that '
                    'public base-model task with all default stages and already computed outputs.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/cvt/configuration_cvt.py#L53',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 384, 384]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'cls_token_value'],
        'reference_backend': None,
    },

    'cwm': {
        'workload': 'causal_lm_continuation',
        'reference_backend': 'sdpa',
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:CwmConfig',
            'model_class': 'transformers:CwmForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'constructor_defaults',
                'description': ('Complete pinned CwmConfig defaults: task placeholder unavailable; documented '
                    'facebook/cwm config gated (403), not a loaded checkpoint configuration.'),
            },
        },
        'config_overrides': {},
        'dimension_overrides': {
            'hidden_size': 384, 'intermediate_size': 1344, 'num_attention_heads': 6, 'num_key_value_heads': 1,
            'head_dim': 64, 'num_hidden_layers': 4, 'vocab_size': 1024, 'max_position_embeddings': 1024,
            'sliding_window': 64, 'bos_token_id': 1, 'eos_token_id': [2, 3, 4],
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 258},
        'outputs': ['logits', 'past_key_values'],
        'dimension_purpose': ('Retains GQA6:1, MLP/H3.5, one whole FSSS layer group, and Llama-3 factor16 frequency scaling. '
            'Context/window bounds both shrink128x; 256-token prompt exceeds window64. Development only.'),
    },

    'd_fine': {
        'reference': {
            'config_class': 'transformers:DFineConfig',
            'model_class': 'transformers:DFineForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ustc-community/dfine-xlarge-coco',
                'revision': 'ea4f6be7350bbe3c199ec6febc74168346cb5a68',
                'description': 'Pinned public object detection example checkpoint ustc-community/dfine-xlarge-coco.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/d_fine/modeling_d_fine.py',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 640]},
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'intermediate_hidden_states', 'intermediate_logits',
            'intermediate_reference_points', 'intermediate_predicted_corners', 'initial_reference_points',
            'encoder_last_hidden_state', 'init_reference_points', 'enc_topk_logits', 'enc_topk_bboxes',
            'enc_outputs_class', 'enc_outputs_coord_logits',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'dab_detr': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:DabDetrConfig',
            'model_class': 'transformers:DabDetrForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'IDEA-Research/dab-detr-resnet-50',
                'revision': 'd8e2856ee1f7a28088f0b8069ceebf2a44cb5042',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/dab_detr/modeling_dab_detr.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 800, 1066]},
        'outputs': ['logits', 'pred_boxes', 'last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'dac': {
        'reference': {
            'config_class': 'transformers:DacConfig',
            'model_class': 'transformers:DacModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'descript/dac_16khz',
                'revision': '7c2fc5e759f1f501aefc6e7a0265cc57f5d17ba7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/dac/modeling_dac.py#L657',
                'description': ('Pinned HF DacModel forward example selects descript/dac_16khz; preserve all12 '
                    'quantizers and encode/decode.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [1, 16384]},
        'workload': 'forward',
        'outputs': ['loss', 'audio_values', 'quantized_representation', 'audio_codes', 'projected_latents'],
        'reference_backend': None,
    },

    'data2vec_audio': {
        'reference': {
            'config_class': 'transformers:Data2VecAudioConfig',
            'model_class': 'transformers:Data2VecAudioModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/data2vec/configuration_data2vec_audio.py#L112',
                'description': ('Preserve constructor-example defaults, including all five grouped positional '
                    'convolutions of kernel 19 with nonaffine LayerNorm, and the post-normalized encoder.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [48000]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'extract_features'],
        'reference_backend': None,
    },

    'data2vec_text': {
        'reference': {
            'config_class': 'transformers:Data2VecTextConfig',
            'model_class': 'transformers:Data2VecTextForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/data2vec-text-base',
                'revision': 'bd0db19c3500ee7a0b626791db67fa6e9fda9a0b',
                'url': 'https://huggingface.co/facebook/data2vec-text-base/blob/bd0db19c3500ee7a0b626791db67fa6e9fda9a0b/config.json',
                'description': ('Preserve the corpus masked-LM task using the checkpoint named by pinned HF '
                    'src/transformers/models/data2vec/configuration_data2vec_text.py:22; complete omitted '
                    'values from the pinned constructor.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'data2vec_vision': {
        'reference': {
            'config_class': 'transformers.models.data2vec.configuration_data2vec_vision:Data2VecVisionConfig',
            'model_class': 'transformers.models.data2vec.modeling_data2vec_vision:Data2VecVisionModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs Data2VecVisionConfig() and '
                    'Data2VecVisionModel(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/data2vec/configuration_data2vec_vision.py#L46',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'deberta': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:DebertaConfig',
            'model_class': 'transformers:DebertaForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned HF configuration documentation names microsoft/deberta-base; use its checkpoint'
                    ' config, completed by pinned constructor defaults.'),
                'checkpoint': 'microsoft/deberta-base',
                'revision': '0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195',
                'url': 'https://huggingface.co/microsoft/deberta-base/blob/0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 640},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'deberta_v2': {
        'reference': {
            'config_class': 'transformers:DebertaV2Config',
            'model_class': 'transformers:DebertaV2ForMaskedLM',
            'source': {
                'checkpoint': 'microsoft/deberta-v2-xlarge',
                'revision': '1d134961d4db8e7e8eb1bc1ab81cb370244c57f7',
                'url': 'https://huggingface.co/microsoft/deberta-v2-xlarge/blob/1d134961d4db8e7e8eb1bc1ab81cb370244c57f7/config.json',
                'kind': 'example_checkpoint',
                'description': ('Preserve the legacy DebertaV2ForMaskedLM task and its named xlarge checkpoint: shared '
                    'content/position projections, both c2p and p2c scores, 256 relative-position buckets '
                    'with LayerNorm, and kernel3 GELU input convolution.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'decision_transformer': {
        'reference': {
            'config_class': 'transformers:DecisionTransformerConfig',
            'model_class': 'transformers:DecisionTransformerModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'edbeeching/decision-transformer-gym-hopper-medium',
                'revision': '8224ec324200b150f10287b8c8c525224e62f319',
                'description': ('Pinned DecisionTransformerModel example selects the official Hopper-medium checkpoint;'
                    ' preserve checkpoint dimensions and return/state/action predictions.'),
            },
        },
        'input': {'kind': 'trajectory', 'batch_size': 1, 'sequence_length': 67},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'state_preds', 'action_preds', 'return_preds'],
        'reference_backend': None,
    },

    'deepseek_v2': {
        'reference': {
            'config_class': 'transformers:DeepseekV2Config',
            'model_class': 'transformers:DeepseekV2ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'deepseek-ai/DeepSeek-V2-Lite',
                'revision': '604d5664dddd88a0433dbae533b7fe9472482de0',
                'url': 'https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json',
                'description': 'Pinned HF documented official Lite checkpoint, including direct queries.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 3, 'num_attention_heads': 2, 'num_key_value_heads': 2, 'kv_lora_rank': 128,
            'vocab_size': 1024, 'bos_token_id': 1022, 'eos_token_id': 1023,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve direct query path,128+64 QK subspaces/value128, dense first then64 '
            'experts/top6/shared2, unnormalized softmax routing and native YaRN. Reduce latent width512 '
            'to128/model/layers. Expanded HF attention caches are larger than compressed MLA.'),
    },

    'deepseek_v3': {
        'reference': {
            'config_class': 'transformers:DeepseekV3Config',
            'model_class': 'transformers:DeepseekV3ForCausalLM',
            'load_device': 'cuda',
            'serialized_weight_suffixes': ['weight_scale_inv', 'gate_up_proj_scale_inv', 'down_proj_scale_inv'],
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'deepseek-ai/DeepSeek-V3',
                'revision': 'e815299b0bcbac849fa540c768ef21845365c9eb',
                'url': 'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json',
                'description': 'Root-approved matching author fallback after invalid/mismatched pinned HF documentation. Native block FP8 retained.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 896, 'intermediate_size': 512, 'moe_intermediate_size': 128,
            'num_hidden_layers': 5, 'num_attention_heads': 16, 'num_key_value_heads': 16, 'q_lora_rank': 384,
            'vocab_size': 1024,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 129},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Keep native blockFP8 weights/dynamicactivationquantization,256experts/top8, attention head '
            'dimensions and all enabled layer types; reduce hidden/intermediate widths and layer count for '
            'development.'),
    },

    'deepseek_vl': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:DeepseekVLConfig',
            'model_class': 'transformers:DeepseekVLForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'deepseek-community/deepseek-vl-1.3b-chat',
                'revision': 'aa84ba9a033b7201d3a0a2e871291f0bb9c27329',
                'description': ('Pinned HF DeepseekVLConfig documentation identifies the published 1.3B chat '
                    'configuration; complete conditional-generation task.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/deepseek_vl/configuration_deepseek_vl.py',
            },
            'prefill_input_names': ['pixel_values'],
        },
        'dimension_overrides': {
            'vision_config': {
                'image_size': 64, 'hidden_size': 128, 'intermediate_size': 512, 'num_hidden_layers': 2,
                'num_attention_heads': 4,
            },
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 512, 'num_hidden_layers': 2,
                'num_attention_heads': 2, 'num_key_value_heads': 2,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 64, 64],
            'sequence_length': 40, 'image_token_positions': list(range(3, 19)),
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve RGB patch16 SigLIP without optional pooling head, exact GELU, two-layer alignment '
            'MLP, MHA Llama with native head_dim128/defaultRoPE and cache. Reduced depth/width/image size '
            'only.'),
    },

    'deepseek_vl_hybrid': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:DeepseekVLHybridConfig',
            'model_class': 'transformers:DeepseekVLHybridForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'deepseek-community/deepseek-vl-7b-chat',
                'revision': '4d6f7f4aad464b5426a5321d176a11da6191ba60',
                'description': ('Pinned HF documented DeepSeek-VL hybrid 7B chat checkpoint; both low-resolution SigLIP'
                    ' and high-resolution SAM branches retained.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/deepseek_vl_hybrid/modeling_deepseek_vl_hybrid.py',
            },
            'prefill_input_names': ['pixel_values', 'high_res_pixel_values'],
        },
        'dimension_overrides': {
            'vision_config': {
                'image_size': 80, 'hidden_size': 128, 'intermediate_size': 512, 'num_hidden_layers': 2,
                'num_attention_heads': 4,
            },
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 512, 'num_hidden_layers': 2,
                'num_attention_heads': 2, 'num_key_value_heads': 2,
            },
            'high_res_vision_config': {
                'image_size': 256, 'hidden_size': 192, 'intermediate_size': 768, 'mlp_dim': 768,
                'num_hidden_layers': 4, 'num_attention_heads': 3, 'output_channels': 64,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 80, 80],
            'sequence_length': 48, 'image_token_positions': list(range(3, 28)),
            'high_res_shape': [3, 256, 256],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': {'': 'sdpa', 'high_res_vision_config': 'eager'},
        'dimension_purpose': ('Retain both vision towers, SAM local-window padding, first global layer2 and distinct final '
            'layer3 branches, learned alpha, nonidentity16-to20 resize and two stride2 projections, '
            'high/low alignment, native Llama head128/defaultRoPE and cache. Reduced depth/width/image '
            'sizes only.'),
    },

    'deformable_detr': {
        'reference': {
            'config_class': 'transformers:DeformableDetrConfig',
            'model_class': 'transformers:DeformableDetrForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'SenseTime/deformable-detr',
                'revision': '83ecd26945199939cb82806f988debdb71e6f43e',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/deformable_detr/modeling_deformable_detr.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 800, 1066]},
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state',
            'intermediate_hidden_states', 'intermediate_reference_points', 'init_reference_points',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'deimv2': {
        'reference': {
            'config_class': 'transformers:Deimv2Config',
            'model_class': 'transformers:Deimv2ForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers',
                'revision': 'b23bfb92684c98e1adf60939fdef28e409991f68',
                'description': ('Pinned public object detection task example '
                    'harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers. Configuration-autodoc Intellindust '
                    'repository has author-format nested config, so use this task-example HF-format config.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/deimv2/modeling_deimv2.py',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 640]},
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'intermediate_hidden_states', 'intermediate_logits',
            'intermediate_reference_points', 'intermediate_predicted_corners', 'initial_reference_points',
            'encoder_last_hidden_state', 'init_reference_points', 'enc_topk_logits', 'enc_topk_bboxes',
            'enc_outputs_class', 'enc_outputs_coord_logits',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'deit': {
        'reference': {
            'config_class': 'transformers.models.deit.configuration_deit:DeiTConfig',
            'model_class': 'transformers.models.deit.modeling_deit:DeiTModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs DeiTModel(DeiTConfig()) with random'
                    ' weights.'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/deit/configuration_deit.py#L33-L46',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'depth_anything': {
        'reference': {
            'config_class': 'transformers:DepthAnythingConfig',
            'model_class': 'transformers:DepthAnythingForDepthEstimation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'LiheYoung/depth-anything-small-hf',
                'revision': '25216a913fa218ccb7d58cce818d52b728b6c1f6',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/depth_anything/modeling_depth_anything.py',
                'description': 'Pinned public depth-estimation task checkpoint, default relative-depth output.',
            },
        },
        'reference_backend': None,
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 518, 686]},
        'outputs': ['predicted_depth'],
        'workload': 'forward',
    },

    'depth_pro': {
        'reference': {
            'config_class': 'transformers:DepthProConfig',
            'model_class': 'transformers:DepthProForDepthEstimation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'apple/DepthPro-hf',
                'revision': 'de816c8ce7168afcb231f96d501d72b869d0beda',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/depth_pro/modeling_depth_pro.py',
                'description': 'Pinned public task example checkpoint, including default field-of-view prediction.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 1536, 1536]},
        'outputs': ['predicted_depth', 'field_of_view'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'detr': {
        'reference': {
            'config_class': 'transformers:DetrConfig',
            'model_class': 'transformers:DetrForSegmentation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/detr-resnet-50-panoptic',
                'revision': 'd53b52a799403a8867920f82c869e40732b47037',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/detr/modeling_detr.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 800, 1066]},
        'outputs': ['logits', 'pred_boxes', 'pred_masks', 'last_hidden_state', 'encoder_last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'dia': {'reference': {'config_class': 'transformers:DiaConfig',
                           'model_class': 'transformers:DiaForConditionalGeneration',
                           'source': {'kind': 'constructor_defaults',
                                      'description': 'Pinned HF conversion script convert_dia_to_hf.py '
                                                     'explicitly initializes DiaConfig as author '
                                                     'Dia1.6B; public author checkpoint config uses a '
                                                     'different schema. Verified author checkpoint '
                                                     'nari-labs/Dia-1.6B@257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd; '
                                                     'retain all9audiochannels.'},
                           'forward_kwargs': {'use_cache': True},
                           'reuse_encoder': True},
             'dimension_overrides': {'encoder_config': {'hidden_size': 64,
                                                        'num_hidden_layers': 2,
                                                        'num_attention_heads': 4,
                                                        'num_key_value_heads': 4,
                                                        'head_dim': 32,
                                                        'intermediate_size': 128},
                                     'decoder_config': {'hidden_size': 96,
                                                        'num_hidden_layers': 2,
                                                        'num_attention_heads': 4,
                                                        'num_key_value_heads': 1,
                                                        'head_dim': 32,
                                                        'intermediate_size': 192,
                                                        'cross_num_attention_heads': 4,
                                                        'cross_num_key_value_heads': 4,
                                                        'cross_head_dim': 32,
                                                        'cross_hidden_size': 64}},
             'reference_backend': {'': 'sdpa', 'encoder_config': 'sdpa', 'decoder_config': 'sdpa'},
             'input': {'kind': 'seq2seq_tokens',
                       'batch_size': 1,
                       'encoder_sequence_length': 33,
                       'decoder_sequence_length': 19,
                       'encoder_vocab_size': 256,
                       'decoder_vocab_size': 1024,
                       'decoder_channels': 9,
                       'decoder_start_token_id': 1026,
                       'encoder_prefix_token_ids': [],
                       'encoder_suffix_token_ids': [],
                       'description': 'Synthetic byte text and nine-channel audio token inference. '
                                      'Keep33 text steps and17 audio prefill frames plus2 supplied '
                                      'continuations. Each channel starts with native BOS1026; other '
                                      'frames contain code IDs0–1023. Sampling bounds retain full text '
                                      'range and exclude reserved audio IDs; model vocabulary1028 and '
                                      'full logits remain unchanged.'},
             'workload': 'seq2seq_continuation',
             'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
             'dimension_purpose': 'Inherited two-layer encoder/decoder dimensions are unchanged by '
                                  'this cache-interface and input-preparation fix: encoder64/128, '
                                  'decoder96/192, attention head32, decoder query/KV4:1, cross '
                                  'query/KV4:4. These FFN ratios2:1 and head32 differ from native4:1 '
                                  'and head128 and retain their separate dimension-policy review '
                                  'caveat. All9audio channels and original model vocabularies256/1028 '
                                  'remain.33text steps and17audio prefill frames plus2 supplied frames '
                                  'exercise self-cache growth and constant cross-cache reuse.'},

    'diffllama': {
        'reference': {
            'config_class': 'transformers:DiffLlamaConfig',
            'model_class': 'transformers:DiffLlamaForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'kajuma/DiffLlama-0.3B-handcut',
                'revision': 'c2ad1209f1c1ebd4d4ab2eaf5c8002b541b87140',
                'url': 'https://huggingface.co/kajuma/DiffLlama-0.3B-handcut/blob/c2ad1209f1c1ebd4d4ab2eaf5c8002b541b87140/config.json',
                'description': ('Forward example google/diffllama-7b is inaccessible404. Use checkpoint explicitly '
                    'named by pinned DiffLlamaConfig autodoc, preserving its tied head, GQA and Llama3 '
                    'scaling instead of historical default overrides.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'dinov2': {
        'reference': {
            'config_class': 'transformers.models.dinov2.configuration_dinov2:Dinov2Config',
            'model_class': 'transformers.models.dinov2.modeling_dinov2:Dinov2Model',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs Dinov2Config() and '
                    'Dinov2Model(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/dinov2/configuration_dinov2.py#L42',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'dinov2_with_registers': {
        'reference': {
            'config_class': 'transformers.models.dinov2_with_registers.configuration_dinov2_with_registers:Dinov2WithRegistersConfig',
            'model_class': 'transformers.models.dinov2_with_registers.modeling_dinov2_with_registers:Dinov2WithRegistersModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs Dinov2WithRegistersConfig() and '
                    'Dinov2WithRegistersModel(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/dinov2_with_registers/configuration_dinov2_with_registers.py#L47',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'dinov3_convnext': {
        'reference': {
            'config_class': 'transformers.models.dinov3_convnext.configuration_dinov3_convnext:DINOv3ConvNextConfig',
            'model_class': 'transformers.models.dinov3_convnext.modeling_dinov3_convnext:DINOv3ConvNextModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs DINOv3ConvNextConfig() and '
                    'DINOv3ConvNextModel(config).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/dinov3_convnext/configuration_dinov3_convnext.py#L28',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'dinov3_vit': {
        'reference': {
            'config_class': 'transformers.models.dinov3_vit.configuration_dinov3_vit:DINOv3ViTConfig',
            'model_class': 'transformers.models.dinov3_vit.modeling_dinov3_vit:DINOv3ViTModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs DINOv3ViTConfig() and '
                    'DINOv3ViTModel(config).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/dinov3_vit/configuration_dinov3_vit.py#L59',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'distilbert': {
        'reference': {
            'config_class': 'transformers:DistilBertConfig',
            'model_class': 'transformers:DistilBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Preserve DistilBertForMaskedLM; pinned HF documentation identifies the base uncased '
                    'checkpoint and links the canonical distilbert release organization.'),
                'checkpoint': 'distilbert/distilbert-base-uncased',
                'revision': '12040accade4e8a0f71eabdb258fecc2e7e948be',
                'url': 'https://huggingface.co/distilbert/distilbert-base-uncased/blob/12040accade4e8a0f71eabdb258fecc2e7e948be/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'donut_swin': {
        'reference': {
            'config_class': 'transformers:DonutSwinConfig',
            'model_class': 'transformers:DonutSwinModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/donut/configuration_donut_swin.py',
                'description': ('Pinned configuration documentation explicitly constructs the base model with '
                    'constructor defaults; preserve all four stages and default pooling.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['last_hidden_state', 'pooler_output'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'dots1': {
        'reference': {
            'config_class': 'transformers:Dots1Config',
            'model_class': 'transformers:Dots1ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'rednote-hilab/dots.llm1.base',
                'revision': 'e73576bb6d3b810e616a949aa8f785d6940cbe51',
                'url': 'https://huggingface.co/rednote-hilab/dots.llm1.base/blob/e73576bb6d3b810e616a949aa8f785d6940cbe51/config.json',
                'description': 'Pinned HF model documentation checkpoint; preserve enabled inference computations.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 3, 'num_attention_heads': 2, 'num_key_value_heads': 2, 'vocab_size': 1024,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain original head width128/MHA, first dense then routed layers,128 experts/top6 and two '
            'shared experts, full attention and sigmoid/grouped correction routing.'),
    },

    'dpr': {
        'reference': {
            'config_class': 'transformers:DPRConfig',
            'model_class': 'transformers:DPRQuestionEncoder',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned DPRQuestionEncoder forward example loads the single-nq-base question '
                    'checkpoint; preserve the unprojected first-token output and automatic padding mask.'),
                'checkpoint': 'facebook/dpr-question_encoder-single-nq-base',
                'revision': 'd04a52f6d2f96c60117a925e8c24c4043a75f265',
                'url': 'https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/blob/d04a52f6d2f96c60117a925e8c24c4043a75f265/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'forward',
        'outputs': ['pooler_output'],
        'reference_backend': None,
    },

    'dpt': {
        'reference': {
            'config_class': 'transformers:DPTConfig',
            'model_class': 'transformers:DPTModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/dpt/configuration_dpt.py#L69-L80',
                'description': 'The pinned base-model example explicitly constructs DPTConfig() and DPTModel(configuration).',
            },
        },
        'input': {'kind': 'image', 'shape': [3, 384, 384], 'batch_size': 1},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'edgetam': {
        'reference': {
            'config_class': 'transformers:EdgeTamConfig',
            'model_class': 'transformers:EdgeTamModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'yonigozlan/EdgeTAM-hf',
                'revision': 'c266ce53b3fc00f0f495b583f6a116c4e57f53bb',
                'description': ('Pinned HF configuration example checkpoint; explicitly select its EdgeTamModel image '
                    'subset, with a positive point prompt. The checkpoint also contains video components.'),
            },
        },
        'input': {'kind': 'external', 'shape': [3, 1024, 1024], 'batch_size': 1},
        'outputs': ['pred_masks', 'iou_scores', 'object_score_logits', 'image_embeddings'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'edgetam_video': {
        'reference': {
            'config_class': 'transformers:EdgeTamVideoConfig',
            'model_class': 'transformers:EdgeTamVideoModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'yonigozlan/EdgeTAM-hf',
                'revision': 'c266ce53b3fc00f0f495b583f6a116c4e57f53bb',
                'description': 'Pinned HF configuration auto_docstring checkpoint; main model_doc IDs return404.',
            },
        },
        'input': {'kind': 'external', 'name': 'video', 'shape': [18, 3, 1024, 1024], 'batch_size': 1},
        'outputs': ['pred_masks', 'object_score_logits'],
        'workload': 'edgetam_video',
        'reference_backend': None,
    },

    'efficientloftr': {
        'reference': {
            'config_class': 'transformers:EfficientLoFTRConfig',
            'model_class': 'transformers:EfficientLoFTRForKeypointMatching',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'zju-community/efficientloftr',
                'revision': 'face1a79050ffa3e9da28720d1cf93aaf2e8f421',
                'description': 'Pinned HF configuration example; forward example efficient_loftr is stale404.',
            },
        },
        'input': {'kind': 'image', 'shape': [2, 3, 480, 640], 'batch_size': 1},
        'outputs': ['matches', 'matching_scores', 'keypoints'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'efficientnet': {
        'reference': {
            'config_class': 'transformers.models.efficientnet.configuration_efficientnet:EfficientNetConfig',
            'model_class': 'transformers.models.efficientnet.modeling_efficientnet:EfficientNetModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': 'The pinned public EfficientNetModel example constructs EfficientNetConfig().',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/efficientnet/configuration_efficientnet.py#L51',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 600, 600]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'electra': {
        'reference': {
            'config_class': 'transformers:ElectraConfig',
            'model_class': 'transformers:ElectraForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Preserve the existing ElectraForMaskedLM task using the official small generator; its '
                    'computational config matches the small discriminator named by pinned ElectraConfig, '
                    'including the active 128-to-256 embedding projection.'),
                'checkpoint': 'google/electra-small-generator',
                'revision': '0dbbbf928f000ba9d2be7eb769dee35e8d9d6e39',
                'url': 'https://huggingface.co/google/electra-small-generator/blob/0dbbbf928f000ba9d2be7eb769dee35e8d9d6e39/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'emu3': {
        'reference': {
            'config_class': 'transformers:Emu3Config',
            'model_class': 'transformers:Emu3ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'BAAI/Emu3-Chat-hf',
                'revision': '414c0a163edad789827ee473a71b75c7de546347',
                'description': 'Pinned native public image-conditioned ordinary text generation; PerceptionLM also exercises video inputs.',
            },
            'generation_config': {
                'do_sample': True, 'eos_token_id': 151850, 'max_new_tokens': 50000, 'pad_token_id': 151643,
                'top_k': 2048, 'transformers_version': '4.47.0.dev0',
            },
            'generation_output_name': 'sequences',
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 512, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 1, 'max_position_embeddings': 512,
            },
            'vq_config': {'base_channels': 32, 'hidden_size': 128},
        },
        'input': {
            'kind': 'external_prepared',
            'description': 'Explicit common BF16 pixel tensors plus token IDs; full vocabulary and native image placeholder layout.',
        },
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 4, 'do_sample': False},
        'outputs': ['sequences'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Shrink depth and widths while retaining native head widths, full vocabulary, image encoding '
            'branches, all modal inputs, native pooling and connector ratios. Four ordinary autoregressive '
            'steps.'),
    },

    'encodec': {
        'reference': {
            'config_class': 'transformers:EncodecConfig',
            'model_class': 'transformers:EncodecModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/encodec_24khz',
                'revision': 'c1dbe2ae3f1de713481a3b3e7c47f357092ee040',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/encodec/modeling_encodec.py#L775',
                'description': ('Pinned HF public forward example selects facebook/encodec_24khz; omitted bandwidth '
                    'selects first configured1.5kbps (2 active quantizers), full encode/decode.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [1, 24001]},
        'workload': 'forward',
        'outputs': ['audio_codes', 'audio_values'],
        'reference_backend': None,
    },

    'encoder_decoder': {
        'reference': {
            'config_class': 'transformers:EncoderDecoderConfig',
            'model_class': 'transformers:EncoderDecoderModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'patrickvonplaten/bert2bert-cnn_dailymail-fp16',
                'revision': '51b5d5cac0fa0ed09ed505df5800579996a2fe12',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/encoder-decoder.md#L43-L44',
                'description': ('Pinned public usage example selects this BERT encoder + causal BERT decoder '
                    'checkpoint. Preserve enabled decoder cross-attention, matching encoder/decoder width, '
                    'decoder embedding/head tying and ordinary returned cache. There is no universal '
                    'default tower pair.'),
            },
            'native_cache_defaults': True,
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 129,
            'decoder_sequence_length': 33,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'eomt': {
        'reference': {
            'config_class': 'transformers:EomtConfig',
            'model_class': 'transformers:EomtForUniversalSegmentation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'tue-mps/coco_panoptic_eomt_large_640',
                'revision': 'dcd130bed9b1ebda7041fd660fddb16f905b9c3b',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/eomt/configuration_eomt.py',
                'description': 'Pinned HF documented EoMT large checkpoint configuration.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 640]},
        'outputs': ['class_queries_logits', 'masks_queries_logits', 'last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'eomt_dinov3': {
        'reference': {
            'config_class': 'transformers:EomtDinov3Config',
            'model_class': 'transformers:EomtDinov3ForUniversalSegmentation',
            'source': {
                'kind': 'constructor_defaults',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/eomt_dinov3/configuration_eomt_dinov3.py',
                'description': ('Pinned HF constructor-defined architecture, including its two-label head. This is not '
                    'claimed equivalent to the distinct checkpoints named in the configuration and task '
                    'documentation.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 640]},
        'outputs': ['class_queries_logits', 'masks_queries_logits', 'last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'ernie': {
        'reference': {
            'config_class': 'transformers:ErnieConfig',
            'model_class': 'transformers:ErnieForPreTraining',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned ErnieForPreTraining forward example loads ERNIE 1.0. Preserve both prediction '
                    'and relationship logits, its ReLU activation, and completed use_task_id=false.'),
                'checkpoint': 'nghuyong/ernie-1.0-base-zh',
                'revision': '79091d8ab370d2dffccdf61ea2153399cb4a121d',
                'url': 'https://huggingface.co/nghuyong/ernie-1.0-base-zh/blob/79091d8ab370d2dffccdf61ea2153399cb4a121d/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'forward',
        'outputs': ['prediction_logits', 'seq_relationship_logits'],
        'reference_backend': None,
    },

    'ernie4_5': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:Ernie4_5Config',
            'model_class': 'transformers:Ernie4_5ForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'baidu/ERNIE-4.5-0.3B-PT',
                'revision': 'b565cf6caebdb7a1eadf00100857b1ed5e044f12',
                'description': 'Pinned configuration documentation example for the public causal-LM class.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'ernie4_5_moe': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:Ernie4_5_MoeConfig',
            'model_class': 'transformers:Ernie4_5_MoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'baidu/ERNIE-4.5-21B-A3B-PT',
                'revision': '87db95487941cb39592ee0abca3b9155a6d19c5c',
                'url': 'https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/87db95487941cb39592ee0abca3b9155a6d19c5c/config.json',
                'description': 'Pinned HF model documentation checkpoint; preserve enabled inference computations.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 640, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 3, 'num_attention_heads': 5, 'num_key_value_heads': 1, 'vocab_size': 1024,
            'moe_layer_end_index': 2,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain head width128 and5:1 GQA, first dense then routed layers,64 experts/top6/two shared '
            'experts, FP32 router and FP32 interleaved RoPE.'),
    },

    'ernie4_5_vl_moe': {
        'reference': {
            'config_class': 'transformers:Ernie4_5_VLMoeConfig',
            'model_class': 'transformers:Ernie4_5_VLMoeForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'baidu/ERNIE-4.5-VL-28B-A3B-PT',
                'revision': 'e3815e65c607ea211bfe21b46ab0cd264b76731c',
                'url': 'https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT/blob/e3815e65c607ea211bfe21b46ab0cd264b76731c/config.json',
                'description': ('Pinned official native HF task example; equal author expert-count list requires '
                    'explicit representation correction to native scalar, preserving independent '
                    'text/vision groups.'),
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
            'prefill_sequence_input_names': ['mm_token_type_ids', 'moe_mm_token_type_ids'],
        },
        'config_overrides': {'moe_num_experts': 64},
        'dimension_overrides': {
            'hidden_size': 640,
            'intermediate_size': 3072,
            'num_hidden_layers': 3,
            'num_attention_heads': 5,
            'num_key_value_heads': 1,
            'max_position_embeddings': 1024,
            'vocab_size': 512,
            'moe_num_experts': 8,
            'moe_intermediate_size': [192, 64],
            'image_token_id': 400,
            'video_token_id': 401,
            'image_start_token_id': 402,
            'image_end_token_id': 403,
            'video_start_token_id': 404,
            'video_end_token_id': 405,
            'vision_config': {'hidden_size': 160, 'intermediate_size': 640, 'num_heads': 2, 'depth': 2},
        },
        'input': {
            'kind': 'text_image',
            'text_batch_size': 1,
            'sequence_length': 33,
            'image_batch_size': 1,
            'shape': [16, 588],
            'video_shape': [64, 588],
            'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 4, 4]],
            'video_grid_thw': [[4, 4, 4]],
            'image_token_positions': [3, 4, 5, 6],
            'video_token_positions': list(range(10, 18)),
            'input_ids': [
                [
                    1, 3, 402, 400, 400, 400, 400, 403, 4, 404, 401, 401, 401, 401, 401, 401, 401, 401, 405,
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                ],
            ],
            'mm_token_type_ids': True,
            'moe_mm_token_type_ids': [[0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve head128 5:1GQA, firstdense then2modality-isolatedtop6-of8expertlayers, '
            'unequal192/64expertwidths and2sharedexperts, FP32router/floor. '
            'Nativevisionhead80/patch14/QuickGELU, spatial2merge, temporal2resampler duplicateimage and '
            'pair4videoframes, alltext/visionmarkers routeappropriateexperts; original[64,64]expertcounts '
            'normalizedto scalar64before dimensionresize8.'),
    },

    'esm': {
        'reference': {
            'config_class': 'transformers:EsmConfig',
            'model_class': 'transformers:EsmForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/esm-1b',
                'revision': '6fb10624652aa86b64c620ae624baa5b5b8c3b78',
                'url': 'https://huggingface.co/facebook/esm-1b/blob/6fb10624652aa86b64c620ae624baa5b5b8c3b78/config.json',
                'description': ('Pinned configuration documentation names facebook/esm-1b. Its integer dropout 0 is '
                    'explicitly represented as float 0.0 for current strict config parsing; computation '
                    'unchanged.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'config_overrides': {'hidden_dropout_prob': 0.0},
        'outputs': ['logits'],
    },

    'esmfold': {
        'reference': {
            'config_class': 'transformers:EsmConfig',
            'model_class': 'transformers:EsmForProteinFolding',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/esmfold_v1',
                'revision': '75a3841ee059df2bf4d56688166c8fb459ddd97a',
                'description': ('Pinned public protein-folding forward example. Preserve fp16_esm=false, '
                    'max_recycles=4, use_esm_attn_map=false and all 23 public tensor outputs.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/esm/modeling_esmfold.py#L2068',
            },
        },
        'dimension_overrides': {
            'hidden_size': 64,
            'intermediate_size': 128,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'esmfold_config': {
                'lddt_head_hid_dim': 32,
                'trunk': {
                    'num_blocks': 2,
                    'sequence_state_dim': 64,
                    'pairwise_state_dim': 32,
                    'sequence_head_width': 16,
                    'pairwise_head_width': 16,
                    'structure_module': {
                        'sequence_dim': 64, 'pairwise_dim': 32, 'ipa_dim': 8, 'resnet_dim': 32,
                        'num_blocks': 2, 'num_heads_ipa': 4,
                    },
                },
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 8, 'vocab_size': 20},
        'workload': 'forward',
        'outputs': [
            'frames', 'sidechain_frames', 'unnormalized_angles', 'angles', 'positions', 'states', 's_s',
            's_z', 'distogram_logits', 'lm_logits', 'aatype', 'atom14_atom_exists', 'residx_atom14_to_atom37',
            'residx_atom37_to_atom14', 'atom37_atom_exists', 'residue_index', 'lddt_head', 'plddt',
            'ptm_logits', 'ptm', 'aligned_confidence_probs', 'predicted_aligned_error',
            'max_predicted_aligned_error',
        ],
        'reference_backend': 'eager',
        'dimension_purpose': ('Reduce ESM/trunk/structure depth and widths for development while retaining rotary ESM-2 '
            'token-dropout compensation, both triangular multiplication directions and both axial '
            'attentions, gated sequence/pair blocks, invariant point attention, quaternion updates, all '
            'seven torsion angles, all sidechain frames and atom14 positions, confidence heads, and all '
            'four default recycles. Eight ordinary amino-acid input IDs use the public AF2 alphabet (20 '
            'residues), not the 33-token ESM alphabet. All 23 outputs retained; no pretrained-quality or '
            'checkpoint-size performance claim.'),
    },

    'eurobert': {
        'reference': {
            'config_class': 'transformers:EuroBertConfig',
            'model_class': 'transformers:EuroBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned EuroBertForMaskedLM forward example loads EuroBERT-210m; retain bidirectional '
                    'multi-head attention, RoPE theta 250000, SwiGLU, RMSNorm, and the untied bias-free '
                    'vocabulary head.'),
                'checkpoint': 'EuroBERT/EuroBERT-210m',
                'revision': '39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e',
                'url': 'https://huggingface.co/EuroBERT/EuroBERT-210m/blob/39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'evolla': {
        'reference': {
            'config_class': 'transformers:EvollaConfig',
            'model_class': 'transformers:EvollaForProteinText2Text',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'westlake-repl/Evolla-10B-DPO-hf',
                'revision': '2799ba61443b4257e9d36500ed5cf27e72f95ed5',
                'description': 'Pinned model_doc protein-conditioned sampled text generation plus checkpoint generation_config.json.',
            },
            'generation_config': {
                '_from_model_config': True, 'bos_token_id': 1, 'eos_token_id': 2, 'do_sample': True,
                'max_new_tokens': 512, 'temperature': 0.6, 'top_p': 0.9,
                'transformers_version': '4.52.0.dev0', 'use_cache': False, 'pad_token_id': 0,
            },
        },
        'dimension_overrides': {
            'protein_encoder_config': {'hidden_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'intermediate_size': 128},
            'vocab_size': 256,
            'hidden_size': 128,
            'intermediate_size': 256,
            'num_hidden_layers': 4,
            'num_attention_heads': 4,
            'num_key_value_heads': 1,
            'aligner_num_add_layers': 2,
            'resampler_depth': 2,
            'resampler_dim_head': 32,
            'resampler_heads': 2,
            'resampler_num_latents': 4,
            'pad_token_id': 0,
            'bos_token_id': 1,
            'eos_token_id': 2,
        },
        'input': {'kind': 'external'},
        'outputs': ['sequences', 'logits'],
        'workload': 'generate',
        'generation_seed': 3141,
        'generation_kwargs': {'max_new_tokens': 4},
        'reference_backend': None,
        'dimension_purpose': ('Retains ordinary structure-aware protein tokens, token dropout, rotary encoder, two resampler '
            'blocks and two text aligners. Explicit width/depth/vocabulary and output-length reductions; '
            'native sampling temperature0.6/top_p0.9/inherited top_k50/use_cachefalse retained. BOS/EOS/pad'
            ' IDs remapped only to development vocabulary. Recomputes full protein and text paths each '
            'generation step. Four actual trained scalar gates from the pinned checkpoint replace '
            'zero-initialized development gates identically on both sides; all other development weights '
            'retain native initialization. Compares sampled sequences and every exposed pre-processor '
            'per-step logit.'),
    },

    'exaone4': {
        'reference': {
            'config_class': 'transformers:Exaone4Config',
            'model_class': 'transformers:Exaone4ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'LGAI-EXAONE/EXAONE-4.0-32B',
                'revision': 'a1d54d1c148c30881ed27e035b650da489b51b92',
                'url': 'https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/blob/a1d54d1c148c30881ed27e035b650da489b51b92/config.json',
                'description': 'Pinned HF exaone4 public task docs or configuration checkpoint; retain actual computational settings.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 4099},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'exaone4_5': {
        'reference': {
            'config_class': 'transformers:Exaone4_5_Config',
            'model_class': 'transformers:Exaone4_5_ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'LGAI-EXAONE/EXAONE-4.5-33B',
                'revision': '570aa4b15a4f45ba1133072b45f50198f6e3b4fd',
                'description': 'Pinned EXAONE4.5 public conditional generation checkpoint, ordinary image/video prefill and continuation.',
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 640, 'head_dim': 128, 'intermediate_size': 1024, 'num_hidden_layers': 4,
                'num_attention_heads': 5, 'num_key_value_heads': 1, 'vocab_size': 1024,
                'max_position_embeddings': 16384, 'sliding_window': 64,
                'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention'],
            },
            'vision_config': {
                'depth': 2, 'hidden_size': 128, 'intermediate_size': 256, 'num_heads': 4,
                'num_key_value_heads': 1, 'out_hidden_size': 640, 'fullatt_block_indexes': [1],
            },
            'image_token_id': 900,
            'video_token_id': 901,
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'sequence_length': 110, 'image_batch_size': 1,
            'shape': [120, 1176], 'video_shape': [240, 1176], 'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 10, 12]], 'video_grid_thw': [[2, 10, 12]],
            'image_token_positions': list(range(4, 34)), 'video_token_positions': list(range(38, 98)),
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Native text head128 and 5:1GQA, LLLG pattern, all llama3 frequency bands; explicit development'
            ' text window64 with109-token prefill and110th decode exercises restricted attention. '
            'Vision4:1GQA and both window/full blocks retain patch14, temporal2, merge2, window112; '
            'image10x12 and video2x10x12 exercise multiple partialwindows.'),
    },

    'exaone_moe': {
        'reference': {
            'config_class': 'transformers:ExaoneMoeConfig',
            'model_class': 'transformers:ExaoneMoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'LGAI-EXAONE/K-EXAONE-236B-A23B',
                'revision': '61e6d578eb102b578e5704e2916ac841df9eca0a',
                'url': 'https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B/blob/61e6d578eb102b578e5704e2916ac841df9eca0a/config.json',
                'description': 'Pinned HF task documentation official checkpoint; preserve enabled computation and outputs.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 384, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 5, 'num_attention_heads': 8, 'num_key_value_heads': 1, 'head_dim': 64,
            'vocab_size': 1024,
            'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention'],
            'mlp_layer_types': ['dense', 'sparse', 'sparse', 'sparse', 'sparse'],
            'is_moe_layer': [False, True, True, True, True], 'sliding_windows': [128, 128, 128, 0, 128],
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain attention width4/3 hidden and8:1 GQA, dense first then routed, LLLG/local pattern with '
            'global NoPE, actual window128 and270-token input. Native128 experts/top8/shared1 remain.'),
    },

    'falcon': {
        'reference': {
            'config_class': 'transformers:FalconConfig',
            'model_class': 'transformers:FalconForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'tiiuae/falcon-7b-instruct',
                'revision': '8782b5c5d8c9290412416618f36a133653e85285',
                'url': 'https://huggingface.co/tiiuae/falcon-7b-instruct/blob/8782b5c5d8c9290412416618f36a133653e85285/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/falcon.md public causal-LM example; preserve '
                    'checkpoint computational settings and native default inference behavior.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 515},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'falcon_h1': {
        'reference': {
            'config_class': 'transformers:FalconH1Config',
            'model_class': 'transformers:FalconH1ForCausalLM',
            'source': {
                'checkpoint': 'tiiuae/Falcon-H1-7B-Instruct',
                'revision': '41e72f27effbab80cd45b6e884688452253a3686',
                'url': 'https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/blob/41e72f27effbab80cd45b6e884688452253a3686/config.json',
                'kind': 'example_checkpoint',
                'description': ('Pinned task docstring uses a literal placeholder; '
                    'docs/source/en/model_doc/falcon_h1.md:52 supplies the ordinary 7B-Instruct example. '
                    'Preserve every-layer parallel attention/Mamba2, enabled after-gate RMS norm, explicit '
                    'SSM width, all MuP scales, and GQA6.'),
            },
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'falcon_mamba': {
        'reference': {
            'config_class': 'transformers:FalconMambaConfig',
            'model_class': 'transformers:FalconMambaForCausalLM',
            'source': {
                'checkpoint': 'tiiuae/falcon-mamba-7b-instruct',
                'revision': 'b250fc9399d14f56aca18e9ea70bbfb1f73479eb',
                'url': 'https://huggingface.co/tiiuae/falcon-mamba-7b-instruct/blob/b250fc9399d14f56aca18e9ea70bbfb1f73479eb/config.json',
                'kind': 'example_checkpoint',
                'description': ('Pinned HF docs/source/en/model_doc/falcon_mamba.md:62 first ordinary causal-LM '
                    'example. Complete checkpoint configuration with pinned constructor defaults. The '
                    'checkpoint has expand16 but explicit intermediate8192 at hidden4096; pinned superclass'
                    ' restores explicit intermediate_size. Preserve the effective ratio2 and untied '
                    'embeddings.'),
            },
            'cache_argument': 'cache_params',
            'cache_output': 'cache_params',
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 130},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'cache_params'],
    },

    'fast_vlm': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:FastVlmConfig',
            'model_class': 'transformers:FastVlmForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'KamilaMila/FastVLM-0.5B',
                'revision': '2d7bc6dd5c1f5d7bafa99812f45b4db689209e50',
                'description': 'Pinned HF FastVLM conditional-generation forward example uses the public 0.5B checkpoint.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/fast_vlm/modeling_fast_vlm.py',
            },
            'prefill_input_names': ['pixel_values'],
        },
        'dimension_overrides': {
            'vision_config': {
                'hidden_size': 256,
                'num_features': 256,
                'model_args': {'layers': [1, 1, 1, 1, 1], 'embed_dims': [8, 16, 32, 64, 128], 'mlp_ratios': [4, 4, 4, 4, 4]},
            },
            'text_config': {
                'hidden_size': 896, 'intermediate_size': 1792, 'num_hidden_layers': 2,
                'layer_types': ['full_attention', 'full_attention'],
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 128, 128],
            'sequence_length': 24, 'image_token_positions': [3, 4, 5, 6],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': {'': 'sdpa', 'vision_config': 'eager'},
        'dimension_purpose': ('All five FastViT stages, four downsampling transitions, both position convolutions, both '
            'attention stages, learned scales, final squeeze-excitation, native wrapper pooling and GELU '
            'projector preserved; tied Qwen2 retains14:2 GQA heads and native head_dim64/defaultRoPE. Only '
            'depth/width/input size reduced.'),
    },

    'fastspeech2_conformer': {'reference': {'config_class': 'transformers:FastSpeech2ConformerWithHifiGanConfig',
                                         'model_class': 'transformers:FastSpeech2ConformerWithHifiGan',
                                         'source': {'kind': 'example_checkpoint',
                                                    'checkpoint': 'espnet/fastspeech2_conformer_with_hifigan',
                                                    'revision': '7c7b76ccfcda92b7e9708f0c78e0b73f6f2247a3',
                                                    'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer.py#L1550-L1562',
                                                    'description': 'Pinned public text-to-waveform example '
                                                                   'selects the complete '
                                                                   'FastSpeech2ConformerWithHifiGan task, '
                                                                   'including duration/pitch/energy, '
                                                                   'postnet and full HiFi-GAN.'},
                                         'fan_in_normal_modules': ['vocoder']},
                           'dimension_overrides': {'model_config': {'hidden_size': 192,
                                                                    'encoder_num_attention_heads': 1,
                                                                    'decoder_num_attention_heads': 1,
                                                                    'encoder_layers': 2,
                                                                    'decoder_layers': 2,
                                                                    'encoder_linear_units': 768,
                                                                    'decoder_linear_units': 768,
                                                                    'duration_predictor_channels': 128,
                                                                    'pitch_predictor_channels': 128,
                                                                    'energy_predictor_channels': 128,
                                                                    'speech_decoder_postnet_units': 128,
                                                                    'encoder_config': {'layers': 2,
                                                                                       'num_attention_heads': 1,
                                                                                       'linear_units': 768},
                                                                    'decoder_config': {'layers': 2,
                                                                                       'num_attention_heads': 1,
                                                                                       'linear_units': 768}},
                                                   'vocoder_config': {'upsample_initial_channel': 64}},
                           'dimension_purpose': 'Two encoder/decoder blocks, width192/one head preserves '
                                                'native192-wide heads and4:1 feed-forward ratio. Predictor '
                                                'channels128 and postnet units128 reduce ordinary widths '
                                                'while retaining every predictor/postnet layer. HiFi-GAN '
                                                'channels64 retains all4 upsamplers, rates[8,8,2,2], all3 '
                                                'residual kernel types and all3 dilation stages. Native '
                                                'mel80/vocab78 and convolution kernels remain. '
                                                'Explicit32-token synthetic text exercises encoder kernel7 '
                                                'and decoder kernel31; learned durations determine actual '
                                                'output length.',
                           'input': {'kind': 'tokens',
                                     'vocab_size': 78,
                                     'batch_size': 1,
                                     'sequence_length': 32},
                           'workload': 'forward',
                           'default_dtype': 'float32',
                           'reference_backend': None,
                           'outputs': ['spectrogram',
                                       'encoder_last_hidden_state',
                                       'duration_outputs',
                                       'pitch_outputs',
                                       'energy_outputs',
                                       'waveform'],
                           'notes': 'FP32 is explicit: native length_regulator allocates FP32, causing '
                                    'BF16 decoder convolution dtype failure. Direct nearest-even '
                                    'composition uses admitted casts/fixed half scaling/subtractions, two '
                                    'CodecTop1 predicates and gather; repeat expansion uses existing '
                                    'Offset2Batch per token, linear output storage with unoptimized launch '
                                    'count.'},
    'flaubert': {
        'reference': {
            'config_class': 'transformers:FlaubertConfig',
            'model_class': 'transformers:FlaubertWithLMHeadModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'flaubert/flaubert_base_uncased',
                'revision': '5b5cf0a16b62b6bc97d41646aafea90664a2ed48',
                'url': 'https://huggingface.co/flaubert/flaubert_base_uncased/blob/5b5cf0a16b62b6bc97d41646aafea90664a2ed48/config.json',
                'description': ('Pinned HF task example or configuration documentation checkpoint; preserve checkpoint '
                    'computation flags, completed by pinned constructor defaults.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'flava': {
        'reference': {
            'config_class': 'transformers:FlavaConfig',
            'model_class': 'transformers:FlavaModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/flava-full',
                'revision': '51e2b5fac49169959b01da7f3708d8e0d8d7304a',
                'description': ('Pinned FlavaModel.forward example: paired text/image, joint encoder enabled. Public '
                    'base task, no pretraining losses/codebook.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/flava/modeling_flava.py#L1120-L1155',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 2, 'sequence_length': 37,
            'shape': [3, 224, 224], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'image_embeddings', 'text_embeddings', 'multimodal_embeddings', 'image_output.last_hidden_state',
            'image_output.pooler_output', 'text_output.last_hidden_state', 'text_output.pooler_output',
            'multimodal_output.last_hidden_state', 'multimodal_output.pooler_output',
            'image_output.hidden_states', 'text_output.hidden_states',
        ],
        'reference_backend': None,
    },

    'flex_olmo': {
        'reference': {
            'config_class': 'transformers:FlexOlmoConfig',
            'model_class': 'transformers:FlexOlmoForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'allenai/FlexOlmo-7x7B-1T',
                'revision': '0dabfa2e5fd7d8c4bec7e3d3b1a05608255fb096',
                'description': ('Pinned HF configuration checkpoint; model example allenai/FlexOlmo-1B-7B-0924 '
                    'returns404. Retain checkpoint7experts/top7 without forced sparse routing.'),
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 704, 'num_hidden_layers': 2, 'num_attention_heads': 2,
            'num_key_value_heads': 2, 'vocab_size': 1024, 'bos_token_id': 1, 'eos_token_id': 1,
            'pad_token_id': 0,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 271},
        'reference_backend': 'sdpa',
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'dimension_purpose': ('Retain native head128, MHA, all7experts/top7, joint Q/K norms, post-normalized branches and '
            'FP32 RoPE. Special IDs remapped inside reduced vocabulary. The 271-token input provides '
            '269 prompt tokens and two supplied-token continuations; compare all logits and logical '
            'key/value caches at each step.'),
    },

    'florence2': {
        'reference': {
            'config_class': 'transformers:Florence2Config',
            'model_class': 'transformers:Florence2ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'florence-community/Florence-2-large',
                'revision': '4271c66b88cdbc05735372ec13b2360108de5317',
                'description': 'Pinned native Florence2 task checkpoint; complete visual conditioning and native three-beam conditional text generation.',
                'url': 'https://huggingface.co/florence-community/Florence-2-large',
            },
            'generation_config': {
                '_from_model_config': True, 'bos_token_id': 0, 'decoder_start_token_id': 2,
                'early_stopping': True, 'eos_token_id': 2, 'forced_bos_token_id': 0, 'forced_eos_token_id': 2,
                'no_repeat_ngram_size': 3, 'num_beams': 3, 'pad_token_id': 1,
                'transformers_version': '4.56.1',
            },
        },
        'reference_backend': {'': 'sdpa', 'vision_config': 'sdpa', 'text_config': 'sdpa'},
        'dimension_overrides': {
            'vision_config': {
                'depths': [1, 1, 1, 1], 'embed_dim': [32, 64, 128, 256], 'num_heads': [1, 2, 4, 8],
                'num_groups': [1, 2, 4, 8], 'projection_dim': 128,
            },
            'text_config': {
                'd_model': 128, 'encoder_layers': 2, 'decoder_layers': 2, 'encoder_attention_heads': 2,
                'decoder_attention_heads': 2, 'encoder_ffn_dim': 512, 'decoder_ffn_dim': 512,
            },
        },
        'input': {'kind': 'text', 'batch_size': 1, 'sequence_length': 160},
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 8},
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'dimension_purpose': ('All four DaViT stages, spatial and channel attention, depthwise positional convolutions, '
            'native window12, global plus spatial feature sources, full BART encoder/decoder with KV cache,'
            ' and native three-beam search retained. Image416x384 exercises window padding and the '
            'non-square13x12 final grid. Eight generated steps exercise forced BOS/EOS and native trigram '
            'repetition blocking. Original vocabulary and position embeddings retained; width and depth '
            'reductions only.'),
    },

    'fnet': {
        'reference': {
            'config_class': 'transformers:FNetConfig',
            'model_class': 'transformers:FNetForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/fnet-base',
                'revision': 'd89b6fad3cf5384848b783dc480f9685f49d008c',
                'url': 'https://huggingface.co/google/fnet-base/blob/d89b6fad3cf5384848b783dc480f9685f49d008c/config.json',
                'description': ('Pinned FNetForMaskedLM generated forward example loads google/fnet-base. Preserve its '
                    'non-TPU Fourier computation, gelu_new, and the pooler executed by the masked-LM '
                    'backbone even though its output is discarded.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
        'default_dtype': 'float32',
    },

    'focalnet': {
        'reference': {
            'config_class': 'transformers:FocalNetConfig',
            'model_class': 'transformers:FocalNetModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/focalnet/configuration_focalnet.py',
                'description': 'Pinned configuration example initializes FocalNetModel from FocalNetConfig defaults.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['last_hidden_state', 'pooler_output'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'fsmt': {
        'reference': {
            'config_class': 'transformers:FSMTConfig',
            'model_class': 'transformers:FSMTForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/wmt19-ru-en',
                'revision': '683f21ae88f6f8f4f1d27933591ed6309f0bee1f',
                'description': ('Pinned FSMTForConditionalGeneration translation example selects facebook/wmt19-ru-en. '
                    'Default use_cache=True retains its first-step last-token slicing; old artifact forced '
                    'use_cache=False and is not copied.'),
            },
            'decoder_full_history': True,
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 3, 'encoder_prefix_token_ids': [],
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'funnel': {
        'reference': {
            'config_class': 'transformers:FunnelConfig',
            'model_class': 'transformers:FunnelForMaskedLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'funnel-transformer/small',
                'revision': '81e9124473a26bd4ff7244e1c1c544ef8e22ed08',
                'url': 'https://huggingface.co/funnel-transformer/small/blob/81e9124473a26bd4ff7244e1c1c544ef8e22ed08/config.json',
                'description': 'Pinned HF documented checkpoint interpreted with pinned config defaults.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 511},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'fuyu': {
        'reference': {
            'config_class': 'transformers:FuyuConfig',
            'model_class': 'transformers:FuyuForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'adept/fuyu-8b',
                'revision': 'f41defefdb89be0d28cac19d94ce216e37cb6be5',
                'url': 'https://huggingface.co/adept/fuyu-8b/blob/f41defefdb89be0d28cac19d94ce216e37cb6be5/config.json',
                'description': 'Pinned native task example checkpoint; LXMERT config example selects its base task.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 512,
            'intermediate_size': 1024,
            'vocab_size': 1024,
            'image_token_id': 1023,
            'text_config': {'hidden_size': 512, 'intermediate_size': 1024, 'vocab_size': 1024},
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 35,
            'shape': [16, 2700], 'image_input_name': 'image_patches',
            'image_token_positions': list(range(0, 16)),
        },
        'workload': 'forward',
        'outputs': ['logits', 'past_key_values'],
        'dimension_purpose': ('Retain36layers/64heads,30x30RGB patches,half-head rotary,QKnorm,squaredReLU; reduce '
            'widths/vocabulary and use16patch developmentimage. Pinned forward merges by image token mask; '
            'optionalpatch indices unused.'),
        'reference_backend': 'sdpa',
    },

    'gemma': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:GemmaConfig',
            'model_class': 'transformers:GemmaForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'checkpoint': 'google/gemma-7b',
                'revision': 'ff6768d9368919a1f025a54f9f5aa0ee591730bb',
                'url': 'https://huggingface.co/google/gemma-7b/resolve/ff6768d9368919a1f025a54f9f5aa0ee591730bb/config.json',
                'kind': 'constructor_defaults',
                'checkpoint_access_error': {'type': 'GatedRepoError', 'status_code': 403},
                'description': ('The pinned GemmaForCausalLM example and GemmaConfig documentation name '
                    'google/gemma-7b. Its metadata resolves the recorded revision, but config.json returns '
                    'HTTP403 GatedRepoError. Use complete pinned GemmaConfig constructor defaults; the '
                    'recorded checkpoint revision identifies the inaccessible example, not a loaded '
                    'checkpoint configuration.'),
                'source_locations': [
                    'src/transformers/models/gemma/configuration_gemma.py:GemmaConfig',
                    'src/transformers/models/gemma/modeling_gemma.py:GemmaForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'gemma3': {
        'reference': {
            'config_class': 'transformers:Gemma3Config',
            'model_class': 'transformers:Gemma3ForConditionalGeneration',
            'prefill_sequence_input_names': ['token_type_ids'],
            'continuation_outputs': ['logits', 'past_key_values'],
            'source': {
                'kind': 'pinned_recipe',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/gemma3/convert_gemma3_weights.py#L184-L204',
                'description': ('Complete 4B multimodal configuration from the pinned HF conversion recipe, including '
                    'its shared vision configuration at lines 112-125. Explicit fallback for inaccessible '
                    'checkpoint configuration and incompatible no-argument defaults.'),
                'parameters': {
                    'text_config': {
                        'vocab_size': 262144,
                        'hidden_size': 2560,
                        'intermediate_size': 10240,
                        'num_attention_heads': 8,
                        'head_dim': 256,
                        'num_hidden_layers': 34,
                        'num_key_value_heads': 4,
                        'sliding_window': 1024,
                        'rope_parameters': {
                            'full_attention': {'rope_type': 'linear', 'factor': 8.0},
                            'sliding_attention': {'rope_type': 'default'},
                        },
                        'rope_theta': 1000000,
                        'rope_local_base_freq': 10000,
                        'attn_logit_softcapping': None,
                        'query_pre_attn_scalar': 256,
                    },
                    'vision_config': {
                        'hidden_size': 1152, 'intermediate_size': 4304, 'num_hidden_layers': 27,
                        'num_attention_heads': 16, 'num_channels': 3, 'image_size': 896, 'patch_size': 14,
                        'hidden_act': 'gelu_pytorch_tanh', 'layer_norm_eps': 1e-06, 'attention_dropout': 0.0,
                        'vision_use_head': False,
                    },
                },
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'text_image',
            'text_batch_size': 1,
            'image_batch_size': 1,
            'sequence_length': 1027,
            'shape': [3, 896, 896],
            'image_token_positions': list(range(2, 258)),
            'fixed_token_ids': {'0': 2, '1': 255999, '258': 256000},
            'image_token_type_ids': True,
            'attention_mask': True,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
    },

    'gemma3n': {
        'reference': {
            'config_class': 'transformers:Gemma3nConfig',
            'model_class': 'transformers:Gemma3nForConditionalGeneration',
            'continuation_outputs': ['logits', 'past_key_values'],
            'randomize_zero_parameters': [f'model.language_model.layers.{i}.altup.correct_output_scale' for i in range(15)],
            'source': {
                'kind': 'constructor_defaults',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/gemma3n/configuration_gemma3n.py#L401-L453',
                'description': ('The composite config example explicitly constructs default text, vision, and audio configs. '
                    'Its snippet omits vision/audio config imports and incorrectly constructs Gemma3nTextConfig as the model. '
                    'Select the demonstrated three default subconfigs via the actual usable Gemma3nConfig() constructor '
                    'and Gemma3nForConditionalGeneration, rather than substituting an unrelated checkpoint. '
                    'Retain all three modalities; vision is mobilenetv5_300m_enc, not the stale resnet50 prose default.'),
            },
        },
        'reference_backend': None,
        'dimension_overrides': {
            'text_config': {'vocab_size':512,'vocab_size_per_layer_input':256,'hidden_size':1024,
                'intermediate_size':8192,'num_hidden_layers':15,'num_attention_heads':4,'num_key_value_heads':1,
                'head_dim':256,'num_kv_shared_layers':5,'hidden_size_per_layer_input':128,
                'activation_sparsity_pattern':[.95]*5+[0.]*10},
            'audio_config': {'hidden_size':384,'conf_num_attention_heads':2,'conf_num_hidden_layers':2,'vocab_offset':384},
            'vision_config': {'vocab_offset':256,'model_args':{'channel_multiplier':.125}},
            'boi_token_id':254,'eoi_token_id':256,'image_token_id':257,
            'boa_token_id':255,'eoa_token_id':384,'audio_token_id':385,
        },
        'dimension_purpose': ('Text halves width/head count while keeping 256-wide heads, Q:KV4:1, FF8:1, AltUp4, Laurel64, '
            'PLE width/hidden ratio1:8, rotary bases, softcaps, and512 sliding window. Three five-layer attention cycles '
            'retain sparse nonshared, dense nonshared, and dense shared stages; explicitly set sparse5 so scaling does not '
            'erase dense nonshared layers. Audio keeps192-wide heads,128 mel bins, both subsampling convs, chunk12, '
            'left context13, causal kernel5, FF4:1, and4:1 reduction across two repeated complete Conformer blocks. '
            'Vision scales convolution widths only, retaining every MobileNetV5 stage/block, all attention heads/key/value '
            'dimensions, projection2048, and256 image tokens. Image512 retains MSFA nearest alignment and nontrivial '
            'average pooling into16x16 output; audio132frames exercises three attention chunks, partial chunk and padding. '
            'Keep188 audio soft tokens including native learned padding. Prefill516 crosses the unchanged512 window; '
            'two supplied tokens check cache continuation. Vocabulary offsets are remapped consistently, retaining128 '
            'hard tokens per nontext modality. The existing named-zero-parameter initializer makes AltUp scales informative; native vision/audio scales stay unchanged.'),
        'input': {
            'kind':'text_image', 'text_batch_size':1, 'image_batch_size':1,
            'sequence_length':518, 'shape':[3,512,512], 'audio_shape':[132,128],
            'audio_time_axis':0, 'input_features_lengths':[119],
            'dtypes':{'input_features_mask':'bool'},
            'image_token_positions':list(range(3,259)),
            'audio_token_positions':list(range(261,449)),
            'fixed_token_ids':{0:2,1:10,2:254,259:256,260:255,449:384,516:2,517:10},
        },
        'workload':'causal_lm_continuation',
        'outputs':['logits','image_hidden_states','audio_hidden_states','past_key_values'],
    },
    'gemma4': {
        'reference': {
            'config_class': 'transformers:Gemma4Config',
            'model_class': 'transformers:Gemma4ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/gemma-4-e2b-it',
                'revision': '3e22461f65e89153144f8adb70e3b8c2cc9845a7',
                'description': ('First pinned model documentation example uses image-text generation with E2B, SDPA, '
                    'static cache; preserve checkpoint sampling configuration. Optional audio/video inputs '
                    'are absent in this image task.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/gemma4.md',
            },
            'generation_config': {
                'bos_token_id': 2, 'do_sample': True, 'eos_token_id': [1, 106, 50], 'pad_token_id': 0,
                'temperature': 1.0, 'top_k': 64, 'top_p': 0.95, 'transformers_version': '5.5.0.dev0',
            },
        },
        'dimension_overrides': {
            'text_config': {
                'vocab_size': 256, 'hidden_size': 96, 'intermediate_size': 192, 'head_dim': 16,
                'max_position_embeddings': 256, 'sliding_window': 16, 'vocab_size_per_layer_input': 256,
                'hidden_size_per_layer_input': 16, 'global_head_dim': 32,
            },
            'vision_config': {
                'hidden_size': 48, 'intermediate_size': 96, 'head_dim': 4, 'position_embedding_size': 32,
                'global_head_dim': 4,
            },
            'audio_config': {
                'hidden_size': 32, 'num_attention_heads': 4, 'subsampling_conv_channels': [8, 4],
                'output_proj_dims': 96,
            },
            'image_token_id': 250,
            'audio_token_id': 251,
            'video_token_id': 252,
            'boi_token_id': 245,
            'boa_token_id': 246,
            'eoi_token_id': 247,
            'eoa_token_id': 248,
        },
        'input': {
            'kind': 'text_image',
            'text_batch_size': 1,
            'image_batch_size': 1,
            'sequence_length': 20,
            'shape': [36, 768],
            'image_token_positions': [2, 3, 4, 5],
            'image_position_ids': [
                [
                    [0, 0],
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [0, 5],
                    [1, 0],
                    [1, 1],
                    [1, 2],
                    [1, 3],
                    [1, 4],
                    [1, 5],
                    [2, 0],
                    [2, 1],
                    [2, 2],
                    [2, 3],
                    [2, 4],
                    [2, 5],
                    [3, 0],
                    [3, 1],
                    [3, 2],
                    [3, 3],
                    [3, 4],
                    [3, 5],
                    [4, 0],
                    [4, 1],
                    [4, 2],
                    [4, 3],
                    [4, 4],
                    [4, 5],
                    [5, 0],
                    [5, 1],
                    [5, 2],
                    [5, 3],
                    [5, 4],
                    [5, 5],
                ],
            ],
            'attention_mask': True,
        },
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'generation_seed': 781,
        'generation_kwargs': {'max_new_tokens': 3, 'cache_implementation': 'static', 'disable_compile': True},
        'reference_backend': 'sdpa',
        'dimension_purpose': ('All 35 text layers, 20 shared-KV layers, all 16 vision layers, 8/1 grouped text heads, 12 '
            'vision heads, PLE, doubled shared-layer MLP and clipped linears retained. Widths/vocab '
            'reduced; 6x6 patches pool to four image tokens; length20 exceeds reduced sliding window16 and '
            'three generation steps exercise rolling static cache. Audio remains configured and its 752 '
            'state entries are retained but the documented image task does not invoke it. Common '
            'development state uses finite +/-0.5 clipping bounds to exercise clipping. Eager uncompiled '
            'reference and composition are compared explicitly; no native-checkpoint performance claim.'),
    },

    'git': {
        'reference': {
            'config_class': 'transformers:GitConfig',
            'model_class': 'transformers:GitForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/git-base-coco',
                'revision': 'a13141da42abd4a8cbf283601a8104265f537cee',
                'url': 'https://huggingface.co/microsoft/git-base-coco/blob/a13141da42abd4a8cbf283601a8104265f537cee/config.json',
                'description': 'Pinned native task example checkpoint; LXMERT config example selects its base task.',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 19,
            'shape': [3, 224, 224], 'batch_size': 1, 'attention_mask': True,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'glm': {
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'reference': {
            'config_class': 'transformers:GlmConfig',
            'model_class': 'transformers:GlmForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'THUDM/glm-4-9b-chat',
                'revision': 'bd8234fe5e0c09c48637a92abb0c797cb5fa0e73',
                'description': ('Interpreted author configuration: legacy ChatGLM fields mapped explicitly; official '
                    'source confirms theta5000000. Pinned HF converter dict getattr instead produces10000. '
                    'Unmodified pinned GlmConfig/model receives explicit overrides.'),
            },
        },
        'config_overrides': {
            'vocab_size': 151552,
            'intermediate_size': 13696,
            'num_hidden_layers': 40,
            'max_position_embeddings': 131072,
            'rms_norm_eps': 1.5625e-07,
            'head_dim': 128,
            'attention_bias': True,
            'num_attention_heads': 32,
            'hidden_size': 4096,
            'attention_dropout': 0.0,
            'use_cache': True,
            'eos_token_id': [151329, 151336, 151338],
            'pad_token_id': 151329,
            'tie_word_embeddings': False,
            'num_key_value_heads': 2,
            'rope_parameters': {'rope_type': 'default', 'rope_theta': 5000000, 'partial_rotary_factor': 0.5},
        },
        'dimension_overrides': {
            'hidden_size': 1024, 'intermediate_size': 3424, 'num_attention_heads': 16,
            'num_key_value_heads': 1, 'head_dim': 64, 'num_hidden_layers': 2, 'vocab_size': 1024,
            'max_position_embeddings': 1024, 'pad_token_id': 2, 'eos_token_id': [2, 3, 4],
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 257},
        'outputs': ['logits'],
        'dimension_purpose': ('Retains GQA16:1, MLP/H3.34375, QKV bias and half-head interleaved RoPE. Author theta5000000 '
            'unchanged; shared PAD/first EOS identity preserved. Development only.'),
    },

    'glm4': {
        'reference': {
            'config_class': 'transformers:Glm4Config',
            'model_class': 'transformers:Glm4ForCausalLM',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/glm4/configuration_glm4.py',
                'description': ('Pinned public configuration example directly constructs this config. GLM4 autodoc '
                    'checkpoint names multimodal GLM-OCR, not the standalone text task; use explicit '
                    'constructor example.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'glm46v': {
        'reference': {
            'config_class': 'transformers:Glm46VConfig',
            'model_class': 'transformers:Glm46VForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'zai-org/GLM-4.1V-9B-Thinking',
                'revision': '3c1471e51dc811b589d4d12b1c1c7c1c941267c2',
                'description': ('Pinned conditional-generation task; matching documented checkpoint config. GLM46V '
                    'explicitly documents 4.1V components; OCR forward example conflicts with its matching '
                    'config example.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/glm46v/modeling_glm46v.py',
            },
            'prefill_input_names': ['pixel_values', 'image_grid_thw'],
            'native_position_ids': True,
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 2048, 'intermediate_size': 4096, 'num_hidden_layers': 2,
                'num_attention_heads': 16, 'num_key_value_heads': 1,
            },
            'vision_config': {
                'hidden_size': 128, 'num_heads': 1, 'depth': 2, 'intermediate_size': 256,
                'out_hidden_size': 2048, 'image_size': 56,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [24, 1176],
            'flatten_pixel_batch': True, 'image_grid_thw': [[1, 6, 4]],
            'sequence_length': 16, 'image_token_positions': [3, 4, 5, 6, 7, 8], 'mm_token_type_ids': True,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'rope_deltas', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduced depth, text/vision width and learned image grid. Full vocabulary/special IDs; native '
            'text head128 and GQA ratio; vision head64(OCR)/128(dense), four text norms, Conv3d/2d merge2, '
            'all gated MLPs; nonidentity learned-position interpolation from4x4to6x4 for dense vision. OCR '
            'retains explicit projected text width512 vs residual384.'),
    },

    'glm4_moe': {
        'reference': {
            'config_class': 'transformers:Glm4MoeConfig',
            'model_class': 'transformers:Glm4MoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'zai-org/GLM-4.5',
                'revision': 'cbb2c7cfb52fa128a9660cb1a7a78e017899e115',
                'description': 'Pinned HF config documented matchingcheckpoint; forward example meta-glm4_moe is stale.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 512, 'moe_intermediate_size': 128,
            'num_hidden_layers': 4, 'num_attention_heads': 12, 'num_key_value_heads': 1, 'head_dim': 64,
            'n_routed_experts': 16, 'vocab_size': 512, 'max_position_embeddings': 128, 'pad_token_id': 0,
            'eos_token_id': 2,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 18},
        'outputs': ['logits', 'past_key_values'],
        'workload': 'causal_lm_continuation',
        'dimension_purpose': ('Preserve3initialdense+1routedlayer,12:1GQA,QKVbias,QKnorm,halfheadRoPE,sharedexpert '
            'andtop8of16routing; reduceotherdimensions.'
            ' Keep the prompt and first continuation; add a second continuation and compare all logical caches.'),
    },

    'glm4_moe_lite': {
        'reference': {
            'config_class': 'transformers:Glm4MoeLiteConfig',
            'model_class': 'transformers:Glm4MoeLiteForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'zai-org/GLM-4.7-Flash',
                'revision': '7dd20894a642a0aa287e9827cb1a1f7f91386b67',
                'url': 'https://huggingface.co/zai-org/GLM-4.7-Flash/blob/7dd20894a642a0aa287e9827cb1a1f7f91386b67/config.json',
                'description': 'Official matching-author fallback: pinned task example404, configuration doc points to different glm4_moe architecture.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 512, 'intermediate_size': 512, 'moe_intermediate_size': 128,
            'num_hidden_layers': 3, 'num_attention_heads': 5, 'num_key_value_heads': 5, 'q_lora_rank': 192,
            'vocab_size': 1024, 'max_position_embeddings': 2048, 'pad_token_id': 0, 'bos_token_id': 1,
            'eos_token_id': 2,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Keep MLA latent512+RoPE64, native QK256/V256, one dense then sparse '
            'layers,64experts/top4/shared and plain interleavedRoPE. Projectionwidth2.5xhidden retained; '
            'reduce widths/layers/context.'),
    },

    'glm4v': {
        'reference': {
            'config_class': 'transformers:Glm4vConfig',
            'model_class': 'transformers:Glm4vForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'zai-org/GLM-4.1V-9B-Thinking',
                'revision': '3c1471e51dc811b589d4d12b1c1c7c1c941267c2',
                'description': ('Pinned conditional-generation task; matching documented checkpoint config. GLM46V '
                    'explicitly documents 4.1V components; OCR forward example conflicts with its matching '
                    'config example.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/glm4v/modeling_glm4v.py',
            },
            'prefill_input_names': ['pixel_values', 'image_grid_thw'],
            'native_position_ids': True,
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 2048, 'intermediate_size': 4096, 'num_hidden_layers': 2,
                'num_attention_heads': 16, 'num_key_value_heads': 1,
            },
            'vision_config': {
                'hidden_size': 128, 'num_heads': 1, 'depth': 2, 'intermediate_size': 256,
                'out_hidden_size': 2048, 'image_size': 56,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [24, 1176],
            'flatten_pixel_batch': True, 'image_grid_thw': [[1, 6, 4]],
            'sequence_length': 16, 'image_token_positions': [3, 4, 5, 6, 7, 8], 'mm_token_type_ids': True,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'rope_deltas', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduced depth, text/vision width and learned image grid. Full vocabulary/special IDs; native '
            'text head128 and GQA ratio; vision head64(OCR)/128(dense), four text norms, Conv3d/2d merge2, '
            'all gated MLPs; nonidentity learned-position interpolation from4x4to6x4 for dense vision. OCR '
            'retains explicit projected text width512 vs residual384.'),
    },

    'glm4v_moe': {
        'reference': {
            'config_class': 'transformers:Glm4vMoeConfig',
            'model_class': 'transformers:Glm4vMoeForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'zai-org/GLM-4.5V',
                'revision': 'ed47433b37111465ec527affaaddceff371bca04',
                'description': ('Pinned conditional-generation task; matching documented checkpoint config. GLM46V '
                    'explicitly documents 4.1V components; OCR forward example conflicts with its matching '
                    'config example.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py',
            },
            'prefill_input_names': ['pixel_values', 'image_grid_thw'],
            'native_position_ids': True,
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 512, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 12, 'num_key_value_heads': 1, 'moe_intermediate_size': 128,
                'n_routed_experts': 16,
            },
            'vision_config': {
                'hidden_size': 128, 'num_heads': 1, 'depth': 2, 'intermediate_size': 256,
                'out_hidden_size': 512, 'image_size': 56,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [24, 1176],
            'flatten_pixel_batch': True, 'image_grid_thw': [[1, 6, 4]],
            'sequence_length': 16, 'image_token_positions': [3, 4, 5, 6, 7, 8], 'mm_token_type_ids': True,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'rope_deltas', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve explicit text head128,12:1GQA, first dense then routed layer, top8/16 experts, one '
            'expert group and one shared expert, normalized sigmoid routing and FP32 correction bias. Full '
            'vocabulary/specialIDs, default cached text; full GLM gated vision and nonidentity '
            'learned-position resize.'),
    },

    'glm_image': {
        'reference': {
            'config_class': 'transformers:GlmImageConfig',
            'model_class': 'transformers:GlmImageForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'zai-org/GLM-Image',
                'revision': '2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4',
                'subfolder': 'vision_language_encoder',
                'description': ('Pinned image-editing conditional-generation public task; official matching component '
                    'subfolder resolves missing root config. Pinned example also omits required modality '
                    'kwargs; this case includes actual source and target image grids.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/glm_image/modeling_glm_image.py',
            },
            'prefill_input_names': ['pixel_values', 'image_grid_thw'],
            'native_position_ids': True,
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 2048, 'intermediate_size': 4096, 'num_hidden_layers': 2,
                'num_attention_heads': 16, 'num_key_value_heads': 1,
            },
            'vision_config': {'hidden_size': 192, 'intermediate_size': 384, 'depth': 2, 'num_heads': 2, 'image_size': 64},
            'vq_config': {'latent_channels': 192, 'embed_dim': 128},
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [6, 768],
            'flatten_pixel_batch': True, 'image_grid_thw': [[1, 2, 3], [1, 2, 3]], 'sequence_length': 17,
            'image_token_positions': [3, 4, 5, 6, 7, 8],
            'input_ids': [[42, 43, 16384, 167855, 167855, 167855, 167855, 167855, 167855, 16385, 44, 45, 46, 47, 16384, 2, 101]],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'rope_deltas', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Image-to-image prompt with complete source image and final target image marker/grid. Preserve '
            'text head128,16:1GQA, full input and image-output vocabularies, 16384-entry normalized VQ '
            'codebook, unconditional two loss terms, native vision head96, all four text norms and '
            'learned-position resize4x4to2x3. Depth/width/image dimensions only reduced; VQ embed '
            'width2048to128 remains power-of-two.'),
    },

    'glm_moe_dsa': {
        'reference': {
            'config_class': 'transformers:GlmMoeDsaConfig',
            'model_class': 'transformers:GlmMoeDsaForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'zai-org/GLM-5',
                'revision': 'c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2',
                'description': 'Pinned HF configuration documented GLM5 defaultBF16checkpoint; generic forwardexample is stale.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 128, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 4, 'num_attention_heads': 4, 'num_key_value_heads': 4, 'q_lora_rank': 64,
            'kv_lora_rank': 64, 'qk_nope_head_dim': 64, 'qk_rope_head_dim': 64, 'v_head_dim': 128,
            'head_dim': 64, 'index_n_heads': 32, 'index_head_dim': 128, 'index_topk': 4,
            'n_routed_experts': 16, 'vocab_size': 512, 'max_position_embeddings': 128, 'pad_token_id': 0,
            'eos_token_id': 2, 'qk_head_dim': 128,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 17},
        'outputs': ['logits'],
        'workload': 'causal_lm',
        'dimension_purpose': ('Preserve3dense+1routedlayer, fullBF16indexer everylayer andseparateindexkeycache; '
            'index_topk2048->4 over16prompttokens retainsproper sparse selection; reduceotherdimensions. '
            'Retain all32published index heads: reducing to4 introduced numerous exactly-zero ReLU-score '
            'ties and backend-dependent valid top-k sets; original4-head failure remains recorded.'),
    },

    'glm_ocr': {
        'reference': {
            'config_class': 'transformers:GlmOcrConfig',
            'model_class': 'transformers:GlmOcrForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'zai-org/GLM-OCR',
                'revision': '2e85a62840ccac27daa451df36c736c4636b8628',
                'description': ('Pinned conditional-generation task; matching documented checkpoint config. GLM46V '
                    'explicitly documents 4.1V components; OCR forward example conflicts with its matching '
                    'config example.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/glm_ocr/modeling_glm_ocr.py',
            },
            'prefill_input_names': ['pixel_values', 'image_grid_thw'],
            'native_position_ids': True,
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 384, 'intermediate_size': 768, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 2,
            },
            'vision_config': {
                'hidden_size': 128, 'num_heads': 2, 'depth': 2, 'intermediate_size': 256,
                'out_hidden_size': 384, 'image_size': 56,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [24, 1176],
            'flatten_pixel_batch': True, 'image_grid_thw': [[1, 6, 4]],
            'sequence_length': 16, 'image_token_positions': [3, 4, 5, 6, 7, 8], 'mm_token_type_ids': True,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'rope_deltas', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduced depth, text/vision width and learned image grid. Full vocabulary/special IDs; native '
            'text head128 and GQA ratio; vision head64(OCR)/128(dense), four text norms, Conv3d/2d merge2, '
            'all gated MLPs; nonidentity learned-position interpolation from4x4to6x4 for dense vision. OCR '
            'retains explicit projected text width512 vs residual384.'),
    },

    'glmasr': {
        'reference': {
            'config_class': 'transformers:GlmAsrConfig',
            'model_class': 'transformers:GlmAsrForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'zai-org/GLM-ASR-Nano-2512',
                'revision': '61ba4e0b3309b6656edea3e93e419f7bd5c61957',
                'url': 'https://huggingface.co/zai-org/GLM-ASR-Nano-2512/blob/61ba4e0b3309b6656edea3e93e419f7bd5c61957/config.json',
                'description': 'Pinned public speech example generates text from audio; retain full audio encoder and native greedy token generation.',
            },
            'generation_config': {
                '_from_model_config': True, 'bos_token_id': 1, 'eos_token_id': [59246, 59253, 59255],
                'transformers_version': '5.0.0.dev0',
            },
        },
        'dimension_overrides': {
            'audio_config': {
                'hidden_size': 64, 'intermediate_size': 256, 'num_attention_heads': 2,
                'num_key_value_heads': 2, 'head_dim': 32, 'num_hidden_layers': 2,
                'max_position_embeddings': 64,
            },
            'text_config': {
                'hidden_size': 128, 'intermediate_size': 384, 'num_attention_heads': 4,
                'num_key_value_heads': 1, 'head_dim': 32, 'num_hidden_layers': 2,
                'max_position_embeddings': 256,
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 13,
            'shape': [128, 64], 'image_input_name': 'input_features',
            'audio_token_positions': list(range(2, 10)), 'attention_mask': True, 'pad_token_id': 59246,
            'input_features_lengths': [64],
        },
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 3},
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduce widths/depth/frame counts, retain128mel bins,4frameaudio concatenation,GQA4:1,half-head'
            ' encoder rotary positions,full-head text rotary positions and native vocabulary;13prompttokens'
            ' include8audio slots.'),
    },

    'glpn': {
        'reference': {
            'config_class': 'transformers:GLPNConfig',
            'model_class': 'transformers:GLPNForDepthEstimation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'vinvino02/glpn-kitti',
                'revision': 'e9e84852cb27aa6347db0f98bf86e296505ac014',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/glpn/modeling_glpn.py',
                'description': 'Pinned public depth-estimation task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 480, 640]},
        'outputs': ['predicted_depth'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'got_ocr2': {
        'reference': {
            'config_class': 'transformers:GotOcr2Config',
            'model_class': 'transformers:GotOcr2ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'stepfun-ai/GOT-OCR-2.0-hf',
                'revision': 'd3017ef2c2c1395888c8d635c5e0508bcb0ac78d',
                'description': 'Official pinned HF conditional-generation example and published default configuration.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/got_ocr2/modeling_got_ocr2.py',
            },
            'prefill_input_names': ['pixel_values'],
            'continuation_outputs': ['logits', 'past_key_values'],
        },
        'input': {
            'kind': 'external', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 1024, 1024],
            'sequence_length': 288, 'image_token_positions': list(range(21, 277)), 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values', 'image_hidden_states'],
        'reference_backend': None,
    },

    'gpt2': {
        'reference': {
            'config_class': 'transformers:GPT2Config',
            'model_class': 'transformers:GPT2LMHeadModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openai-community/gpt2',
                'revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
                'url': 'https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json',
                'description': ('Pinned GPT2LMHeadModel runtime-generated forward example loads openai-community/gpt2. '
                    'Preserve gelu_new, learned absolute positions and default caching.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'gpt_bigcode': {
        'reference': {
            'config_class': 'transformers:GPTBigCodeConfig',
            'model_class': 'transformers:GPTBigCodeForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'bigcode/gpt_bigcode-santacoder',
                'revision': '291931872cae83498cf984b16319f47f5e9e7a07',
                'url': 'https://huggingface.co/bigcode/gpt_bigcode-santacoder/blob/291931872cae83498cf984b16319f47f5e9e7a07/config.json',
                'description': ('Pinned docs/source/en/model_doc/gpt_bigcode.md:70-73 explicitly loads this '
                    'causal-language-model checkpoint. Preserve its multi-query attention, '
                    'gelu_pytorch_tanh and default caching. The example uses FlashAttention2; this '
                    'evaluation selects supported SDPA and reports that backend separately.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'gpt_neo': {
        'reference': {
            'config_class': 'transformers:GPTNeoConfig',
            'model_class': 'transformers:GPTNeoForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'EleutherAI/gpt-neo-1.3B',
                'revision': 'dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8',
                'url': 'https://huggingface.co/EleutherAI/gpt-neo-1.3B/blob/dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/gpt_neo.md public causal-LM example; preserve '
                    'checkpoint computational settings and native default inference behavior.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'gpt_neox': {
        'reference': {
            'config_class': 'transformers:GPTNeoXConfig',
            'model_class': 'transformers:GPTNeoXForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'EleutherAI/gpt-neox-20b',
                'revision': 'c292233c833e336628618a88a648727eb3dff0a7',
                'url': 'https://huggingface.co/EleutherAI/gpt-neox-20b/blob/c292233c833e336628618a88a648727eb3dff0a7/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/gpt_neox.md selects this causal-LM checkpoint; '
                    'preserve its activation, residual order, partial rotary, biases and weight tying.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'gpt_neox_japanese': {
        'reference': {
            'config_class': 'transformers:GPTNeoXJapaneseConfig',
            'model_class': 'transformers:GPTNeoXJapaneseForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'abeja/gpt-neox-japanese-2.7b',
                'revision': 'c4958c0a96d523dd8841fc38da0741056ce470ec',
                'url': 'https://huggingface.co/abeja/gpt-neox-japanese-2.7b/blob/c4958c0a96d523dd8841fc38da0741056ce470ec/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/gpt_neox_japanese.md public causal-LM example; '
                    'preserve checkpoint computational settings and native default inference behavior.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 195},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'gpt_oss': {
        'reference': {
            'config_class': 'transformers:GptOssConfig',
            'model_class': 'transformers:GptOssForCausalLM',
            'load_device': 'cuda:0',
            'serialized_weight_suffixes': ['_blocks', '_scales'],
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openai/gpt-oss-20b',
                'revision': '6cee5e81ee83917806bbde320786a8fb61efebee',
                'url': 'https://huggingface.co/openai/gpt-oss-20b/resolve/6cee5e81ee83917806bbde320786a8fb61efebee/config.json',
                'description': ('Pinned documented20B checkpoint with native MXFP4expert quantization. Requires '
                    'supplied common packed state; no unquantized fallback.'),
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 256, 'num_hidden_layers': 2, 'num_attention_heads': 8,
            'num_key_value_heads': 1, 'head_dim': 64, 'vocab_size': 1024, 'pad_token_id': 1023,
            'eos_token_id': 1022, 'layer_types': ['sliding_attention', 'full_attention'],
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'eager',
        'dimension_purpose': ('Retain32experts/top4,8:1GQA, biased projections, sinks, YaRN, sliding/full pair, native '
            'window128;270tokens crosses sliding cutoff. Common weights are native-HF-initialized then '
            'HF-MXFP4-quantized, explicitly synthetic development weights.'),
    },

    'gptj': {
        'reference': {
            'config_class': 'transformers:GPTJConfig',
            'model_class': 'transformers:GPTJForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'EleutherAI/gpt-j-6B',
                'revision': '47e169305d2e8376be1d31e765533382721b2cc1',
                'url': 'https://huggingface.co/EleutherAI/gpt-j-6B/blob/47e169305d2e8376be1d31e765533382721b2cc1/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/gptj.md public causal-LM example; preserve '
                    'checkpoint computational settings and native default inference behavior.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'granite': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:GraniteConfig',
            'model_class': 'transformers:GraniteForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ibm-granite/granite-3.0-8b-base',
                'revision': '4fad7f8ad56393dcef4e34e37a35962bd091f320',
                'url': 'https://huggingface.co/ibm-granite/granite-3.0-8b-base/resolve/4fad7f8ad56393dcef4e34e37a35962bd091f320/config.json',
                'description': ('The pinned causal-LM example identifier returned HTTP404; use the accessible '
                    'checkpoint named by the pinned configuration-class documentation.'),
                'causal_lm_example_evidence': {
                    'checkpoint': 'meta-granite/Granite-2-7b-hf', 'accessible': False,
                    'error_type': 'RepositoryNotFoundError', 'status_code': 404,
                },
                'source_locations': [
                    'src/transformers/models/granite/configuration_granite.py:GraniteConfig',
                    'src/transformers/models/granite/modeling_granite.py:GraniteForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'granite_speech': {
        'reference': {
            'config_class': 'transformers:GraniteSpeechConfig',
            'model_class': 'transformers:GraniteSpeechForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ibm-granite/granite-speech-3.3-2b',
                'revision': '4ac2f02f413c6169ae8c0ccc217115a366e552d7',
                'url': 'https://huggingface.co/ibm-granite/granite-speech-3.3-2b/blob/4ac2f02f413c6169ae8c0ccc217115a366e552d7/config.json',
                'description': ('Pinned main docs checkpoint unavailable(API404/direct401); parent-authorized fallback '
                    'to exact checkpoint named in pinned configuration_granite_speech.py23,71. Preserve '
                    'speech transcription and enabled trainedq/vLoRA.'),
            },
            'generation_config': {
                '_from_model_config': True, 'bos_token_id': 0, 'eos_token_id': 0, 'pad_token_id': 0,
                'transformers_version': '4.52.4', 'use_cache': True, 'suppress_tokens': [49159],
            },
            'adapter_config': {
                'alpha_pattern': {},
                'auto_mapping': None,
                'base_model_name_or_path': 'ibm-granite/granite-speech-3.3-2b',
                'bias': 'none',
                'fan_in_fan_out': False,
                'inference_mode': True,
                'init_lora_weights': True,
                'layer_replication': None,
                'layers_pattern': None,
                'layers_to_transform': None,
                'loftq_config': {},
                'lora_alpha': 32,
                'lora_dropout': 0.0,
                'megatron_config': None,
                'megatron_core': 'megatron.core',
                'modules_to_save': None,
                'peft_type': 'LORA',
                'r': 64,
                'rank_pattern': {},
                'revision': None,
                'target_modules': ['v_proj', 'q_proj'],
                'task_type': 'CAUSAL_LM',
                'use_dora': False,
                'use_rslora': False,
            },
        },
        'dimension_overrides': {
            'encoder_config': {
                'hidden_dim': 64, 'dim_head': 16, 'num_heads': 4, 'num_layers': 4, 'context_size': 8,
                'max_pos_emb': 16,
            },
            'projector_config': {'encoder_hidden_size': 64, 'hidden_size': 64, 'intermediate_size': 256, 'num_attention_heads': 4},
            'text_config': {
                'hidden_size': 128, 'intermediate_size': 512, 'num_attention_heads': 4,
                'num_key_value_heads': 1, 'num_hidden_layers': 2, 'max_position_embeddings': 256,
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 11,
            'shape': [19, 160], 'image_input_name': 'input_features',
            'audio_token_positions': [2, 3, 4, 5, 6, 7], 'attention_mask': True, 'pad_token_id': 0,
        },
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 3},
        'reference_backend': 'eager',
        'dimension_purpose': ('19frames cross8frame attention blocks and15frame QFormer windows; native rank64alpha32adapter '
            'retained with trainedfirst2layers and leadinginput/outputchannel slices, random base shared '
            'weights alreadyroundedBF16. Fulltrainedcheckpointnotdownloaded. Pass '
            'explicitgranite-common-state.pt.'),
    },

    'granite_speech_plus': {
        'reference': {
            'config_class': 'transformers:GraniteSpeechPlusConfig',
            'model_class': 'transformers:GraniteSpeechPlusForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ibm-granite/granite-speech-4.1-2b-plus',
                'revision': '1454e6e1e33845ca9280ff65f52cf1141ba6e6e2',
                'url': 'https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/blob/1454e6e1e33845ca9280ff65f52cf1141ba6e6e2/config.json',
                'description': ('Pinned main speech transcription example generates with full Conformer, intermediate '
                    'CTC feedback, hidden-layer concatenation, windowed QFormer and scaled Granite; public '
                    'checkpoint disables LoRA.'),
            },
            'generation_config': {
                '_from_model_config': True, 'bos_token_id': 100257, 'eos_token_id': 100257,
                'output_attentions': False, 'output_hidden_states': False, 'pad_token_id': 100256,
                'transformers_version': '5.6.0.dev0', 'use_cache': True,
            },
        },
        'dimension_overrides': {
            'encoder_config': {
                'hidden_dim': 64, 'dim_head': 16, 'num_heads': 4, 'num_layers': 4, 'context_size': 8,
                'max_pos_emb': 16,
            },
            'projector_config': {
                'encoder_hidden_size': 128, 'hidden_size': 64, 'intermediate_size': 256,
                'num_attention_heads': 4,
            },
            'text_config': {
                'hidden_size': 128, 'intermediate_size': 256, 'num_attention_heads': 4,
                'num_key_value_heads': 1, 'num_hidden_layers': 2, 'max_position_embeddings': 256,
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 11,
            'shape': [19, 160], 'image_input_name': 'input_features',
            'audio_token_positions': [2, 3, 4, 5, 6, 7], 'attention_mask': True, 'pad_token_id': 100256,
        },
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 3},
        'reference_backend': 'eager',
        'dimension_purpose': ('19feature frames cross8frame attention blocks and15frame projector windows with padding, '
            'yielding6audio slots; preserve input160,CTCvocab348,layer3concat,4Conformers,2QFormer '
            'layers,all Granite scales,native text vocabulary.'),
    },

    'granitemoe': {
        'reference': {
            'config_class': 'transformers:GraniteMoeConfig',
            'model_class': 'transformers:GraniteMoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ibm/PowerMoE-3b',
                'revision': '13fcb5a98001438bed01cf1ac4b423751dc4c2ea',
                'url': 'https://huggingface.co/ibm/PowerMoE-3b/blob/13fcb5a98001438bed01cf1ac4b423751dc4c2ea/config.json',
                'description': ('Pinned HF model documentation causal-LM example checkpoint; retain all Granite scalar '
                    'multipliers, tied head, native routed expert counts and selected expert counts.'),
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'granitemoehybrid': {
        'reference': {
            'config_class': 'transformers:GraniteMoeHybridConfig',
            'model_class': 'transformers:GraniteMoeHybridForCausalLM',
            'source': {
                'checkpoint': 'ibm-granite/granite-4.0-h-tiny',
                'revision': '791e0d3d28c86e106c9b6e0b4cecdee0375b6124',
                'url': 'https://huggingface.co/ibm-granite/granite-4.0-h-tiny/blob/791e0d3d28c86e106c9b6e0b4cecdee0375b6124/config.json',
                'kind': 'example_checkpoint',
                'description': ('Pinned task example granite-4.0-h-tiny. Preserve top6/64 routed plus ungated shared '
                    'experts every layer, NoPE, tied head, fixed Granite scales and source prefix through '
                    'Mamba after attention.'),
            },
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 270},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'granitemoeshared': {
        'reference': {
            'config_class': 'transformers:GraniteMoeSharedConfig',
            'model_class': 'transformers:GraniteMoeSharedForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ibm-research/moe-7b-1b-active-shared-experts',
                'revision': '6194d7218c3e792b939b535f5640f9c32be9b442',
                'url': 'https://huggingface.co/ibm-research/moe-7b-1b-active-shared-experts/blob/6194d7218c3e792b939b535f5640f9c32be9b442/config.json',
                'description': ('Pinned HF model documentation causal-LM example checkpoint; retain all Granite scalar '
                    'multipliers, tied head, native routed expert counts and selected expert counts.'),
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'grounding_dino': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:GroundingDinoConfig',
            'model_class': 'transformers:GroundingDinoForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'IDEA-Research/grounding-dino-tiny',
                'revision': 'a2bb814dd30d776dcf7e30523b00659f4f141c71',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/grounding_dino/modeling_grounding_dino.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 9,
            'shape': [3, 800, 1066], 'input_ids': [[101, 1037, 4937, 1012, 1037, 6556, 2491, 1012, 102]],
            'token_type_id': 0, 'attention_mask': True, 'batch_size': 1,
        },
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'intermediate_hidden_states',
            'intermediate_reference_points', 'init_reference_points', 'encoder_last_hidden_state_vision',
            'encoder_last_hidden_state_text', 'enc_outputs_class', 'enc_outputs_coord_logits',
            'encoder_logits', 'encoder_pred_boxes', 'input_ids',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'groupvit': {
        'reference': {
            'config_class': 'transformers:GroupViTConfig',
            'model_class': 'transformers:GroupViTModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'nvidia/groupvit-gcc-yfcc',
                'revision': '751f6d9a37c9e9e42ba527cc99d3c21b137bd2a3',
                'hf_source_revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://huggingface.co/nvidia/groupvit-gcc-yfcc/blob/751f6d9a37c9e9e42ba527cc99d3c21b137bd2a3/config.json',
                'description': ('Pinned GroupViTModel example full paired default task; both towers, both hard grouping'
                    ' stages, projection MLPs and every default tensor output retained. Segmentation '
                    'remains disabled as in checkpoint.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 3, 'image_batch_size': 2, 'sequence_length': 32,
            'shape': [3, 224, 224], 'eos_positions': [9, 20, 31], 'bos_token_id': 49406,
            'eos_token_id': 49407, 'pad_token_id': 49407, 'attention_mask': True, 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'helium': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:HeliumConfig',
            'model_class': 'transformers:HeliumForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'kyutai/helium-1-preview',
                'revision': '645c94758493d06ba1b4706c5674d6510e3a84cb',
                'url': 'https://huggingface.co/kyutai/helium-1-preview/resolve/645c94758493d06ba1b4706c5674d6510e3a84cb/config.json',
                'description': ('The pinned causal-LM example identifier returned HTTP404; use the accessible '
                    'checkpoint named by the pinned configuration-class documentation.'),
                'causal_lm_example_evidence': {
                    'checkpoint': 'google/helium-7b', 'accessible': False,
                    'error_type': 'RepositoryNotFoundError', 'status_code': 404,
                },
                'source_locations': [
                    'src/transformers/models/helium/configuration_helium.py:HeliumConfig',
                    'src/transformers/models/helium/modeling_helium.py:HeliumForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'hgnet_v2': {
        'reference': {
            'config_class': 'transformers:HGNetV2Config',
            'model_class': 'transformers:HGNetV2Backbone',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py',
                'description': 'Pinned public backbone task example constructs the architecture using constructor defaults.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['feature_maps'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'hiera': {
        'reference': {
            'config_class': 'transformers:HieraConfig',
            'model_class': 'transformers:HieraModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/hiera/configuration_hiera.py',
                'description': 'Pinned configuration example initializes HieraModel from constructor defaults.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['last_hidden_state', 'pooler_output'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'higgs_audio_v2': {
        'reference': {
            'config_class': 'transformers:HiggsAudioV2Config',
            'model_class': 'transformers:HiggsAudioV2ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'eustlb/higgs-audio-v2-generation-3B-base',
                'revision': '9f2f4918a42f20c4684d5882b96cf610f8075fb3',
                'description': ('Pinned native ForConditionalGeneration example invokes mixed text/audio forward; audio'
                    ' waveform decoding is external to this model.'),
                'url': 'https://huggingface.co/eustlb/higgs-audio-v2-generation-3B-base',
            },
            'forward_kwargs': {'use_cache': True},
        },
        'reference_backend': 'sdpa',
        'dimension_overrides': {
            'hidden_size': 384, 'intermediate_size': 512, 'num_hidden_layers': 2, 'num_attention_heads': 3,
            'num_key_value_heads': 1,
        },
        'input': {'kind': 'text', 'batch_size': 1, 'sequence_length': 8},
        'workload': 'forward',
        'outputs': ['logits', 'past_key_values'],
        'dimension_purpose': ('Two full dual-FFN layers, text/audio normalization branches, original8 audio codebooks1026 '
            'entries,3:1 GQA/head128, nativeLlama3 RoPE scaling, originalvocab/specialIDs. Three audio '
            'frames and ordinary text jointly exercise mixed input path; all returned KV states included.'),
    },

    'higgs_audio_v2_tokenizer': {
        'reference': {
            'config_class': 'transformers:HiggsAudioV2TokenizerConfig',
            'model_class': 'transformers:HiggsAudioV2TokenizerModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'eustlb/higgs-audio-v2-tokenizer',
                'revision': '528e871c2a26c4f0f7773b9754e2e1acae20899d',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/higgs_audio_v2_tokenizer.md',
                'description': ('Pinned model documentation provides this accessible checkpoint; implementation '
                    'docstring checkpoint missing. Preserve24-to16kHz semantic resampling, all semantic '
                    'state averaging, stride2 semantic sampling, DAC and projected codebooks.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [1, 9600]},
        'workload': 'forward',
        'outputs': ['audio_codes', 'audio_values'],
        'reference_backend': None,
    },

    'hubert': {
        'reference': {
            'config_class': 'transformers:HubertConfig',
            'model_class': 'transformers:HubertModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/hubert-large-ls960-ft',
                'revision': 'ece5fabbf034c1073acae96d5401b25be96709d8',
                'url': 'https://huggingface.co/facebook/hubert-large-ls960-ft/blob/ece5fabbf034c1073acae96d5401b25be96709d8/config.json',
                'description': ('Pinned HubertModel.forward example loads this checkpoint. Its biased layer-normalized '
                    'feature convolutions and pre-normalized encoder differ from bare HubertConfig '
                    'defaults.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [48000]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'hunyuan_v1_dense': {
        'reference': {
            'config_class': 'transformers:HunYuanDenseV1Config',
            'model_class': 'transformers:HunYuanDenseV1ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'tencent/Hunyuan-0.5B-Pretrain',
                'revision': '79c2b0c66919d28ade3e4d5934fbfddde3bf1c3b',
                'url': 'https://huggingface.co/tencent/Hunyuan-0.5B-Pretrain/blob/79c2b0c66919d28ade3e4d5934fbfddde3bf1c3b/config.json',
                'description': 'Pinned HF documentation example checkpoint.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'hunyuan_v1_moe': {
        'reference': {
            'config_class': 'transformers:HunYuanMoEV1Config',
            'model_class': 'transformers:HunYuanMoEV1ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'tencent/Hunyuan-A13B-Instruct',
                'revision': '290ddb9a56ed23c2c83a1c8081533e58925df952',
                'url': 'https://huggingface.co/tencent/Hunyuan-A13B-Instruct/blob/290ddb9a56ed23c2c83a1c8081533e58925df952/config.json',
                'description': 'Pinned HF model documentation checkpoint; preserve enabled inference computations.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 128, 'num_hidden_layers': 3, 'num_attention_heads': 4,
            'num_key_value_heads': 1, 'head_dim': 64, 'vocab_size': 1024, 'moe_topk': [8, 8, 8],
            'moe_intermediate_size': [128, 128, 128], 'num_shared_experts': [1, 1, 1], 'pad_token_id': 1020,
            'eos_token_id': 1021, 'eod_token_id': 1022, 'sep_token_id': 1023,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain4:1 GQA, attention width equal hidden width, all64 experts/top8, shared expert and '
            'post-RoPE Q/K norms. Reduce head/model width/depth/vocabulary; retain fixed-alpha dynamic '
            'RoPE. Remap special-token IDs into reduced vocabulary, retaining distinct padding/EOS/EOD/SEP '
            'identities.'
            ' Keep the prompt and first continuation; add a second continuation and compare all logical caches.'),
    },

    'hy_v3': {
        'reference': {
            'config_class': 'transformers:HYV3Config',
            'model_class': 'transformers:HYV3ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'tencent/Hy3-preview',
                'revision': '549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a',
                'url': 'https://huggingface.co/tencent/Hy3-preview/blob/549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a/config.json',
                'description': 'Pinned HF task documentation official checkpoint; preserve enabled computation and outputs.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 3, 'num_attention_heads': 8, 'num_key_value_heads': 1, 'head_dim': 64,
            'vocab_size': 1024, 'pad_token_id': 1019, 'bos_token_id': 1020, 'eos_token_id': 1021,
            'eod_token_id': 1022, 'sep_token_id': 1023,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain attention width2x hidden/8:1 GQA, dense first then routed layers, original192 '
            'experts/top8/shared1, FP32 correction bias and enabled router outputs. Reduce dimensions; '
            'remap distinct special-token IDs within reduced vocabulary.'),
        'outputs': ['logits', 'router_logits'],
    },

    'ibert': {
        'reference': {
            'config_class': 'transformers:IBertConfig',
            'model_class': 'transformers:IBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned IBertConfig names kssteven/ibert-roberta-base. That checkpoint explicitly sets '
                    'quant_mode=false; retain its floating-point wrapper path, pad-aware positions, and '
                    'native input-dtype IntLayerNorm calculation.'),
                'checkpoint': 'kssteven/ibert-roberta-base',
                'revision': '4f98e9110b04a8958444d3af8ed39287834fbb90',
                'url': 'https://huggingface.co/kssteven/ibert-roberta-base/blob/4f98e9110b04a8958444d3af8ed39287834fbb90/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'idefics': {
        'reference': {
            'config_class': 'transformers:IdeficsConfig',
            'model_class': 'transformers:IdeficsForVisionText2Text',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'HuggingFaceM4/idefics-9b',
                'revision': '4986969d084dd36ed0acf20990c64faa7e817df1',
                'description': 'Pinned HF Idefics vision-text generation example checkpoint.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/idefics/modeling_idefics.py',
            },
            'prefill_input_names': ['pixel_values'],
            'prefill_sequence_input_names': ['image_attention_mask'],
            'decode_sequence_input_names': ['image_attention_mask'],
            'decode_from_prefill_outputs': {'perceiver_embeddings': 'image_hidden_states'},
            'native_cache_defaults': True,
            'native_position_ids': True,
            'continuation_outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        },
        'input': {
            'kind': 'text_image',
            'text_batch_size': 1,
            'image_batch_size': 1,
            'shape': [2, 3, 224, 224],
            'sequence_length': 514,
            'image_attention_mask': [
                [
                    [False, False],
                    [False, False],
                    [False, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [True, False],
                    [True, True],
                ],
            ],
            'fixed_token_ids': {'3': 32000, '256': 32001},
            'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'idefics2': {
        'reference': {
            'config_class': 'transformers:Idefics2Config',
            'model_class': 'transformers:Idefics2ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'HuggingFaceM4/idefics2-8b-base',
                'revision': 'e37a0b376ca55a497c68de86505c60a0f8d7d713',
                'description': ('Pinned HF conditional-generation example checkpoint; complete image/text forward with '
                    'cached decode.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/idefics2/modeling_idefics2.py',
            },
            'prefill_input_names': ['pixel_values'],
            'native_cache_defaults': True,
            'native_position_ids': True,
            'continuation_outputs': ['logits', 'past_key_values'],
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [1, 3, 980, 980],
            'sequence_length': 4099, 'image_token_positions': list(range(3, 67)), 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'idefics3': {
        'reference': {
            'config_class': 'transformers:Idefics3Config',
            'model_class': 'transformers:Idefics3ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'HuggingFaceM4/Idefics3-8B-Llama3',
                'revision': 'fddb4ff79181e55a994674777e06cd5456ce3dc3',
                'description': ('Pinned HF conditional-generation example checkpoint; complete image/text forward with '
                    'cached decode.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/idefics3/modeling_idefics3.py',
            },
            'prefill_input_names': ['pixel_values'],
            'native_cache_defaults': True,
            'native_position_ids': True,
            'continuation_outputs': ['logits', 'past_key_values'],
        },
        'input': {
            'kind': 'text_image',
            'text_batch_size': 1,
            'image_batch_size': 1,
            'shape': [2, 3, 364, 364],
            'sequence_length': 363,
            'image_token_positions': [
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
                74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
                135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
                154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 180,
                181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
                219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
                238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
                257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
                276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294,
                295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
                314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
                333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
            ],
            'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'ijepa': {
        'reference': {
            'config_class': 'transformers.models.ijepa.configuration_ijepa:IJepaConfig',
            'model_class': 'transformers.models.ijepa.modeling_ijepa:IJepaModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs IJepaModel(IJepaConfig()) with '
                    'random weights.'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/ijepa/configuration_ijepa.py#L31-L44',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'imagegpt': {
        'reference': {
            'config_class': 'transformers:ImageGPTConfig',
            'model_class': 'transformers:ImageGPTModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'openai/imagegpt-small',
                'revision': '10c8a8402cf80c0eaaf31fb5bc86012b34169481',
                'url': 'https://huggingface.co/openai/imagegpt-small/blob/10c8a8402cf80c0eaaf31fb5bc86012b34169481/config.json',
                'description': 'Pinned HF ImageGPTModel example checkpoint with processor-quantized image input IDs.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 1024},
        'workload': 'causal_lm_continuation',
        'outputs': ['last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'instructblip': {
        'reference': {
            'config_class': 'transformers:InstructBlipConfig',
            'model_class': 'transformers:InstructBlipForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Salesforce/instructblip-vicuna-7b',
                'revision': '19103d0c5b5263c8a7891012e08573439fb6607f',
                'description': ('The pinned conditional-generation example selects Vicuna 7B. Preserve instruction '
                    'tokens in query-transformer self-attention, visual cross-attention on learned queries,'
                    ' and distinct query/text feed-forward layers. The video example uses four frames.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/instructblip/modeling_instructblip.py',
            },
        },
        'dimension_overrides': {
            'vision_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'image_size': 56,
            },
            'qformer_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 4,
                'num_attention_heads': 4, 'encoder_hidden_size': 256,
            },
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 4, 'head_dim': 64,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 56, 56],
            'sequence_length': 74, 'qformer_sequence_length': 17, 'image_token_positions': list(range(3, 35)),
        },
        'workload': 'forward',
        'outputs': [
            'logits', 'language_model_outputs.logits', 'vision_outputs.last_hidden_state',
            'vision_outputs.pooler_output', 'qformer_outputs.last_hidden_state',
            'qformer_outputs.pooler_output', 'language_model_outputs.past_key_values',
        ],
        'reference_backend': {'': 'sdpa', 'vision_config': 'sdpa', 'qformer_config': 'eager', 'text_config': 'sdpa'},
        'dimension_purpose': ('Retain 32 learned queries and 17 instruction tokens. Four query-transformer layers preserve '
            'alternating visual cross-attention and both query/text feed-forward branches. Keep the native '
            'patch width, vocabularies, and modality tokens. The video case uses four frames. Compare all '
            'default visual, query-transformer, language, and cache outputs from a complete forward call.'),
    },

    'instructblipvideo': {
        'reference': {
            'config_class': 'transformers:InstructBlipVideoConfig',
            'model_class': 'transformers:InstructBlipVideoForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Salesforce/instructblip-vicuna-7b',
                'revision': '19103d0c5b5263c8a7891012e08573439fb6607f',
                'description': ('The pinned conditional-generation example selects Vicuna 7B. Preserve instruction '
                    'tokens in query-transformer self-attention, visual cross-attention on learned queries,'
                    ' and distinct query/text feed-forward layers. The video example uses four frames.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py',
            },
        },
        'dimension_overrides': {
            'vision_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'image_size': 56,
            },
            'qformer_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 4,
                'num_attention_heads': 4, 'encoder_hidden_size': 256,
            },
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 4, 'head_dim': 64,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [4, 3, 56, 56],
            'sequence_length': 174, 'qformer_sequence_length': 17,
            'video_token_positions': list(range(3, 131)),
        },
        'workload': 'forward',
        'outputs': [
            'logits', 'language_model_outputs.logits', 'vision_outputs.last_hidden_state',
            'vision_outputs.pooler_output', 'qformer_outputs.last_hidden_state',
            'qformer_outputs.pooler_output', 'language_model_outputs.past_key_values',
        ],
        'reference_backend': {'': 'sdpa', 'vision_config': 'sdpa', 'qformer_config': 'eager', 'text_config': 'sdpa'},
        'dimension_purpose': ('Retain 32 learned queries and 17 instruction tokens. Four query-transformer layers preserve '
            'alternating visual cross-attention and both query/text feed-forward branches. Keep the native '
            'patch width, vocabularies, and modality tokens. The video case uses four frames. Compare all '
            'default visual, query-transformer, language, and cache outputs from a complete forward call.'),
    },

    'internvl': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:InternVLConfig',
            'model_class': 'transformers:InternVLForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'OpenGVLab/InternVL3-1B-hf',
                'revision': '014c0583a0d4bedf29fbe2dbff4f865eb998e171',
                'description': 'Official pinned HF conditional-generation example and published default configuration.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/internvl/modeling_internvl.py',
            },
            'prefill_input_names': ['pixel_values'],
        },
        'dimension_overrides': {
            'vision_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'image_size': [56, 56],
            },
            'text_config': {
                'hidden_size': 448, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 7, 'num_key_value_heads': 1,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 56, 56],
            'sequence_length': 40, 'image_token_positions': [3, 4, 5, 6],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values', 'image_hidden_states'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain 7:1 grouped-query attention, learned visual layer scales, and spatial downsampling by '
            'two. Dynamic rotary frequency scaling remains enabled; its 32,768-token boundary is checked '
            'separately because the primary workload is short.'),
    },

    'jais2': {
        'reference': {
            'config_class': 'transformers:Jais2Config',
            'model_class': 'transformers:Jais2ForCausalLM',
            'native_cache_defaults': True,
            'native_position_ids': True,
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/jais2/configuration_jais2.py#L35-L47',
                'description': ('Complete documented Jais2Config constructor fallback; prior access to the '
                    'forward-example inceptionai/Jais-2-8B-Chat config returned gated 403. Access was not '
                    'retried during this no-download review. This workload does not claim '
                    'checkpoint-configuration equivalence.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'jamba': {
        'reference': {
            'config_class': 'transformers:JambaConfig',
            'model_class': 'transformers:JambaForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'ai21labs/Jamba-v0.1',
                'revision': '9efd11575ba791d9e3d25d4c8b670e78506b2df7',
                'url': 'https://huggingface.co/ai21labs/Jamba-v0.1/blob/9efd11575ba791d9e3d25d4c8b670e78506b2df7/config.json',
                'description': 'Pinned HF public task example checkpoint.',
            },
            'conv_cache_history': 3,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'janus': {
        'reference': {
            'config_class': 'transformers:JanusConfig',
            'model_class': 'transformers:JanusForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'deepseek-community/Janus-Pro-1B',
                'revision': '1655280bb75959cc1cb85529a2a8b26e7016072e',
                'description': 'Pinned native public image-conditioned ordinary text generation; PerceptionLM also exercises video inputs.',
            },
            'generation_config': {
                '_from_model_config': False,
                'bos_token_id': 100000,
                'eos_token_id': 100001,
                'generation_kwargs': {'boi_token_id': 100003},
                'guidance_scale': 5,
                'pad_token_id': 100002,
                'transformers_version': '4.52.0.dev0',
            },
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 128, 'intermediate_size': 256, 'num_hidden_layers': 2,
                'num_attention_heads': 1, 'num_key_value_heads': 1, 'max_position_embeddings': 512,
            },
            'vision_config': {
                'hidden_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'image_size': 64,
                'projection_dim': 128, 'num_image_tokens': 16,
            },
            'vq_config': {'base_channels': 32, 'latent_channels': 32, 'image_token_embed_dim': 128, 'projection_dim': 128},
        },
        'input': {
            'kind': 'external_prepared',
            'description': 'Explicit common BF16 pixel tensors plus token IDs; full vocabulary and native image placeholder layout.',
        },
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 4, 'do_sample': False, 'generation_mode': 'text'},
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Shrink depth and widths while retaining native head widths, full vocabulary, image encoding '
            'branches, all modal inputs, native pooling and connector ratios. Four ordinary autoregressive '
            'steps.'),
    },

    'jetmoe': {
        'reference': {
            'config_class': 'transformers:JetMoeConfig',
            'model_class': 'transformers:JetMoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'jetmoe/jetmoe-8b',
                'revision': 'd8fd02ccf7911aa8148a63c7984ffd2e465b0352',
                'url': 'https://huggingface.co/jetmoe/jetmoe-8b/blob/d8fd02ccf7911aa8148a63c7984ffd2e465b0352/config.json',
                'description': 'Pinned HF documented matching checkpoint.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 256, 'num_hidden_layers': 3, 'num_key_value_heads': 4,
            'kv_channels': 64, 'vocab_size': 1024, 'num_attention_heads': 8, 'head_dim': 64,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Keep8experts/top2 for BOTH query/output attention and MLP, attentionwidth2xhidden, '
            'KVwidth=hidden, tiedembeddings and biases. Reduce widths/layers; standardRoPE/fullattention '
            'unchanged.'),
    },

    'jina_embeddings_v3': {
        'reference': {
            'config_class': 'transformers:JinaEmbeddingsV3Config',
            'model_class': 'transformers:JinaEmbeddingsV3ForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Preserve the native pinned masked-LM task using its named HF checkpoint configuration.'
                    ' The pinned core class has no active task adapters; documented task-specific adapters '
                    'require separate load_adapter and set_adapter calls.'),
                'checkpoint': 'jinaai/jina-embeddings-v3-hf',
                'revision': 'd18862d9a48706220815554fac3ebb4dfa46fc28',
                'url': 'https://huggingface.co/jinaai/jina-embeddings-v3-hf/blob/d18862d9a48706220815554fac3ebb4dfa46fc28/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'kosmos2': {
        'reference': {
            'config_class': 'transformers:Kosmos2Config',
            'model_class': 'transformers:Kosmos2ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/kosmos-2-patch14-224',
                'revision': 'e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c',
                'url': 'https://huggingface.co/microsoft/kosmos-2-patch14-224/blob/e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c/config.json',
                'description': 'Pinned native task example checkpoint; LXMERT config example selects its base task.',
            },
        },
        'dimension_overrides': {
            'text_config': {'embed_dim': 256, 'ffn_dim': 512, 'vocab_size': 1024},
            'vision_config': {'hidden_size': 128, 'intermediate_size': 256},
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 81,
            'shape': [3, 224, 224], 'image_embeds_positions': list(range(1, 65)),
        },
        'workload': 'forward',
        'outputs': [
            'logits', 'past_key_values', 'image_embeds', 'vision_model_output.last_hidden_state',
            'vision_model_output.pooler_output',
        ],
        'dimension_purpose': ('Retain24vision/24text layers,16vision/32text heads,224image/14patch and64latentqueries; reduce'
            ' widths/vocabulary only; exerciseimageprojection attention and publicvision outputs.'),
        'reference_backend': 'sdpa',
    },

    'kosmos2_5': {
        'reference': {
            'fan_in_normal_modules': ['vision_model'],
            'config_class': 'transformers:Kosmos2_5Config',
            'model_class': 'transformers:Kosmos2_5ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/kosmos-2.5',
                'revision': 'ec3c8051b697166514a31d646cfa36d6ef4c93d7',
                'url': 'https://huggingface.co/microsoft/kosmos-2.5/blob/ec3c8051b697166514a31d646cfa36d6ef4c93d7/config.json',
                'description': 'Pinned native Kosmos2.5 conditional-generation model forward example checkpoint; this case tests its explicit forward outputs and initial cache.',
            },
        },
        'dimension_overrides': {
            'latent_query_num': 64,
            'text_config': {'embed_dim': 128, 'ffn_dim': 256, 'vocab_size': 1024},
            'vision_config': {'hidden_size': 192, 'intermediate_size': 384, 'head_dim': 8},
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 81,
            'shape': [64, 770], 'image_input_name': 'flattened_patches', 'patch_grid_columns': 8,
            'image_embeds_positions': list(range(1, 65)),
        },
        'workload': 'forward',
        'outputs': ['logits', 'past_key_values', 'image_embeds', 'vision_model_output.last_hidden_state'],
        'dimension_purpose': ('Retain18vision/24text layers,24vision/16text heads,768patch width; reduce '
            'hidden/MLP/vocabulary widths,64patchdevelopmentgrid and64latentqueries from2048. All '
            'normalization,projectionattention and decoder paths retained.'),
        'reference_backend': 'sdpa',
        'initialization_reason': (
            'Native vision initialization makes all 18 BF16 attention and MLP residual blocks exact identities. '
            'Use shared fan-in random vision matrices to exercise those blocks; preserve embeddings, '
            'biases, normalization and all other weights.'),
    },

    'kyutai_speech_to_text': {
        'reference': {
            'config_class': 'transformers:KyutaiSpeechToTextConfig',
            'model_class': 'transformers:KyutaiSpeechToTextForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'kyutai/stt-2.6b-en-trfs',
                'revision': '005de8e7800698a4c9963a5ac000e185b410c2f5',
                'description': 'Published streaming waveform generation, native one-frame codec windows and strict FP32 codec.',
            },
            'generation_config': {
                'audio_window_size': 1, 'bos_token_id': 48000, 'cache_implementation': 'sliding_window',
                'codec_cache_implementation': 'sliding_window', 'codec_use_cache': True, 'pad_token_id': 3,
                'transformers_version': '4.53.0.dev0',
            },
        },
        'dimension_overrides': {
            'hidden_size': 128, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 4,
            'head_dim': 32, 'ffn_dim': 512,
        },
        'reference_backend': {'': 'sdpa', 'codec_config': 'sdpa'},
        'input': {
            'kind': 'external',
            'description': ('Native saved feature extractor applied to seeded1snoise at24kHz, with published silence '
                'prefix/delay; full official Mimi codec weights plus random reduced main state.'),
            'dtypes': {'input_values': 'float32'},
        },
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 20},
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'dimension_purpose': ('Main2layers/128hidden but original32codebooks,vocabularies,full native FP32 '
            'codec/all32quantizers;20steps cover BOS,firstwindow, repeatedfirstwindow,advance and '
            'nonzeroaudio aftersilence. Native windows375/250 retained but eviction boundary not reached.'),
    },

    'laguna': {
        'reference': {
            'config_class': 'transformers:LagunaConfig',
            'model_class': 'transformers:LagunaForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'poolside/laguna-XS.2',
                'revision': '69e3f4046616e40fb55ac54e0e2e6accbe5cadfe',
                'url': 'https://huggingface.co/poolside/laguna-XS.2/blob/69e3f4046616e40fb55ac54e0e2e6accbe5cadfe/config.json',
                'description': 'Pinned HF documented matching checkpoint.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 128, 'intermediate_size': 512, 'moe_intermediate_size': 128,
            'shared_expert_intermediate_size': 128, 'num_hidden_layers': 5, 'num_attention_heads': 6,
            'num_key_value_heads': 1, 'num_attention_heads_per_layer': [6, 8, 8, 8, 6], 'head_dim': 64,
            'vocab_size': 1024, 'sliding_window': 128,
            'layer_types': ['full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention'],
            'mlp_layer_types': ['dense', 'sparse', 'sparse', 'sparse', 'sparse'],
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Keep global/local/local/local/global, native6:1 vs8:1 GQA and projectionwidth3x/4x hidden, '
            'globalhalf-headYaRN64 and localfullheadRoPE, per-headsoftplusgate; '
            'densefirstthen256experts/top8/shared. Window512->128 and270tokens exercise window. Reduce '
            'model dimensions.'),
    },

    'lasr': {
        'reference': {
            'config_class': 'transformers:LasrCTCConfig',
            'model_class': 'transformers:LasrForCTC',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/lasr/configuration_lasr.py#L113-L121',
                'description': ('Pinned LasrCTCConfig example explicitly constructs LasrForCTC(LasrCTCConfig()). The '
                    'forward example names unavailable nvidia/lasr-ctc-1.1b; constructor fallback preserves'
                    ' original CTC logits task without equating it to gated google/medasr.'),
            },
        },
        'reference_backend': None,
        'input': {'kind': 'spectrogram', 'name': 'input_features', 'batch_size': 1, 'shape': [2057, 128]},
        'workload': 'forward',
        'outputs': ['logits'],
    },

    'layoutlm': {
        'reference': {
            'config_class': 'transformers:LayoutLMConfig',
            'model_class': 'transformers:LayoutLMForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/layoutlm-base-uncased',
                'revision': '30e3cdd39c11f09757b0fcf7598533d05052acef',
                'url': 'https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/30e3cdd39c11f09757b0fcf7598533d05052acef/config.json',
                'description': ('The pinned masked-LM forward example loads this checkpoint. Retain all six '
                    'box-coordinate embeddings, their ordered sum with text embeddings, and the default '
                    'executed pooler.'),
            },
        },
        'input': {'kind': 'document', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'layoutlmv3': {
        'reference': {
            'config_class': 'transformers:LayoutLMv3Config',
            'model_class': 'transformers:LayoutLMv3Model',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/layoutlmv3-base',
                'revision': 'cfbbbff0762e6aab37086fdd4739ad14fe7d5db4',
                'url': 'https://huggingface.co/microsoft/layoutlmv3-base/blob/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/config.json',
                'description': 'Pinned HF public task forward example checkpoint; retain all enabled image/text paths and native image resolution.',
            },
        },
        'input': {'kind': 'document', 'batch_size': 1, 'sequence_length': 512, 'image_shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'led': {
        'reference': {
            'config_class': 'transformers:LEDConfig',
            'model_class': 'transformers:LEDForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'allenai/led-large-16384-arxiv',
                'revision': '0f8b26971c44af9d4e21edd17ccb5e000f22dac1',
                'url': 'https://huggingface.co/allenai/led-large-16384-arxiv/blob/0f8b26971c44af9d4e21edd17ccb5e000f22dac1/config.json',
                'description': ('First pinned task example is summarization using global attention on first input '
                    'token; second generic example differs and is not silently substituted.'),
            },
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 1537,
            'decoder_sequence_length': 139, 'global_first_token': True,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'levit': {
        'reference': {
            'config_class': 'transformers:LevitConfig',
            'model_class': 'transformers:LevitModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/levit/configuration_levit.py',
                'description': 'Pinned configuration example initializes LevitModel from constructor defaults.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['last_hidden_state', 'pooler_output'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'lfm2': {
        'reference': {
            'config_class': 'transformers:Lfm2Config',
            'model_class': 'transformers:Lfm2ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'LiquidAI/LFM2-1.2B',
                'revision': '40f3da0d0164913923aee9462c23077868b816a3',
                'url': 'https://huggingface.co/LiquidAI/LFM2-1.2B/blob/40f3da0d0164913923aee9462c23077868b816a3/config.json',
                'description': 'Pinned HF config-documentation checkpoint; forward-example meta-lfm2 repository is missing404.',
            },
            'conv_cache_history': 3,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'lfm2_moe': {
        'reference': {
            'config_class': 'transformers:Lfm2MoeConfig',
            'model_class': 'transformers:Lfm2MoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'LiquidAI/LFM2-8B-A1B',
                'revision': 'c1c44ff9fc00db3ebf4516970563f5f383d23670',
                'url': 'https://huggingface.co/LiquidAI/LFM2-8B-A1B/blob/c1c44ff9fc00db3ebf4516970563f5f383d23670/config.json',
                'description': 'Pinned HF config-documentation checkpoint; forward-example meta-lfm2 repository is missing404.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 896, 'moe_intermediate_size': 224,
            'num_hidden_layers': 4, 'num_attention_heads': 4, 'num_key_value_heads': 1, 'vocab_size': 1024,
            'layer_types': ['conv', 'conv', 'full_attention', 'conv'],
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 130},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('First4 blocks retain2dense+2MoE, attention and conv; experts32/top4, head64/4:1 GQA, conv3 '
            'unchanged.'),
    },

    'lfm2_vl': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:Lfm2VlConfig',
            'model_class': 'transformers:Lfm2VlForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'LiquidAI/LFM2-VL-1.6B',
                'revision': '2c2e7f9b0f48a478d7d497ef1864519dfd3994cd',
                'url': 'https://huggingface.co/LiquidAI/LFM2-VL-1.6B/blob/2c2e7f9b0f48a478d7d497ef1864519dfd3994cd/config.json',
                'description': 'Pinned native HF public image-text task example, SigLIP2 patches and LFM2 hybrid convolution/attention decoder.',
            },
            'prefill_input_names': ['pixel_values', 'spatial_shapes', 'pixel_attention_mask'],
        },
        'dimension_overrides': {
            'projector_hidden_size': 320,
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 1536, 'block_ff_dim': 1536, 'num_hidden_layers': 4,
                'num_attention_heads': 4, 'num_key_value_heads': 1, 'vocab_size': 512,
                'max_position_embeddings': 1024, 'layer_types': ['conv', 'conv', 'full_attention', 'conv'],
            },
            'vision_config': {
                'hidden_size': 144, 'intermediate_size': 512, 'num_hidden_layers': 2,
                'num_attention_heads': 2, 'num_patches': 16,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [32, 768],
            'spatial_shapes': [[4, 6]], 'sequence_length': 19, 'image_token_positions': [3, 4, 5, 6, 7, 8],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values', 'image_hidden_states'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain head64/4:1 grouped attention, initial conv/conv/attention/conv blocks and conv3 states;'
            ' native SigLIP2 head72/patch16, source4x4 learned positions resized to4x6 valid patches '
            'in32slot container; factor2 projector emits6imageslots. All caches and default image features '
            'compared.'),
    },

    'lightglue': {
        'reference': {
            'config_class': 'transformers:LightGlueConfig',
            'model_class': 'transformers:LightGlueForKeypointMatching',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ETH-CVG/lightglue_superpoint',
                'revision': '5f5f626efee99f37dbc9eafea879d24d297aeab8',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/lightglue/modeling_lightglue.py',
                'description': 'Pinned native public image-pair matching example, including the invoked SuperPoint detector.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [2, 3, 480, 640]},
        'outputs': ['matches', 'matching_scores', 'keypoints', 'prune', 'mask'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'lighton_ocr': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:LightOnOcrConfig',
            'model_class': 'transformers:LightOnOcrForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'lightonai/LightOnOCR-1B-1025',
                'revision': '7e3e7b0cb83e237e7d237af5a583a002ea632547',
                'description': 'Official pinned HF conditional-generation example and published default configuration.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/lighton_ocr/modeling_lighton_ocr.py',
            },
            'prefill_input_names': ['pixel_values', 'image_sizes'],
        },
        'dimension_overrides': {
            'vision_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'image_size': 84,
            },
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 1024, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 128,
                'layer_types': ['full_attention', 'full_attention'],
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 84, 84],
            'sequence_length': 40, 'image_token_positions': [3, 4, 5, 6, 7, 8], 'image_sizes': [[56, 84]],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values', 'image_hidden_states'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain a language head width of 128 even though hidden width divided by head count is 64, 2:1 '
            'grouped-query attention, tied embeddings, and query/key normalization. A padded 84 by 84 image'
            ' with actual size 56 by 84 exercises image cropping, two-dimensional positions, and the '
            'learned 2 by 2 patch merge.'),
    },

    'lilt': {
        'reference': {
            'config_class': 'transformers:LiltConfig',
            'model_class': 'transformers:LiltModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'SCUT-DLVCLab/lilt-roberta-en-base',
                'revision': '96328faae824978e47bd0e8ea09f66b85efda5f5',
                'url': 'https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base/resolve/96328faae824978e47bd0e8ea09f66b85efda5f5/config.json',
                'description': ('The pinned base-model forward example loads this checkpoint. Retain pad-aware '
                    'positions, separate text/layout streams with coupled attention scores, both complete '
                    'feedforward paths and the default text pooler.'),
            },
        },
        'input': {'kind': 'document', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'llama': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:LlamaConfig',
            'model_class': 'transformers:LlamaForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'constructor_defaults',
                'description': ('Complete pinned LlamaConfig defaults; the named causal-LM example '
                    'meta-llama/Llama-2-7b-hf configuration was inaccessible (HTTP 403), so no checkpoint '
                    'configuration is claimed. Pinned constructor retains MHA, derived head dimension, full'
                    ' default RoPE, SiLU SwiGLU, bias-free projections, and untied LM head.'),
                'example_checkpoint': 'meta-llama/Llama-2-7b-hf',
                'example_checkpoint_revision': '01c7f73d771dfac7d292323805ebc428287df4f9',
                'example_configuration_access': 'HTTP 403 GatedRepoError; metadata revision only',
                'source_locations': [
                    'src/transformers/models/llama/configuration_llama.py:LlamaConfig',
                    'src/transformers/models/llama/modeling_llama.py:LlamaForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'llama4': {'reference': {'config_class': 'transformers:Llama4TextConfig',
                   'model_class': 'transformers:Llama4ForCausalLM',
                   'source': {'kind': 'constructor_defaults',
                              'description': 'Text component of the documented Llama4Config() '
                                             'defaults, selecting the ordinary text-only task in '
                                             'model_doc/llama4.md. This does not claim Scout '
                                             'checkpoint or image-path coverage.',
                              'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                              'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/llama4/configuration_llama4.py#L208-L255'}},
     'reference_backend': None,
     'dimension_overrides': {'hidden_size': 640,
                             'intermediate_size': 1024,
                             'intermediate_size_mlp': 2048,
                             'num_hidden_layers': 4,
                             'num_attention_heads': 5,
                             'num_key_value_heads': 1,
                             'vocab_size': 256,
                             'max_position_embeddings': 128,
                             'attention_chunk_size': 64,
                             'floor_scale': 64},
     'dimension_purpose': 'Retain native 128-wide heads, 5:1 query/KV grouping, 1.6:1 expert FF ratio, '
                          '16 experts, top-one routing plus a shared expert, RMS Q/K normalization, and complete '
                          'three-RoPE/one-NoPE cycle with four MoE layers. Width scales by 1/8; depth '
                          'preserves every block kind. Reduce chunk length and NoPE temperature '
                          'threshold together from 8192 to 64 to exercise both boundaries in the 65-token '
                          'prefix; two continuation calls verify retained 63-token chunk caches and '
                          'growing full cache. Vocabulary 256 and 128 positions reduce storage only.',
     'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 67},
     'workload': 'causal_lm_continuation',
     'outputs': ['logits', 'past_key_values']},

    'llava': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'native_cache_defaults': True,
            'native_position_ids': True,
            'config_class': 'transformers:LlavaConfig',
            'model_class': 'transformers:LlavaForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'llava-hf/llava-1.5-7b-hf',
                'revision': 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
                'url': 'https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/config.json',
                'description': ('Pinned public LLaVA1.5 conditionalgeneration example: CLIP penultimate patch features '
                    '(-2), GELU projector, ordinary Llama causaldecoder; no full-feature or multilayer '
                    'selection.'),
            },
            'prefill_input_names': ['pixel_values'],
        },
        'input': {
            'kind': 'text_image', 'image_batch_size': 1, 'text_batch_size': 1, 'shape': [3, 336, 336],
            'sequence_length': 642, 'image_token_positions': list(range(3, 579)), 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'llava_next': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'native_cache_defaults': True,
            'native_position_ids': True,
            'config_class': 'transformers:LlavaNextConfig',
            'model_class': 'transformers:LlavaNextForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'llava-hf/llava-v1.6-mistral-7b-hf',
                'revision': '2424fdd47412fccc66d91719126b420e9fbd7065',
                'url': 'https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/blob/2424fdd47412fccc66d91719126b420e9fbd7065/config.json',
                'description': ('Pinned public LLaVA-NeXT conditional generation example, Mistral-v0.2 full attention. '
                    'Native any-resolution crop packing, spatial unpadding and learned row newlines '
                    'retained.'),
            },
            'prefill_input_names': ['pixel_values', 'image_sizes'],
        },
        'input': {
            'kind': 'text_image', 'image_batch_size': 1, 'text_batch_size': 1, 'shape': [3, 3, 336, 336],
            'sequence_length': 1328, 'image_token_positions': list(range(3, 1265)),
            'image_sizes': [[150, 540]], 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'llava_next_video': {
        'reference': {
            'native_cache_defaults': True,
            'native_position_ids': True,
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:LlavaNextVideoConfig',
            'model_class': 'transformers:LlavaNextVideoForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'llava-hf/LLaVA-NeXT-Video-7B-hf',
                'revision': 'f417d9edeb69a9eaa88283780332d00ce3c66593',
                'url': 'https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/blob/f417d9edeb69a9eaa88283780332d00ce3c66593/config.json',
                'description': ('Pinned documented LLaVA-NeXT-Video checkpoint. Preserve any-resolution image '
                    'tiling/unpadding/newlines plus default average spatial pooling of video frames; native'
                    ' linear RoPE factor2.5 retained through fixed-cache initialization patch.'),
            },
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_sizes'],
        },
        'input': {
            'kind': 'text_image', 'image_batch_size': 1, 'text_batch_size': 1, 'shape': [3, 3, 336, 336],
            'sequence_length': 2480, 'image_token_positions': list(range(3, 1265)),
            'image_sizes': [[150, 540]], 'video_shape': [8, 3, 336, 336],
            'video_token_positions': list(range(1273, 2425)), 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'video_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'llava_onevision': {
        'reference': {
            'native_cache_defaults': True,
            'native_position_ids': True,
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:LlavaOnevisionConfig',
            'model_class': 'transformers:LlavaOnevisionForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
                'revision': '0d50680527681998e456c7b78950205bedd8a068',
                'url': 'https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf/blob/0d50680527681998e456c7b78950205bedd8a068/config.json',
                'description': ('Pinned public OneVision image/video conditional generation checkpoint. Preserve SigLIP'
                    ' final pre-normalization features without head, native anyres_max_9 downsampling, '
                    'learned newlines, bilinear video pooling, Qwen2 GQA and vocabulary.'),
            },
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_sizes'],
        },
        'input': {
            'kind': 'text_image', 'image_batch_size': 1, 'text_batch_size': 1, 'shape': [26, 3, 384, 384],
            'sequence_length': 8876, 'image_token_positions': list(range(3, 7244)),
            'image_sizes': [[1632, 1904]], 'video_shape': [8, 3, 384, 384],
            'video_token_positions': list(range(7252, 8821)), 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'video_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'longcat_flash': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:LongcatFlashConfig',
            'model_class': 'transformers:LongcatFlashForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'meituan-longcat/LongCat-Flash-Chat',
                'revision': 'f0b8e8d7380330f450ca1fde4ec56823a514584c',
                'description': 'Pinned HF configuration documented official LongCatcheckpoint. model_type omitted inlegacyconfig butarchitectureclassmatches.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'ffn_hidden_size': 512, 'expert_ffn_hidden_size': 128, 'num_layers': 2,
            'num_hidden_layers': 4, 'num_attention_heads': 4, 'num_key_value_heads': 4, 'q_lora_rank': 128,
            'kv_lora_rank': 64, 'qk_nope_head_dim': 64, 'qk_rope_head_dim': 32, 'v_head_dim': 64,
            'head_dim': 32, 'n_routed_experts': 16, 'zero_expert_num': 8, 'vocab_size': 512,
            'max_position_embeddings': 128,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 18},
        'outputs': ['logits', 'past_key_values'],
        'workload': 'causal_lm_continuation',
        'dimension_purpose': ('Two logicalblocks eachtwoMLAattentions/twodenseMLPs+shortcutMoE; '
            'bothlearned16/identity8experts, top12of24; preserveLoRAscalings/defaultinterleavedinputRoPE; '
            'reducewidth/depth/ranks.'),
    },

    'longformer': {
        'reference': {
            'config_class': 'transformers:LongformerConfig',
            'model_class': 'transformers:LongformerForMaskedLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'allenai/longformer-base-4096',
                'revision': '301e6a42cb0d9976a6d6a26a079fef81c18aa895',
                'url': 'https://huggingface.co/allenai/longformer-base-4096/blob/301e6a42cb0d9976a6d6a26a079fef81c18aa895/config.json',
                'description': 'Pinned HF masked-LM example omits global attention; preserve this local-only task path.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 769},
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'longt5': {
        'reference': {
            'config_class': 'transformers:LongT5Config',
            'model_class': 'transformers:LongT5ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Stancld/longt5-tglobal-large-16384-pubmed-3k_steps',
                'revision': '5d5845823ae91ea8230b849c8cac211794c58064',
                'description': 'First pinned public LongT5ForConditionalGeneration example; retain gated GELU, untied head and transient-global encoder.',
            },
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 513,
            'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'luke': {
        'reference': {
            'config_class': 'transformers:LukeConfig',
            'model_class': 'transformers:LukeModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'studio-ousia/luke-base',
                'revision': '7438924defd9f3c2018d63c16073bf4bcb6a70aa',
                'url': 'https://huggingface.co/studio-ousia/luke-base/blob/7438924defd9f3c2018d63c16073bf4bcb6a70aa/config.json',
                'description': 'HF LukeModel documented entity-bearing input example.',
            },
        },
        'input': {
            'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512,
            'entity_mentions': [[1], [3, 4], [7, 8, 9], [11, 12, 13, 14]],
        },
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'entity_last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'lw_detr': {
        'reference': {
            'config_class': 'transformers:LwDetrConfig',
            'model_class': 'transformers:LwDetrForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'AnnaZhang/lwdetr_small_60e_coco',
                'revision': 'deebf67dc6e174fe3f4c83d7d5f9ea7eaa651197',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/lw_detr/modeling_lw_detr.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 640]},
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'intermediate_hidden_states',
            'intermediate_reference_points', 'init_reference_points', 'enc_outputs_class',
            'enc_outputs_coord_logits',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'lxmert': {
        'reference': {
            'config_class': 'transformers:LxmertConfig',
            'model_class': 'transformers:LxmertModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'unc-nlp/lxmert-base-uncased',
                'revision': '628572c96242d1496147beec1c13a1bb7869605d',
                'url': 'https://huggingface.co/unc-nlp/lxmert-base-uncased/blob/628572c96242d1496147beec1c13a1bb7869605d/config.json',
                'description': 'Pinned native task example checkpoint; LXMERT config example selects its base task.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 19, 'visual_region_count': 36},
        'workload': 'forward',
        'outputs': ['language_output', 'vision_output', 'pooled_output'],
        'reference_backend': None,
    },

    'm2m_100': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:M2M100Config',
            'model_class': 'transformers:M2M100ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/m2m100_418M',
                'revision': '55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636',
                'description': ('Pinned M2M100ForConditionalGeneration task uses facebook/m2m100_418M; ReLU, sinusoidal'
                    ' padding-aware positions, tied output and stack final norms.'),
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139, 'encoder_prefix_token_ids': [128022],
            'encoder_suffix_token_ids': [2],
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'mamba': {
        'reference': {
            'config_class': 'transformers:MambaConfig',
            'model_class': 'transformers:MambaForCausalLM',
            'source': {
                'checkpoint': 'state-spaces/mamba-130m-hf',
                'revision': '1e76775f628fbf1350fbe4dbb3d971ba64af25a1',
                'url': 'https://huggingface.co/state-spaces/mamba-130m-hf/blob/1e76775f628fbf1350fbe4dbb3d971ba64af25a1/config.json',
                'kind': 'example_checkpoint',
                'description': ('Pinned HF docs/source/en/model_doc/mamba.md:58 first ordinary causal-LM example. '
                    'Complete checkpoint configuration with pinned constructor defaults. Preserve tied '
                    'embeddings and Mamba-v1 recurrent blocks.'),
            },
            'cache_argument': 'cache_params',
            'cache_output': 'cache_params',
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 130},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'cache_params'],
    },

    'mamba2': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'config_overrides': {},
        'reference': {
            'config_class': 'transformers:Mamba2Config',
            'model_class': 'transformers:Mamba2ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'mistralai/Mamba-Codestral-7B-v0.1',
                'revision': '4f086c08c1e0f07bdc50ca25125dbbf7475d21da',
                'url': 'https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1/resolve/4f086c08c1e0f07bdc50ca25125dbbf7475d21da/config.json',
                'description': 'Pinned Transformers docs/source/en/model_doc/mamba2.md causal LM example; computational settings from this checkpoint config.',
            },
            'forward_kwargs': {'logits_to_keep': 0},
            'cache_argument': 'cache_params',
            'cache_output': 'cache_params',
            'conv_cache_history': 3,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 270},
        'outputs': ['logits', 'cache_params'],
    },

    'marian': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:MarianConfig',
            'model_class': 'transformers:MarianMTModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Helsinki-NLP/opus-mt-en-de',
                'revision': '6183067f769a302e3861815543b9f312c71b0ca4',
                'description': ('Pinned MarianMTModel translation example uses Helsinki-NLP/opus-mt-en-de; frozen '
                    'sinusoidal positions and tied shared embeddings retained.'),
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'markuplm': {
        'reference': {
            'config_class': 'transformers:MarkupLMConfig',
            'model_class': 'transformers:MarkupLMModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/markuplm-base',
                'revision': '7b1ef356189e6803bd5b58b305c783fd1d395ae6',
                'url': 'https://huggingface.co/microsoft/markuplm-base/blob/7b1ef356189e6803bd5b58b305c783fd1d395ae6/config.json',
                'description': 'Pinned HF public base-model forward example checkpoint, retaining ordinary document/table metadata.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512, 'xpath_depth': 50},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'mask2former': {
        'reference': {
            'config_class': 'transformers:Mask2FormerConfig',
            'model_class': 'transformers:Mask2FormerForUniversalSegmentation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/mask2former-swin-small-coco-instance',
                'revision': '1b7ac418a0f3cb5edfc5037e488d53587b8bb76b',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mask2former/modeling_mask2former.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 384, 384]},
        'outputs': [
            'class_queries_logits', 'masks_queries_logits', 'encoder_last_hidden_state',
            'pixel_decoder_last_hidden_state', 'transformer_decoder_last_hidden_state',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'maskformer': {
        'reference': {
            'config_class': 'transformers:MaskFormerConfig',
            'model_class': 'transformers:MaskFormerForInstanceSegmentation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/maskformer-swin-base-ade',
                'revision': 'b569351e060953d37fd7dfb8b16ab83c360a13d6',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/maskformer/modeling_maskformer.py',
                'description': 'Pinned first public-task example, ADE semantic segmentation.',
            },
        },
        'default_dtype': 'float32',
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 864]},
        'outputs': [
            'class_queries_logits', 'masks_queries_logits', 'encoder_last_hidden_state',
            'pixel_decoder_last_hidden_state', 'transformer_decoder_last_hidden_state',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'maskformer_swin': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:MaskFormerSwinConfig',
            'model_class': 'transformers:MaskFormerSwinModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/maskformer/configuration_maskformer_swin.py',
                'description': ('Pinned configuration documentation explicitly constructs the base model with '
                    'constructor defaults; preserve all four stages and default pooling.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['last_hidden_state', 'pooler_output'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'mbart': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:MBartConfig',
            'model_class': 'transformers:MBartForConditionalGeneration',
            'forward_kwargs': {},
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'facebook/mbart-large-en-ro',
                'revision': '3f6705eea8aef516fdbdbf3ec20f63a6a522a6ae',
                'description': 'First pinned conditional-generation translation example (en-ro), not the later cc25 mask-filling example.',
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139, 'encoder_prefix_token_ids': [],
            'encoder_suffix_token_ids': [2, 250004], 'content_token_id_min': 6,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'megatron_bert': {
        'reference': {
            'config_class': 'transformers:MegatronBertConfig',
            'model_class': 'transformers:MegatronBertForMaskedLM',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('Retain existing corpus masked-LM task; its pinned class has no dedicated checkpoint '
                    'forward example. Use constructor computation defaults, whose configuration '
                    'documentation constructs the model from defaults.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/megatron_bert/configuration_megatron_bert.py',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'metaclip_2': {
        'reference': {
            'config_class': 'transformers.models.metaclip_2.configuration_metaclip_2:MetaClip2Config',
            'model_class': 'transformers.models.metaclip_2.modeling_metaclip_2:MetaClip2Model',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/metaclip-2-worldwide-huge-quickgelu',
                'revision': 'c139061af7b10fdb2e754b60d2b1182a3d5526c2',
                'hf_source_revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned MetaClip2Model.forward paired text/image example names this checkpoint. '
                    'Preserve both towers, projections, normalized embeddings, all contrasts, and already '
                    'computed nested hidden/projected pooler outputs.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L842',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 3, 'image_batch_size': 2, 'sequence_length': 77,
            'shape': [3, 224, 224], 'eos_positions': [76, 76, 76], 'batch_size': 1, 'bos_token_id': 49406,
            'eos_token_id': 2, 'pad_token_id': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'mgp_str': {
        'reference': {
            'config_class': 'transformers:MgpstrConfig',
            'model_class': 'transformers:MgpstrForSceneTextRecognition',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'alibaba-damo/mgp-str-base',
                'revision': '5d06493b6b2a8c4c023d2c030175c03be30f4202',
                'url': 'https://huggingface.co/alibaba-damo/mgp-str-base/blob/5d06493b6b2a8c4c023d2c030175c03be30f4202/config.json',
                'description': 'Pinned HF public task forward example checkpoint; retain all enabled image/text paths and native image resolution.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 32, 128]},
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'mimi': {
        'reference': {
            'config_class': 'transformers:MimiConfig',
            'model_class': 'transformers:MimiModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'kyutai/mimi',
                'revision': '89091b3e466eb6a9d11e537bf26b144f194978f7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mimi/modeling_mimi.py#L1725',
                'description': ('Pinned HF forward example selects kyutai/mimi. Published config disables '
                    'caches/streaming; both quantizer groups and all32 quantizers, sliding attention '
                    'retained.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [1, 241920]},
        'workload': 'forward',
        'outputs': ['audio_codes', 'audio_values'],
        'reference_backend': None,
    },

    'minicpmv4_6': {
        'reference': {
            'config_class': 'transformers:MiniCPMV4_6Config',
            'model_class': 'transformers:MiniCPMV4_6ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openbmb/MiniCPM-V-4_6',
                'revision': '36f34a661a4bd35d0dc2294cb044d2584646c7d3',
                'url': 'https://huggingface.co/openbmb/MiniCPM-V-4_6/blob/36f34a661a4bd35d0dc2294cb044d2584646c7d3/config.json',
                'description': ('Pinned HF task guide uses official MiniCPM-V-4_6; config auto-doc uses different '
                    'dotted spelling. Full image/video default16xmerger path.'),
            },
            'prefill_input_names': ['pixel_values', 'target_sizes', 'pixel_values_videos', 'target_sizes_videos'],
        },
        'dimension_overrides': {
            'image_token_id': 400,
            'video_token_id': 401,
            'vision_config': {
                'hidden_size': 144, 'intermediate_size': 538, 'num_hidden_layers': 8,
                'num_attention_heads': 2, 'image_size': 56,
            },
            'text_config': {
                'hidden_size': 512, 'intermediate_size': 1792, 'num_hidden_layers': 4,
                'num_attention_heads': 4, 'num_key_value_heads': 1, 'linear_num_key_heads': 2,
                'linear_num_value_heads': 2, 'vocab_size': 512, 'max_position_embeddings': 1024,
                'layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 14, 448],
            'video_shape': [3, 14, 448], 'target_sizes': [[4, 8]], 'target_sizes_videos': [[4, 4], [4, 4]],
            'sequence_length': 20, 'image_token_positions': [3, 4], 'video_token_positions': [8, 9],
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Nativeinsertlayer6retainedwith8visionlayers,before/aftermergerattention;patch14/head72,rectangular4x8image+two4x4videogrids,2x2windowattention/meanresidualmerge'
            ' then2x2MLPmerge. '
            'Qwen3.5texthead256/partial64,4:1GQA,attentionwidth2xhidden,linearheads128/key:value1:1,3:1hybridpattern,tiedhead,allcacheoutputs.'),
    },

    'minimax': {
        'reference': {
            'config_class': 'transformers:MiniMaxConfig',
            'model_class': 'transformers:MiniMaxForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'MiniMaxAI/MiniMax-Text-01-hf',
                'revision': 'f7ce01366e8585a8948f19aedc8e20628c6965e5',
                'url': 'https://huggingface.co/MiniMaxAI/MiniMax-Text-01-hf/blob/f7ce01366e8585a8948f19aedc8e20628c6965e5/config.json',
                'description': 'Pinned HF documented official MiniMax text checkpoint.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 384,
            'intermediate_size': 256,
            'num_hidden_layers': 9,
            'num_attention_heads': 8,
            'num_key_value_heads': 1,
            'head_dim': 64,
            'vocab_size': 1024,
            'max_position_embeddings': 2048,
            'layer_types': [
                'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention',
                'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
                'linear_attention',
            ],
            'bos_token_id': 1022,
            'eos_token_id': 1023,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain seven linear blocks then full attention then linear, attention width4/3 hidden,8:1 '
            'GQA,32 experts/top2 and unchanged residual factors/block256.270 tokens exercise two recurrence'
            ' blocks plus cached decode. Reduce model/context sizes; pinned source ignores extra rotary_dim'
            ' metadata and uses full-head RoPE.'),
    },

    'minimax_m2': {
        'reference': {
            'config_class': 'transformers:MiniMaxM2Config',
            'model_class': 'transformers:MiniMaxM2ForCausalLM',
            'load_device': 'cuda',
            'serialized_weight_suffixes': ['weight_scale_inv', 'gate_up_proj_scale_inv', 'down_proj_scale_inv'],
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'MiniMaxAI/MiniMax-M2',
                'revision': '757303d492a50514c312788b5247a4f696a4c6a3',
                'url': 'https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/757303d492a50514c312788b5247a4f696a4c6a3/config.json',
                'description': 'Root-approved matching author fallback after invalid/mismatched pinned HF documentation. Native block FP8 retained.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 384, 'intermediate_size': 128, 'num_hidden_layers': 3, 'num_attention_heads': 6,
            'num_key_value_heads': 1, 'head_dim': 128, 'vocab_size': 1024, 'max_position_embeddings': 2048,
            'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 129},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Keep native blockFP8 weights/dynamicactivationquantization,256experts/top8, attention head '
            'dimensions and all enabled layer types; reduce hidden/intermediate widths and layer count for '
            'development.'),
    },

    'ministral': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:MinistralConfig',
            'model_class': 'transformers:MinistralForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'mistralai/Ministral-8B-Instruct-2410',
                'revision': '2f494a194c5b980dfb9772cb92d26cbb671fce5a',
                'description': ('Pinned documentation checkpoint uses legacy Mistral metadata; load its explicit '
                    'mixed-attention settings with the public Ministral class.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'ministral3': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:Ministral3Config',
            'model_class': 'transformers:Ministral3ForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'mistralai/Ministral-3-8B-Base-2512',
                'revision': 'd4883f9b36aa2e5d775730d3fdba3d30de51a8ef',
                'url': 'https://huggingface.co/mistralai/Ministral-3-8B-Base-2512/blob/d4883f9b36aa2e5d775730d3fdba3d30de51a8ef/config.json',
                'description': ('Pinned HF task documentation official checkpoint; preserve enabled computation. Text '
                    'config of the documented multimodal checkpoint is the HF Ministral3 text task.'),
                'config_key': 'text_config',
            },
        },
        'dimension_overrides': {
            'hidden_size': 512,
            'intermediate_size': 256,
            'num_hidden_layers': 3,
            'num_attention_heads': 4,
            'num_key_value_heads': 1,
            'head_dim': 128,
            'vocab_size': 1024,
            'max_position_embeddings': 2048,
            'rope_parameters': {'original_max_position_embeddings': 128},
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain head128/4:1 GQA, YaRN factor16/mscale ratio and query-temperature beta. Reduce context '
            'boundary16384 to128 and max context262144 to2048;270 tokens exercise before/after boundary and'
            ' decode at nonunit temperature. This is a computational branch test, not a representative '
            'performance workload.'),
    },

    'mistral': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:MistralConfig',
            'model_class': 'transformers:MistralForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'mistralai/Mistral-7B-v0.1',
                'revision': '27d67f1b5f57dc0953326b2601d68371d40ea8da',
                'url': 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/27d67f1b5f57dc0953326b2601d68371d40ea8da/config.json',
                'description': ('Pinned causal-LM example identifier meta-mistral/Mistral-2-7b-hf returned HTTP404; use'
                    ' the official checkpoint named by the pinned configuration-class documentation. '
                    'Preserve its computational settings.'),
                'invalid_causal_lm_example': 'meta-mistral/Mistral-2-7b-hf',
                'invalid_causal_lm_example_access': 'HTTP404 RepositoryNotFoundError',
                'source_locations': [
                    'src/transformers/models/mistral/configuration_mistral.py:MistralConfig',
                    'src/transformers/models/mistral/modeling_mistral.py:MistralForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 4099},
        'outputs': ['logits', 'past_key_values'],
    },

    'mistral3': {
        'reference': {
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:Mistral3Config',
            'model_class': 'transformers:Mistral3ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
                'revision': '68faf511d618ef198fef186659617cfd2eb8e33a',
                'description': 'Pinned HF Mistral3 public image-conditioned generation checkpoint.',
            },
            'prefill_input_names': ['pixel_values', 'image_sizes'],
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 512, 'intermediate_size': 1024, 'num_attention_heads': 8,
                'num_key_value_heads': 2, 'head_dim': 64, 'vocab_size': 1024,
            },
            'vision_config': {'hidden_size': 256, 'intermediate_size': 512, 'num_attention_heads': 4, 'image_size': 84},
        },
        'reference_backend': {'': 'sdpa', 'text_config': 'sdpa', 'vision_config': 'sdpa'},
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 12,
            'shape': [3, 56, 84], 'image_sizes': [[56, 84]], 'image_token_positions': [0, 1, 2, 3, 4, 5],
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values', 'image_hidden_states'],
        'dimension_purpose': ('Retain40text/24visionlayers,4:1textGQA,2x2spatialmerger; reducewidths/headcount/vocab/image. '
            'Rectangular4x6patchgrid exercises bothvisionpositionaxes and6mergedimagefeatures.'),
    },

    'mistral4': {
        'reference': {
            'config_class': 'transformers:Mistral4Config',
            'model_class': 'transformers:Mistral4ForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'mistralai/Mistral-Small-4-119B-2603',
                'revision': 'a11f36bebf709121056b1dbcc943d1c6afbe494d',
                'config_key': 'text_config',
                'description': ('Official HF Mistral3 wrapper checkpoint; select native Mistral4 text config and '
                    'explicitly retain wrapper static FP8 quantization.'),
            },
            'load_device': 'cuda',
            'serialized_weight_suffixes': ['weight_scale_inv', 'gate_up_proj_scale_inv', 'down_proj_scale_inv', 'activation_scale'],
        },
        'config_overrides': {
            'quantization_config': {
                'activation_scheme': 'static', 'dequantize': False,
                'modules_to_not_convert': ['model.vision_tower', 'model.multi_modal_projector', 'lm_head'],
                'quant_method': 'fp8', 'weight_block_size': None,
            },
        },
        'dimension_overrides': {
            'hidden_size': 128, 'num_attention_heads': 2, 'num_key_value_heads': 2, 'q_lora_rank': 64,
            'kv_lora_rank': 64, 'n_routed_experts': 8, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 2, 'vocab_size': 256, 'max_position_embeddings': 128,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 17},
        'outputs': ['logits'],
        'workload': 'causal_lm',
        'dimension_purpose': ('Retain two all-MoE layers, MLA query/KV ranks, native 128-wide Q/K/V heads with64rotary '
            'dimensions, top4of8experts/sharedexpert, YaRN128 and static per-tensor FP8 at each projection.'
            ' Reduced context means Llama4 long-position scale is implemented but unity for primary test '
            'positions.'),
        'reference_backend': 'eager',
    },

    'mixtral': {
        'reference': {
            'config_class': 'transformers:MixtralConfig',
            'model_class': 'transformers:MixtralForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned docs/source/en/model_doc/mixtral.md:69 loads the base causal LM. Its checkpoint'
                    ' disables sliding_window despite the nearby prose describing sliding windows; retain '
                    'eight experts, top-two renormalized routing, and 4:1 grouped-query attention.'),
                'checkpoint': 'mistralai/Mixtral-8x7B-v0.1',
                'revision': 'fc7ac94680e38d7348cfa806e51218e6273104b0',
                'url': 'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/fc7ac94680e38d7348cfa806e51218e6273104b0/config.json',
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'mlcd': {
        'reference': {
            'config_class': 'transformers.models.mlcd.configuration_mlcd:MLCDVisionConfig',
            'model_class': 'transformers.models.mlcd.modeling_mlcd:MLCDVisionModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'DeepGlint-AI/mlcd-vit-bigG-patch14-448',
                'revision': '4a9e11ca0052ee930f2a180fa716cd47287e801e',
                'hf_source_revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned MLCDVisionModel.forward example names the patch14-448 checkpoint. Preserve '
                    'ordinary base-model outputs; its optional attention-shape diagnostic is omitted.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mlcd/modeling_mlcd.py#L484-L505',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 448, 448]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'mm_grounding_dino': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:MMGroundingDinoConfig',
            'model_class': 'transformers:MMGroundingDinoForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det',
                'revision': 'ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mm_grounding_dino/configuration_mm_grounding_dino.py',
                'description': ('Pinned configuration documentation names this checkpoint; the modeling example instead'
                    ' instantiates GroundingDINO.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 9,
            'shape': [3, 800, 1066], 'input_ids': [[101, 1037, 4937, 1012, 1037, 6556, 2491, 1012, 102]],
            'token_type_id': 0, 'attention_mask': True, 'batch_size': 1,
        },
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'intermediate_hidden_states',
            'intermediate_reference_points', 'init_reference_points', 'encoder_last_hidden_state_vision',
            'encoder_last_hidden_state_text', 'enc_outputs_class', 'enc_outputs_coord_logits',
            'encoder_logits', 'encoder_pred_boxes', 'input_ids',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'mllama': {'reference': {'config_class': 'transformers:MllamaConfig',
                              'model_class': 'transformers:MllamaForConditionalGeneration',
                              'source': {'kind': 'constructor_defaults',
                                         'description': 'Pinned configuration_mllama.py:155-173 documents '
                                                        'MllamaVisionConfig()+MllamaTextConfig() and '
                                                        'MllamaForConditionalGeneration. Use '
                                                        'keyword-equivalent '
                                                        'MllamaConfig(vision_config=...,text_config=...), '
                                                        'equal to bare defaults; literal '
                                                        'two-positional-argument doc syntax fails strict '
                                                        'constructor. This is a constructor workload, not '
                                                        'inaccessible gated checkpoint config.',
                                         'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                                         'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mllama/configuration_mllama.py#L155-L173'},
                              'native_position_ids': True,
                              'native_cache_defaults': True,
                              'continuation_input_names': ['cross_attention_mask'],
                              'continuation_outputs': ['logits', 'past_key_values'],
                              'randomize_zero_parameters': ['model.vision_model.gated_positional_embedding.gate',
                                                            'model.vision_model.pre_tile_positional_embedding.gate',
                                                            'model.vision_model.post_tile_positional_embedding.gate',
                                                            'model.language_model.layers.1.cross_attn_attn_gate',
                                                            'model.language_model.layers.1.cross_attn_mlp_gate',
                                                            'model.language_model.layers.3.cross_attn_attn_gate',
                                                            'model.language_model.layers.3.cross_attn_mlp_gate']},
                'config_overrides': {},
                'dimension_overrides': {'vision_config': {'hidden_size': 160,
                                                          'attention_heads': 2,
                                                          'intermediate_size': 640,
                                                          'num_hidden_layers': 6,
                                                          'num_global_layers': 2,
                                                          'intermediate_layers_indices': [0, 1, 2, 3, 4],
                                                          'vision_output_dim': 960,
                                                          'image_size': 112},
                                        'text_config': {'hidden_size': 512,
                                                        'intermediate_size': 1792,
                                                        'num_hidden_layers': 5,
                                                        'num_attention_heads': 4,
                                                        'num_key_value_heads': 1,
                                                        'cross_attention_layers': [1, 3],
                                                        'vocab_size': 1024,
                                                        'bos_token_id': 1016,
                                                        'eos_token_id': 1017,
                                                        'pad_token_id': 1020},
                                        'image_token_index': 1024},
                'dimension_purpose': 'Preserve vision head width 80 and 4:1 feed-forward ratio; text head '
                                     'width 128, 4:1 query/KV grouping and 3.5:1 feed-forward ratio. Six '
                                     'local vision blocks retain five intermediate feature taps and the '
                                     'final block; two global blocks retain learned gates. Five text layers '
                                     'alternate self/cross/self/cross/self attention. Four tiles and native '
                                     '14-pixel patches at image size 112 retain 64 patch tokens, a class '
                                     'token and seven padding tokens per tile. A 128-token prefix plus two '
                                     'continuations checks both cache types. Vocabulary 1024 remaps special '
                                     'IDs together and retains eight extra embedding entries. Named zero '
                                     'gates are randomized equally on both sides so the image path is '
                                     'active.',
                'input': {'kind': 'text_image',
                          'shape': [1, 4, 3, 112, 112],
                          'text_batch_size': 1,
                          'image_batch_size': 1,
                          'batch_size': 1,
                          'sequence_length': 130,
                          'image_token_positions': [3],
                          'aspect_ratio_ids': [[2]],
                          'aspect_ratio_mask': [[[1, 1, 0, 0]]],
                          'cross_attention_mask': [[[[0, 0, 0, 0]] if position < 3 else [[1, 1, 0, 0]]
                                                       for position in range(130)]]},
                'workload': 'causal_lm_continuation',
                'outputs': ['logits', 'past_key_values'],
                'decode_outputs': ['logits', 'past_key_values'],
                'reference_backend': 'sdpa',
                'notes': 'Native zero position/cross gates are randomized with shared seeded N(0,.2²) by '
                         'explicit reference.randomize_zero_parameters. Preserve already nonzero global '
                         'gates. CPU checks compare generic prefill+2 and separate vision outputs; no CUDA '
                         'claim. RMSNormNative/SiLU are unchanged internal ops; ProductGate is existing '
                         'explicit patch. Primary runner still needs GPU validation. Generic preparation '
                         'including metadata fields and zero-gate hook verified CPU in '
                         'prepared-cpu-report.json.'},

    'mobilebert': {
        'reference': {
            'config_class': 'transformers:MobileBertConfig',
            'model_class': 'transformers:MobileBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned MobileBERT masked-LM documentation uses google/mobilebert-uncased; retain '
                    'trigram embeddings, both bottleneck branches, four FFNs per layer, NoNorm affine maps,'
                    ' and the factorized tied MLM projection.'),
                'checkpoint': 'google/mobilebert-uncased',
                'revision': '1f90a6c24c7879273a291d34a849033eba2dbc0f',
                'url': 'https://huggingface.co/google/mobilebert-uncased/blob/1f90a6c24c7879273a291d34a849033eba2dbc0f/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'mobilenet_v1': {
        'reference': {
            'config_class': 'transformers.models.mobilenet_v1.configuration_mobilenet_v1:MobileNetV1Config',
            'model_class': 'transformers.models.mobilenet_v1.modeling_mobilenet_v1:MobileNetV1Model',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs MobileNetV1Config() and '
                    'MobileNetV1Model(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mobilenet_v1/configuration_mobilenet_v1.py#L31',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'mobilenet_v2': {
        'reference': {
            'config_class': 'transformers.models.mobilenet_v2.configuration_mobilenet_v2:MobileNetV2Config',
            'model_class': 'transformers.models.mobilenet_v2.modeling_mobilenet_v2:MobileNetV2Model',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs MobileNetV2Config() and '
                    'MobileNetV2Model(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mobilenet_v2/configuration_mobilenet_v2.py#L45',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'mobilevit': {
        'reference': {
            'config_class': 'transformers.models.mobilevit.configuration_mobilevit:MobileViTConfig',
            'model_class': 'transformers.models.mobilevit.modeling_mobilevit:MobileViTModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned MobileViTConfig example explicitly constructs the base MobileViTModel from '
                    'constructor defaults. Preserve its full encoder and default pooler plus output-channel'
                    ' expansion.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mobilevit/configuration_mobilevit.py#L38',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 256, 256]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'mobilevitv2': {
        'reference': {
            'config_class': 'transformers.models.mobilevitv2.configuration_mobilevitv2:MobileViTV2Config',
            'model_class': 'transformers.models.mobilevitv2.modeling_mobilevitv2:MobileViTV2Model',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned MobileViTV2Config example explicitly constructs the base MobileViTV2Model '
                    'from constructor defaults. Preserve its full encoder and default pooler.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mobilevitv2/configuration_mobilevitv2.py#L46',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 256, 256]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'modernbert': {
        'reference': {
            'config_class': 'transformers:ModernBertConfig',
            'model_class': 'transformers:ModernBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'answerdotai/ModernBERT-base',
                'revision': '8949b909ec900327062f0ebf497f51aef5e6f0c8',
                'url': 'https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/config.json',
                'description': ('Pinned HF task example or configuration documentation checkpoint; preserve checkpoint '
                    'computation flags, completed by pinned constructor defaults.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'modernbert_decoder': {
        'reference': {
            'config_class': 'transformers:ModernBertDecoderConfig',
            'model_class': 'transformers:ModernBertDecoderForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'blab-jhu/test-32m-dec',
                'revision': 'cad16cec272968e717fef59dee34fc33473bf731',
                'url': 'https://huggingface.co/blab-jhu/test-32m-dec/blob/cad16cec272968e717fef59dee34fc33473bf731/config.json',
                'description': ('Pinned HF task example or configuration documentation checkpoint; preserve checkpoint '
                    'computation flags, completed by pinned constructor defaults.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 195},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'modernvbert': {
        'reference': {
            'config_class': 'transformers:ModernVBertConfig',
            'model_class': 'transformers:ModernVBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'ModernVBERT/modernvbert',
                'revision': '1d1bece9df23cb109642f3ee5d1ceb331a3d1ee5',
                'url': 'https://huggingface.co/ModernVBERT/modernvbert/blob/1d1bece9df23cb109642f3ee5d1ceb331a3d1ee5/config.json',
                'description': ('Pinned ModernVBertForMaskedLM checkpoint annotation. Preserve the '
                    'masked-language-model head and image features; no retrieval projection is selected.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 1153,
            'shape': [17, 3, 512, 512], 'image_token_positions': list(range(1, 1089)), 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': ['logits', 'image_hidden_states'],
        'reference_backend': None,
    },

    'moonshine': {
        'reference': {
            'config_class': 'transformers:MoonshineConfig',
            'model_class': 'transformers:MoonshineForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'UsefulSensors/moonshine-tiny',
                'revision': '390624ed33d594443aa4aa221f5b9f283b545b5a',
                'url': 'https://huggingface.co/UsefulSensors/moonshine-tiny/blob/390624ed33d594443aa4aa221f5b9f283b545b5a/config.json',
                'description': ('Pinned conditional-generation example selects UsefulSensors/moonshine-tiny; retain raw'
                    ' waveform frontend, partial interleaved RoPE, pad36-wideheads to40, gated decoder and '
                    'tied head.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [48000], 'decoder_sequence_length': 139},
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'moshi': {
        'reference': {
            'config_class': 'transformers:MoshiConfig',
            'model_class': 'transformers:MoshiForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'kmhf/hf-moshiko',
                'revision': '06e3abeb0efb7452a21736d39dcde39411c09982',
                'url': 'https://huggingface.co/kmhf/hf-moshiko/blob/06e3abeb0efb7452a21736d39dcde39411c09982/config.json',
                'description': ('Pinned HF public forward example uses get_unconditional_inputs to supply text BOS and '
                    'both audio-code BOS streams. Development inputs retain that first step and append '
                    'ordinary token/code inputs; audio codec and label-dependent depth decoder do not '
                    'execute.'),
            },
        },
        'dimension_overrides': {
            'hidden_size': 128,
            'num_attention_heads': 4,
            'num_key_value_heads': 4,
            'head_dim': 32,
            'ffn_dim': 512,
            'num_hidden_layers': 2,
            'sliding_window': 16,
            'max_position_embeddings': 128,
            'audio_encoder_config': {
                'hidden_size': 32, 'num_filters': 4, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                'num_key_value_heads': 2, 'head_dim': 16, 'intermediate_size': 64,
                'vector_quantization_hidden_dimension': 16, 'codebook_dim': 16, 'upsample_groups': 32,
            },
            'depth_decoder_config': {
                'hidden_size': 64, 'input_size': 128, 'num_attention_heads': 4, 'num_key_value_heads': 4,
                'head_dim': 16, 'ffn_dim': 128, 'num_hidden_layers': 1,
            },
        },
        'input': {'kind': 'moshi_forward', 'batch_size': 1, 'sequence_length': 19},
        'outputs': ['logits', 'last_hidden_state', 'past_key_values', 'depth_past_key_values'],
        'workload': 'forward',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduce temporal width, heads, layers and rolling-cache window to16; use19tokens to exercise '
            'cache truncation. Pinned first forward is full causal attention and retains15cache entries, '
            'verified by native hooks; no author-intent claim. Keep native vocabularies/eight audio '
            'codebooks, text logits, final hidden state and both public cache aliases. Inactive codec/depth'
            ' constructor dimensions reduced for preparation.'),
    },

    'mpnet': {
        'reference': {
            'config_class': 'transformers:MPNetConfig',
            'model_class': 'transformers:MPNetForMaskedLM',
            'source': {
                'checkpoint': 'microsoft/mpnet-base',
                'revision': '6996ce1e91bd2a9c7d7f61daec37463394f73f09',
                'url': 'https://huggingface.co/microsoft/mpnet-base/blob/6996ce1e91bd2a9c7d7f61daec37463394f73f09/config.json',
                'kind': 'example_checkpoint',
                'description': ('Preserve the pinned MPNet masked-LM task and its microsoft/mpnet-base example '
                    'checkpoint: pad-aware absolute positions, learned bidirectional relative-position '
                    'bias, exact GELU, and tied vocabulary decoder.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'mpt': {
        'reference': {
            'config_class': 'transformers:MptConfig',
            'model_class': 'transformers:MptForCausalLM',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/mpt/configuration_mpt.py',
                'description': ('Pinned public configuration example directly constructs MptConfig(). CausalLM forward '
                    'has no checkpoint example; linked mosaicml/mpt-7b config is inaccessible (404/401). '
                    'This explicitly evaluates constructor defaults, including use_cache=False, not a '
                    'reconstruction of the unavailable checkpoint. Official llm-foundry '
                    'pretrain/mpt-7b.yaml supplies training sizes but does not resolve inference caching.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 193},
        'workload': 'forward',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'mt5': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:MT5Config',
            'model_class': 'transformers:MT5ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/mt5-small',
                'revision': '73fb5dbe4756edadc8fbe8c769b0a109493acf7a',
                'description': ('Pinned conditional-generation example checkpoint. Native pinned configuration ties '
                    'embeddings despite the serialized false field.'),
            },
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139, 'encoder_prefix_token_ids': [], 'encoder_suffix_token_ids': [1],
            'content_token_id_min': 3,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'dimension_overrides': {
            'd_model': 256, 'd_kv': 64, 'num_heads': 3, 'd_ff': 512, 'num_layers': 2, 'num_decoder_layers': 2,
            'vocab_size': 1024,
        },
        'dimension_purpose': ('Two blocks in each stack; 64-wide heads, original attention-width ratio and 2x feed-forward '
            'width; gated GELU, relative positions and self/cross caches retained.'),
        'configuration_scope': 'reviewed_scaled_v1',
    },

    'musicflamingo': {
        'reference': {
            'config_class': 'transformers:MusicFlamingoConfig',
            'model_class': 'transformers:MusicFlamingoForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'nvidia/music-flamingo-2601-hf',
                'revision': '6b5be086d52f65a1e204cb0faf70bf54e2741ecd',
                'url': 'https://huggingface.co/nvidia/music-flamingo-2601-hf/blob/6b5be086d52f65a1e204cb0faf70bf54e2741ecd/config.json',
                'description': ('Pinned main example generates from audio and text with native cached generation over '
                    'Qwen2 and timestamp-rotated full audio encoding.'),
            },
            'generation_config': {
                'bos_token_id': 151668, 'eos_token_id': 151645, 'max_new_tokens': 2048,
                'pad_token_id': 151669, 'transformers_version': '5.6.0.dev0',
            },
        },
        'dimension_overrides': {
            'head_dim': 80,
            'audio_config': {
                'hidden_size': 80, 'intermediate_size': 320, 'num_attention_heads': 5, 'num_hidden_layers': 2,
                'max_source_positions': 16,
            },
            'text_config': {
                'hidden_size': 112, 'intermediate_size': 592, 'num_attention_heads': 7,
                'num_key_value_heads': 1, 'num_hidden_layers': 2,
                'layer_types': ['full_attention', 'full_attention'], 'max_position_embeddings': 256,
                'max_window_layers': 2,
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'image_batch_size': 2, 'sequence_length': 21,
            'shape': [128, 32], 'image_input_name': 'input_features',
            'audio_token_positions': list(range(2, 18)), 'attention_mask': True, 'pad_token_id': 151669,
            'input_features_lengths': [32, 32],
        },
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 3},
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Two32frame audio windows yield16tokens in one continuous audio span, exercising nonzero time '
            'and window rotations; preserve mel bins, vocabulary,GQA7:1,native cached generation despite '
            'text config use_cacheFalse,40percent total rotated audio width.'),
    },

    'musicgen': {
        'reference': {
            'config_class': 'transformers:MusicgenConfig',
            'model_class': 'transformers:MusicgenForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/musicgen-small',
                'revision': '4c8334b02c6ec4e8664a91979669a501ec497792',
                'url': 'https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/config.json',
                'description': ('Pinned HF public forward example loads this checkpoint and calls text input plus '
                    'explicit decoder BOS ids for audio-token logits. The audio codec does not execute. '
                    'Longer development decoder input retains the same forward task; generation is outside '
                    'this declared scope.'),
            },
        },
        'input': {'kind': 'musicgen_forward', 'batch_size': 1, 'sequence_length': 193, 'decoder_sequence_length': 139},
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
    },

    'musicgen_melody': {
        'reference': {
            'config_class': 'transformers:MusicgenMelodyConfig',
            'model_class': 'transformers:MusicgenMelodyForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/musicgen-melody',
                'revision': '68d653a95788ec0d2b0abccab22c0b3a200c2d90',
                'url': 'https://huggingface.co/facebook/musicgen-melody/blob/68d653a95788ec0d2b0abccab22c0b3a200c2d90/config.json',
                'description': ('Pinned HF public forward example loads this checkpoint and calls text input plus '
                    'explicit decoder BOS ids for audio-token logits. The audio codec does not execute. '
                    'Longer development decoder input retains the same forward task; generation is outside '
                    'this declared scope.'),
            },
            'reuse_encoder': False,
            'continuation_outputs': ['logits', 'past_key_values'],
        },
        'input': {'kind': 'musicgen_forward', 'batch_size': 1, 'sequence_length': 193, 'decoder_sequence_length': 139},
        'outputs': ['logits', 'encoder_hidden_states', 'past_key_values'],
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
    },

    'mvp': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:MvpConfig',
            'model_class': 'transformers:MvpForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'RUCAIBox/mvp',
                'revision': 'c1d9aeb879f3079101f716f1f6e7109fdd18b4e9',
                'description': ('Pinned public MvpForConditionalGeneration example loads RUCAIBox/mvp; retain default '
                    'ordinary conditional generation and cache creation.'),
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'nanochat': {
        'reference': {
            'native_cache_defaults': True,
            'native_position_ids': True,
            'config_class': 'transformers:NanoChatConfig',
            'model_class': 'transformers:NanoChatForCausalLM',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/nanochat/configuration_nanochat.py',
                'description': ('Documented karpathy/nanochat-d32 supplies original author meta_000650.json but no '
                    'Transformers config.json. Map its declared dimensions via config_overrides; all other '
                    'computational settings are explicitly pinned HF constructor defaults, not proven '
                    'checkpoint-equivalent.'),
                'author_revision': '016dba034c9c0ca9033ad1bc721bceff54680600',
                'author_url': 'https://huggingface.co/karpathy/nanochat-d32/blob/016dba034c9c0ca9033ad1bc721bceff54680600/meta_000650.json',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
        'config_overrides': {
            'hidden_size': 2048, 'num_attention_heads': 16, 'num_key_value_heads': 16,
            'num_hidden_layers': 32, 'max_position_embeddings': 2048, 'vocab_size': 65536,
        },
    },

    'nemotron': {
        'reference': {
            'config_class': 'transformers:NemotronConfig',
            'model_class': 'transformers:NemotronForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'nvidia/Minitron-4B-Base',
                'revision': '3478fb8d5e0a4347e4ab08727f9f95f33f97df3b',
                'url': 'https://huggingface.co/nvidia/Minitron-4B-Base/blob/3478fb8d5e0a4347e4ab08727f9f95f33f97df3b/config.json',
                'description': 'Pinned HF nemotron public task docs or configuration checkpoint; retain actual computational settings.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'nemotron_h': {
        'reference': {
            'config_class': 'transformers:NemotronHConfig',
            'model_class': 'transformers:NemotronHForCausalLM',
            'source': {
                'checkpoint': 'nvidia/Nemotron-H-8B-Reasoning-128K',
                'revision': '2dcbcfd95b103843b6ad8e79690f34480ce5a5ae',
                'url': 'https://huggingface.co/nvidia/Nemotron-H-8B-Reasoning-128K/blob/2dcbcfd95b103843b6ad8e79690f34480ce5a5ae/config.json',
                'kind': 'example_checkpoint',
                'description': ('Pinned task example Zyphra/NemotronH-7B-v1 returns404. Usage docs name '
                    'nvidia/Nemotron-H-8B-Reasoning-128K; this outranks the configuration decorator naming '
                    'a different 30B MoE example. Preserve source Mamba/dense ReLU2/NoPE attention prefix '
                    'and all active transitions; no MoE is active in this documented8B default.'),
            },
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 142},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'nomic_bert': {
        'reference': {
            'config_class': 'transformers:NomicBertConfig',
            'model_class': 'transformers:NomicBertForMaskedLM',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('Retain existing corpus masked-LM task; its pinned class has no dedicated checkpoint '
                    'forward example. Use constructor computation defaults, whose configuration '
                    'documentation constructs the model from defaults.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/nomic_bert/configuration_nomic_bert.py',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'nystromformer': {
        'reference': {
            'config_class': 'transformers:NystromformerConfig',
            'model_class': 'transformers:NystromformerForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'uw-madison/nystromformer-512',
                'revision': '405ccb83538dd58b90104d20af1a08d901c103cd',
                'url': 'https://huggingface.co/uw-madison/nystromformer-512/blob/405ccb83538dd58b90104d20af1a08d901c103cd/config.json',
                'description': ('Published checkpoint preserves num_landmarks=segment_means_seq_len=64, selecting dense'
                    ' attention plus convolution. This case does not exercise landmark approximation or '
                    'iterative inversion.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 510},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'olmo': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:OlmoConfig',
            'model_class': 'transformers:OlmoForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'allenai/OLMo-7B-hf',
                'revision': '11fb3186a2e4f681edea621fa8b4345147a9db6a',
                'url': 'https://huggingface.co/allenai/OLMo-7B-hf/resolve/11fb3186a2e4f681edea621fa8b4345147a9db6a/config.json',
                'description': ('The pinned causal-LM example identifier returned HTTP404; use the accessible '
                    'checkpoint named by the pinned configuration-class documentation.'),
                'causal_lm_example_evidence': {
                    'checkpoint': 'meta-olmo/Olmo-2-7b-hf', 'accessible': False,
                    'error_type': 'RepositoryNotFoundError', 'status_code': 404,
                },
                'source_locations': [
                    'src/transformers/models/olmo/configuration_olmo.py:OlmoConfig',
                    'src/transformers/models/olmo/modeling_olmo.py:OlmoForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'olmo2': {
        'reference': {
            'config_class': 'transformers:Olmo2Config',
            'model_class': 'transformers:Olmo2ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'allenai/OLMo-2-0425-1B',
                'revision': 'a1847dff35000b4271fa70afc5db10fd29fedbdf',
                'url': 'https://huggingface.co/allenai/OLMo-2-0425-1B/blob/a1847dff35000b4271fa70afc5db10fd29fedbdf/config.json',
                'description': 'Pinned HF model documentation causal-LM example checkpoint; preserve its computational settings.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'olmo3': {
        'reference': {
            'config_class': 'transformers:Olmo3Config',
            'model_class': 'transformers:Olmo3ForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'allenai/Olmo-3-7B-Instruct',
                'revision': '6e5971d9eba42665f5bd5a0fcf047f299ce1dccc',
                'url': 'https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc/config.json',
                'description': ('Pinned docs contain allenai/TBA placeholder; use the official author '
                    'Olmo-3-7B-Instruct checkpoint, retaining YaRN and disabled cache.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 4097},
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'olmo_hybrid': {
        'reference': {
            'config_class': 'transformers:OlmoHybridConfig',
            'model_class': 'transformers:OlmoHybridForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'allenai/Olmo-Hybrid-7B',
                'revision': '4f1cc566f9fdf3ce68da2ab6a788a83d89896dcf',
                'url': 'https://huggingface.co/allenai/Olmo-Hybrid-7B/blob/4f1cc566f9fdf3ce68da2ab6a788a83d89896dcf/config.json',
                'description': 'Pinned HF public task example checkpoint.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 384, 'intermediate_size': 1100, 'num_hidden_layers': 4, 'num_attention_heads': 3,
            'num_key_value_heads': 3, 'linear_num_key_heads': 3, 'linear_num_value_heads': 3,
            'vocab_size': 1024, 'pad_token_id': 0,
            'layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
            'eos_token_id': 2,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 130},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Three linear blocks plus NoPE fullattention; native key96/value192, head128, conv4 and '
            'negative-eigenvalue beta scaling remain. Nativepadding remapped0, EOS2 in reducedvocab.'),
    },

    'olmoe': {
        'reference': {
            'config_class': 'transformers:OlmoeConfig',
            'model_class': 'transformers:OlmoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned docs/source/en/model_doc/olmoe.md:63 loads this causal LM with SDPA. Retain 64 '
                    'experts, top-eight routing without renormalization, full-projection Q/K RMSNorm, and '
                    'the checkpoint intermediate/hidden ratio of one-half.'),
                'checkpoint': 'allenai/OLMoE-1B-7B-0924',
                'revision': '6d84c48581ece794365f2b8e9cfb043c68ade9c5',
                'url': 'https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/6d84c48581ece794365f2b8e9cfb043c68ade9c5/config.json',
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'omdet_turbo': {
        'reference': {
            'config_class': 'transformers:OmDetTurboConfig',
            'model_class': 'transformers:OmDetTurboForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'omlab/omdet-turbo-swin-tiny-hf',
                'revision': '7fe93cecfb770c4d76cf71163956221249cab566',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/omdet_turbo/modeling_omdet_turbo.py',
                'description': 'Pinned public-task example checkpoint; native Swin/CLIP default detector.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 640]},
        'outputs': [
            'decoder_coord_logits', 'decoder_class_logits', 'init_reference_points',
            'intermediate_reference_points', 'encoder_coord_logits', 'encoder_class_logits',
            'encoder_extracted_states', 'classes_structure',
        ],
        'workload': 'forward',
        'reference_backend': None,
        'allow_infinite_outputs': ['init_reference_points'],
    },

    'oneformer': {
        'reference': {
            'config_class': 'transformers:OneFormerConfig',
            'model_class': 'transformers:OneFormerForUniversalSegmentation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'shi-labs/oneformer_ade20k_swin_tiny',
                'revision': '05f2812b1eccf9909b3897777450f8d68148cafc',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/oneformer/modeling_oneformer.py',
                'description': ('First public-task example semantic segmentation; exact 77-token processor prompt the '
                    'task is semantic.'),
            },
        },
        'input': {
            'kind': 'image',
            'batch_size': 1,
            'shape': [3, 512, 682],
            'task_inputs': [
                [
                    49406, 518, 10549, 533, 29119, 1550, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ],
        },
        'outputs': [
            'class_queries_logits', 'masks_queries_logits', 'auxiliary_predictions', 'encoder_hidden_states',
            'pixel_decoder_hidden_states', 'transformer_decoder_hidden_states',
            'transformer_decoder_object_queries', 'transformer_decoder_contrastive_queries',
            'transformer_decoder_mask_predictions', 'transformer_decoder_class_predictions',
            'transformer_decoder_auxiliary_predictions', 'task_token', 'attentions',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'openai': {
        'reference': {
            'config_class': 'transformers:OpenAIGPTConfig',
            'model_class': 'transformers:OpenAIGPTLMHeadModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'openai-community/openai-gpt',
                'revision': '1e0d4f3028acbffb47fe933cea64619c5ec1a002',
                'description': 'Pinned HF configuration checkpoint; original GPT causal LM task without KV caching.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'reference_backend': None,
        'workload': 'forward',
        'outputs': ['logits'],
    },

    'openai_privacy_filter': {
        'reference': {
            'config_class': 'transformers:OpenAIPrivacyFilterConfig',
            'model_class': 'transformers:OpenAIPrivacyFilterModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openai/privacy-filter',
                'revision': '7ffa9a043d54d1be65afb281eddf0ffbe629385b',
                'url': 'https://huggingface.co/openai/privacy-filter/resolve/7ffa9a043d54d1be65afb281eddf0ffbe629385b/config.json',
                'description': ('Pinned HF public model task checkpoint; retain FP32 expert arithmetic, bidirectional '
                    'window128, FP32 sinks,128experts/top4 and YaRN.'),
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 128, 'num_hidden_layers': 2, 'num_attention_heads': 7,
            'num_key_value_heads': 1, 'head_dim': 64, 'vocab_size': 1024, 'pad_token_id': 1023,
            'eos_token_id': 1023,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 270},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': 'eager',
        'dimension_purpose': ('Preserve native7:1GQA,128experts/top4, attention window128 and shared pad/EOS '
            'identity;270tokens exercise both sides of bidirectional window.'),
    },

    'opt': {
        'reference': {
            'config_class': 'transformers:OPTConfig',
            'model_class': 'transformers:OPTForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/opt-350m',
                'revision': '08ab08cc4b72ff5593870b5d527cf4230323703c',
                'url': 'https://huggingface.co/facebook/opt-350m/blob/08ab08cc4b72ff5593870b5d527cf4230323703c/config.json',
                'description': ('Pinned OPTForCausalLM.forward example and docs/source/en/model_doc/opt.md load '
                    'facebook/opt-350m. Preserve post-layer normalization, unequal embedding width with '
                    'both projections, ReLU, learned positions and default caching.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'ovis2': {
        'reference': {
            'config_class': 'transformers:Ovis2Config',
            'model_class': 'transformers:Ovis2ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'thisisiron/Ovis2-2B-hf',
                'revision': '4590b7c4392230c4575f153135590393f49dd174',
                'url': 'https://huggingface.co/thisisiron/Ovis2-2B-hf/blob/4590b7c4392230c4575f153135590393f49dd174/config.json',
                'description': 'Pinned HF public conditional-generation example Ovis2-2B; configuration doc names1B, task example takes precedence.',
            },
            'prefill_input_names': ['pixel_values'],
        },
        'dimension_overrides': {
            'hidden_size': 768,
            'vocab_size': 1024,
            'image_token_id': 900,
            'visual_indicator_token_ids': [901, 902, 903, 904, 905],
            'text_config': {
                'hidden_size': 768, 'intermediate_size': 1536, 'num_hidden_layers': 2,
                'num_attention_heads': 6, 'num_key_value_heads': 1, 'vocab_size': 1024,
                'max_position_embeddings': 1024, 'layer_types': ['full_attention', 'full_attention'],
            },
            'vision_config': {
                'hidden_size': 256, 'intermediate_size': 704, 'num_hidden_layers': 2,
                'num_attention_heads': 2, 'image_size': 56, 'vocab_size': 256,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 56, 56],
            'sequence_length': 16,
            'input_ids': [[1, 901, 900, 900, 900, 900, 902, 903, 904, 905, 2, 3, 4, 5, 6, 7]],
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'decode_outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('AIMv2-styleRMSNorm/SiLUvisionhead128,biasedpatch14embedding,nativehiddenstride2 '
            'packs4x4patches into4softvisualtokens;251visualcategories+5indicatorrows all5markers '
            'exercised;6:1Qwen2GQA/head128,tiedtexthead,cacheddecode.'),
    },

    'owlv2': {
        'reference': {
            'config_class': 'transformers:Owlv2Config',
            'model_class': 'transformers:Owlv2ForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/owlv2-base-patch16-ensemble',
                'revision': 'cfd3195ba4ea9592eec887ded089f4c08eff231d',
                'url': 'https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/cfd3195ba4ea9592eec887ded089f4c08eff231d/config.json',
                'description': ('Pinned HF public object-detection example checkpoint. Preserve query-conditioned '
                    'classification, patch/class-token multiplication, box coordinate bias, default '
                    'complete output, and OWLv2 objectness.'),
            },
        },
        'input': {
            'kind': 'text_image',
            'image_batch_size': 1,
            'text_batch_size': 2,
            'shape': [3, 960, 960],
            'sequence_length': 16,
            'eos_positions': [6, 6],
            'batch_size': 1,
            'attention_mask': True,
            'bos_token_id': 49406,
            'eos_token_id': 49407,
            'pad_token_id': 0,
            'input_ids': [
                [49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [49406, 320, 1125, 539, 320, 1929, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        },
        'workload': 'forward',
        'outputs': [
            'logits', 'pred_boxes', 'image_embeds', 'text_embeds', 'class_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output', 'objectness_logits',
        ],
        'reference_backend': None,
    },

    'owlvit': {
        'reference': {
            'config_class': 'transformers:OwlViTConfig',
            'model_class': 'transformers:OwlViTForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/owlvit-base-patch32',
                'revision': 'cbc355fb364588351c5d51c7f74465e8e7ec6f72',
                'url': 'https://huggingface.co/google/owlvit-base-patch32/blob/cbc355fb364588351c5d51c7f74465e8e7ec6f72/config.json',
                'description': ('Pinned HF public object-detection example checkpoint. Preserve query-conditioned '
                    'classification, patch/class-token multiplication, box coordinate bias, default '
                    'complete output, and OWLv2 objectness.'),
            },
        },
        'input': {
            'kind': 'text_image',
            'image_batch_size': 1,
            'text_batch_size': 2,
            'shape': [3, 768, 768],
            'sequence_length': 16,
            'eos_positions': [6, 6],
            'batch_size': 1,
            'attention_mask': True,
            'bos_token_id': 49406,
            'eos_token_id': 49407,
            'pad_token_id': 0,
            'input_ids': [
                [49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [49406, 320, 1125, 539, 320, 1929, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        },
        'workload': 'forward',
        'outputs': [
            'logits', 'pred_boxes', 'image_embeds', 'text_embeds', 'class_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'paddleocr_vl': {
        'reference': {
            'config_class': 'transformers:PaddleOCRVLConfig',
            'model_class': 'transformers:PaddleOCRVLForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'PaddlePaddle/PaddleOCR-VL',
                'revision': '7fa00a8c55b735ba51ba49a9058f3f9c57a99a11',
                'description': ('Pinned conditional-generation task; matching documented checkpoint config. GLM46V '
                    'explicitly documents 4.1V components; OCR forward example conflicts with its matching '
                    'config example.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/paddleocr_vl/modeling_paddleocr_vl.py',
            },
            'input_names': ['input_ids', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids'],
        },
        'dimension_overrides': {
            'hidden_size': 512,
            'intermediate_size': 1024,
            'num_hidden_layers': 2,
            'num_attention_heads': 8,
            'num_key_value_heads': 1,
            'vision_config': {
                'hidden_size': 144, 'num_attention_heads': 2, 'intermediate_size': 288,
                'num_hidden_layers': 2, 'image_size': 56,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [24, 3, 14, 14],
            'flatten_pixel_batch': True, 'image_grid_thw': [[1, 6, 4]], 'video_grid_thw': [],
            'sequence_length': 15, 'image_token_positions': [3, 4, 5, 6, 7, 8], 'mm_token_type_ids': True,
        },
        'workload': 'forward',
        'outputs': ['logits', 'rope_deltas'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve official flat-to-nested config migration, use_cache=False whole forward, head128 with'
            ' projected width1024/residual512 and8:1GQA. Vision nativehead72, tanhGELU, bilinear position '
            'resize4x4to6x4, merge2 projector with exactGELU. Full vocabulary/special IDs.'),
    },

    'paligemma': {'reference': {'config_class': 'transformers:PaliGemmaConfig',
                   'model_class': 'transformers:PaliGemmaForConditionalGeneration',
                   'source': {'kind': 'constructor_defaults',
                              'description': 'Pinned PaliGemmaConfig() bare defaults select coherent '
                                             'Gemma-v1 and unpooled SigLIP. This explicitly differs '
                                             'from the unusable stock-component doc example and gated '
                                             'PaliGemma2 task checkpoint.',
                              'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                              'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/paligemma/configuration_paligemma.py'},
                   'native_position_ids': True,
                   'native_cache_defaults': True,
                   'prefill_sequence_input_names': ['token_type_ids'],
                   'continuation_outputs': ['logits', 'past_key_values']},
     'config_overrides': {'vision_config': {'patch_size': 14,
                                            'vision_use_head': False,
                                            'vocab_size': 257152}},
     'dimension_overrides': {'projection_dim': 512,
                             'hidden_size': 512,
                             'vocab_size': 1024,
                             'vision_config': {'hidden_size': 144,
                                               'intermediate_size': 512,
                                               'num_attention_heads': 2,
                                               'num_hidden_layers': 2},
                             'text_config': {'hidden_size': 512,
                                             'intermediate_size': 4096,
                                             'num_attention_heads': 8,
                                             'num_key_value_heads': 1,
                                             'head_dim': 64,
                                             'num_hidden_layers': 2,
                                             'vocab_size': 1024}},
     'dimension_purpose': 'Retain native 224-pixel images, 14-pixel patches and 256 image tokens. Two repeated '
                          'vision/text blocks; vision width 144 with 2 heads preserves head width 72 and the 32:9 feed-forward ratio; '
                          'text widths 512/4096 preserve the 8:1 feed-forward ratio and 8:1 query/KV grouping, reducing ordinary '
                          'rotary head width from 256 to 64 for bounded validation. Vocabulary 1024 retains native '
                          'out-of-vocabulary image ID 256000 and corresponding embedding replacement. Explicit nested '
                          'vision fields restate bare constructor defaults because partial child '
                          'dictionaries otherwise select stock SigLIP defaults.',
     'input': {'kind': 'text_image',
               'shape': [3, 224, 224],
               'text_batch_size': 1,
               'image_batch_size': 1,
               'batch_size': 1,
               'sequence_length': 514,
               'image_token_positions': list(range(256)),
               'token_type_id': 0},
     'workload': 'causal_lm_continuation',
     'reference_backend': None,
     'outputs': ['logits', 'image_hidden_states', 'past_key_values']},

    'parakeet': {
        'reference': {
            'config_class': 'transformers:ParakeetCTCConfig',
            'model_class': 'transformers:ParakeetForCTC',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'nvidia/parakeet-ctc-1.1b',
                'revision': '20e63a0fed6aedba145b74b826dbd41df0941730',
                'description': 'Pinned official CTC model; retain all three subsampling stages and native processor80mel features.',
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'spectrogram', 'name': 'input_features', 'batch_size': 1, 'shape': [401, 80],
            'attention_mask_length': 401,
        },
        'workload': 'forward',
        'outputs': ['logits'],
    },

    'patchtsmixer': {
        'reference': {
            'config_class': 'transformers:PatchTSMixerConfig',
            'model_class': 'transformers:PatchTSMixerModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'ibm/patchtsmixer-etth1-pretrain',
                'revision': '05513838aadc1ab6b4fa2a851a29c6aceff354b7',
                'description': 'Pinned public base-model example checkpoint; optional training/forecast heads are outside original base-task scope.',
            },
        },
        'input': {'kind': 'continuous', 'name': 'past_values', 'shape': [512, 7], 'batch_size': 2},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'patch_input', 'loc', 'scale'],
        'reference_backend': None,
    },

    'patchtst': {
        'reference': {
            'config_class': 'transformers:PatchTSTConfig',
            'model_class': 'transformers:PatchTSTModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'namctin/patchtst_etth1_pretrain',
                'revision': '5dcaab0b40fa2b8a3843e2197e1f6b6e53ecd363',
                'description': 'Pinned public base-model example checkpoint; optional training/forecast heads are outside original base-task scope.',
            },
        },
        'input': {'kind': 'continuous', 'name': 'past_values', 'shape': [512, 7], 'batch_size': 2},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'patch_input', 'loc', 'scale'],
        'reference_backend': None,
    },

    'pe_audio': {
        'reference': {
            'config_class': 'transformers:PeAudioConfig',
            'model_class': 'transformers:PeAudioModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/pe-av-large',
                'revision': '0d24878d4107d64bef49e53602fc34ce6f94f6d8',
                'description': ('Pinned author model-card full multimodal inference and corresponding modality-text '
                    'subset. Native parent property preserves shared text config.'),
                'config_class': 'transformers:PeAudioVideoConfig',
                'config_property': 'audio_config',
            },
        },
        'input': {
            'kind': 'external',
            'required_keys': ['input_ids', 'input_values', 'attention_mask', 'padding_mask'], 'batch_size': 1,
        },
        'outputs': ['logits_audio_text', 'text_audio_embeds', 'audio_embeds', 'text_outputs', 'audio_outputs'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pe_audio_video': {
        'reference': {
            'config_class': 'transformers:PeAudioVideoConfig',
            'model_class': 'transformers:PeAudioVideoModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/pe-av-large',
                'revision': '0d24878d4107d64bef49e53602fc34ce6f94f6d8',
                'description': ('Pinned author model-card full multimodal inference and corresponding modality-text '
                    'subset. Native parent property preserves shared text config.'),
            },
        },
        'input': {
            'kind': 'external',
            'required_keys': ['input_ids', 'input_values', 'pixel_values_videos', 'attention_mask', 'padding_mask', 'padding_mask_videos'],
            'batch_size': 1,
        },
        'outputs': [
            'audio_embeds', 'video_embeds', 'audio_video_embeds', 'text_audio_embeds', 'text_video_embeds',
            'text_audio_video_embeds', 'audio_plus_text_embeds', 'video_plus_text_embeds', 'text_outputs',
            'audio_outputs', 'video_outputs', 'audio_video_outputs', 'logits_audio_text', 'logits_video_text',
            'logits_audio_video', 'logits_audio_video_text', 'logits_audio_plus_text_video',
            'logits_video_plus_text_audio',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pe_video': {
        'reference': {
            'config_class': 'transformers:PeVideoConfig',
            'model_class': 'transformers:PeVideoModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/pe-av-large',
                'revision': '0d24878d4107d64bef49e53602fc34ce6f94f6d8',
                'description': ('Pinned author model-card full multimodal inference and corresponding modality-text '
                    'subset. Native parent property preserves shared text config.'),
                'config_class': 'transformers:PeAudioVideoConfig',
                'config_property': 'video_config',
            },
        },
        'input': {
            'kind': 'external',
            'required_keys': ['input_ids', 'pixel_values_videos', 'attention_mask', 'padding_mask_videos'],
            'batch_size': 1,
        },
        'outputs': ['logits_video_text', 'text_video_embeds', 'video_embeds', 'text_outputs', 'video_outputs'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pegasus': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:PegasusConfig',
            'model_class': 'transformers:PegasusForConditionalGeneration',
            'forward_kwargs': {},
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'google/pegasus-xsum',
                'revision': '8d8ffc158a3bee9fbb03afacdfc347c823c5ec8b',
                'description': 'Pinned conditional-generation summarization example checkpoint.',
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139, 'encoder_prefix_token_ids': [], 'encoder_suffix_token_ids': [1],
            'content_token_id_min': 105,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'pegasus_x': {
        'reference': {
            'config_class': 'transformers:PegasusXConfig',
            'model_class': 'transformers:PegasusXForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/pegasus-x-large',
                'revision': 'ff38c5db1f5b97b923dbecdaf58b15c8e213a27a',
                'description': ('Pinned public conditional-generation example; checkpoint ReLU and global/local block '
                    'attention preserved rather than historical artifact GELU override.'),
            },
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 1153,
            'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'perceiver': {
        'reference': {
            'config_class': 'transformers:PerceiverConfig',
            'model_class': 'transformers:PerceiverModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('Pinned PerceiverConfig documentation explicitly constructs PerceiverConfig() then '
                    'PerceiverModel(configuration). Optional input preprocessor/decoder/postprocessor '
                    'remain None; scope is latent encoding of continuous input, not a downstream '
                    'language/image task.'),
            },
        },
        'input': {'kind': 'continuous', 'name': 'inputs', 'shape': [513, 768], 'batch_size': 1},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'perception_lm': {
        'reference': {
            'config_class': 'transformers:PerceptionLMConfig',
            'model_class': 'transformers:PerceptionLMForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'facebook/Perception-LM-1B',
                'revision': '2b1a854663b80d6c8b9a10e4b229be97c7f6be1f',
                'description': 'Pinned native public image-conditioned ordinary text generation; PerceptionLM also exercises video inputs.',
            },
            'generation_config': {'bos_token_id': 128000, 'eos_token_id': [128001, 128009], 'transformers_version': '4.54.0.dev0'},
        },
        'dimension_overrides': {
            'text_config': {
                'hidden_size': 256, 'intermediate_size': 512, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 1,
            },
            'vision_config': {
                'model_args': {
                    'depth': 2, 'embed_dim': 64, 'global_pool': '', 'img_size': [56, 56], 'init_values': 0.1,
                    'ref_feat_shape': [32, 32], 'use_post_transformer_norm': False, 'num_heads': 4,
                },
                'num_features': 64,
            },
        },
        'input': {
            'kind': 'external_prepared',
            'description': 'Explicit common BF16 pixel tensors plus token IDs; full vocabulary and native image placeholder layout.',
        },
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 4, 'do_sample': False},
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'reference_backend': {'text_config': 'sdpa', 'vision_config': 'eager'},
        'dimension_purpose': ('Shrink depth and widths while retaining native head widths, full vocabulary, image encoding '
            'branches, all modal inputs, native pooling and connector ratios. Four ordinary autoregressive '
            'steps.'),
    },

    'persimmon': {
        'reference': {
            'config_class': 'transformers:PersimmonConfig',
            'model_class': 'transformers:PersimmonForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'adept/persimmon-8b-base',
                'revision': '94dc4e0bb7eeb26ec521eb3f78c36c91f6fe866b',
                'url': 'https://huggingface.co/adept/persimmon-8b-base/blob/94dc4e0bb7eeb26ec521eb3f78c36c91f6fe866b/config.json',
                'description': 'Pinned HF persimmon public task docs or configuration checkpoint; retain actual computational settings.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'phi': {
        'reference': {
            'config_class': 'transformers:PhiConfig',
            'model_class': 'transformers:PhiForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/phi-1',
                'revision': 'd4c0adcb065e84e00ca814e35cba3012ea9841ab',
                'url': 'https://huggingface.co/microsoft/phi-1/blob/d4c0adcb065e84e00ca814e35cba3012ea9841ab/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/phi.md selects this causal-LM checkpoint; preserve '
                    'its activation, residual order, partial rotary, biases and weight tying.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'phi3': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:Phi3Config',
            'model_class': 'transformers:Phi3ForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/Phi-3-mini-4k-instruct',
                'revision': 'f39ac1d28e925b323eae81227eaba4464caced4e',
                'description': 'Pinned configuration documentation checkpoint; causal-LM example identifier is unavailable.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 2050},
        'outputs': ['logits', 'past_key_values'],
    },

    'phi4_multimodal': {
        'variants': {
            'vision': {
                'reference': {
                    'config_class': 'transformers:Phi4MultimodalConfig',
                    'model_class': 'transformers:Phi4MultimodalForCausalLM',
                    'source': {
                        'kind': 'example_checkpoint', 'checkpoint': 'microsoft/Phi-4-multimodal-instruct',
                        'revision': '93f923e1a7727d1c4f446756212d9d3e8fcc5d81',
                        'description': 'Author README literal vision generation example; official pinned config conversion and active random vision LoRA at the original rank/scaling.',
                        'url': 'https://huggingface.co/microsoft/Phi-4-multimodal-instruct',
                        'config_converter': 'transformers.models.phi4_multimodal.convert_phi4_multimodal_weights_to_hf:convert_config',
                    },
                    'adapter_config': {
                        'r': 256, 'lora_alpha': 512, 'lora_dropout': 0.0,
                        'target_modules': 'model.layers.\\d+.((self_attn.(qkv|o)_proj)|(mlp.(gate_up|down)_proj))',
                        'bias': 'none', 'task_type': 'CAUSAL_LM', 'use_dora': False, 'use_rslora': False,
                        'inference_mode': True,
                    },
                    'generation_config': {
                        '_from_model_config': True, 'bos_token_id': 199999, 'eos_token_id': [200020, 199999],
                        'pad_token_id': 199999, 'transformers_version': '4.46.1', 'use_cache': True,
                    },
                },
                'reference_backend': {'': 'sdpa', 'vision_config': 'sdpa', 'audio_config': 'sdpa'},
                'dimension_overrides': {
                    'hidden_size': 384,
                    'intermediate_size': 1024,
                    'num_hidden_layers': 1,
                    'num_attention_heads': 3,
                    'num_key_value_heads': 1,
                    'vision_config': {
                        'hidden_size': 144, 'intermediate_size': 538, 'num_hidden_layers': 2,
                        'num_attention_heads': 2, 'image_size': 448, 'crop_size': 448,
                    },
                    'audio_config': {
                        'hidden_size': 128, 'intermediate_size': 192, 'num_blocks': 1,
                        'num_attention_heads': 2, 'ext_pw_out_channel': 128,
                        'depthwise_separable_out_channel': 128, 'depthwise_seperable_out_channel': 128,
                        'nemo_conv_channels': 8,
                    },
                },
                'input': {'kind': 'supplied'},
                'workload': 'generate',
                'generation_kwargs': {
                    'max_new_tokens': 4, 'do_sample': False, 'use_cache': True,
                    'eos_token_id': [200020, 199999], 'pad_token_id': 199999,
                },
                'outputs': ['sequences', 'logits', 'past_key_values'],
                'dimension_purpose': (
                    'Hidden and feed-forward widths scale by eight, preserving attention head widths and '
                    'feed-forward ratios. Retain one text/audio block and two vision blocks for penultimate '
                    'features; keep text GQA3:1, partial LongRoPE, vocabulary, and active LoRA ranks/scales. '
                    'Audio convolution channels reduce to eight while retaining all three stride-two stages '
                    'and depthwise groups. Keep 448-pixel crops and author-example image geometry. Synthetic '
                    '4008-frame audio crosses the 500-frame encoder chunk boundary. Both modalities generate '
                    'four tokens and compare logits and final cache; vision source semantics remain under review.'),
            },
            'speech': {
                'reference': {
                    'config_class': 'transformers:Phi4MultimodalConfig',
                    'model_class': 'transformers:Phi4MultimodalForCausalLM',
                    'source': {
                        'kind': 'example_checkpoint', 'checkpoint': 'microsoft/Phi-4-multimodal-instruct',
                        'revision': '93f923e1a7727d1c4f446756212d9d3e8fcc5d81',
                        'description': 'Author README literal speech generation example; official pinned config conversion and active random speech LoRA at the original rank/scaling.',
                        'url': 'https://huggingface.co/microsoft/Phi-4-multimodal-instruct',
                        'config_converter': 'transformers.models.phi4_multimodal.convert_phi4_multimodal_weights_to_hf:convert_config',
                    },
                    'adapter_config': {
                        'r': 320, 'lora_alpha': 640, 'lora_dropout': 0.01,
                        'target_modules': 'model.layers.\\d+.((self_attn.(qkv|o)_proj)|(mlp.(gate_up|down)_proj))',
                        'bias': 'none', 'task_type': 'CAUSAL_LM', 'use_dora': False, 'use_rslora': False,
                        'inference_mode': True,
                    },
                    'generation_config': {
                        '_from_model_config': True, 'bos_token_id': 199999, 'eos_token_id': [200020, 199999],
                        'pad_token_id': 199999, 'transformers_version': '4.46.1', 'use_cache': True,
                    },
                },
                'reference_backend': {'': 'sdpa', 'vision_config': 'sdpa', 'audio_config': 'sdpa'},
                'dimension_overrides': {
                    'hidden_size': 384,
                    'intermediate_size': 1024,
                    'num_hidden_layers': 1,
                    'num_attention_heads': 3,
                    'num_key_value_heads': 1,
                    'vision_config': {
                        'hidden_size': 144, 'intermediate_size': 538, 'num_hidden_layers': 2,
                        'num_attention_heads': 2, 'image_size': 448, 'crop_size': 448,
                    },
                    'audio_config': {
                        'hidden_size': 128, 'intermediate_size': 192, 'num_blocks': 1,
                        'num_attention_heads': 2, 'ext_pw_out_channel': 128,
                        'depthwise_separable_out_channel': 128, 'depthwise_seperable_out_channel': 128,
                        'nemo_conv_channels': 8,
                    },
                },
                'input': {'kind': 'supplied'},
                'workload': 'generate',
                'generation_kwargs': {
                    'max_new_tokens': 4, 'do_sample': False, 'use_cache': True,
                    'eos_token_id': [200020, 199999], 'pad_token_id': 199999,
                },
                'outputs': ['sequences', 'logits', 'past_key_values'],
                'dimension_purpose': (
                    'Hidden and feed-forward widths scale by eight, preserving attention head widths and '
                    'feed-forward ratios. Retain one text/audio block and two vision blocks for penultimate '
                    'features; keep text GQA3:1, partial LongRoPE, vocabulary, and active LoRA ranks/scales. '
                    'Audio convolution channels reduce to eight while retaining all three stride-two stages '
                    'and depthwise groups. Keep 448-pixel crops and author-example image geometry. Synthetic '
                    '4008-frame audio crosses the 500-frame encoder chunk boundary. Both modalities generate '
                    'four tokens and compare logits and final cache; vision source semantics remain under review.'),
            },
        },
    },

    'phimoe': {
        'reference': {
            'config_class': 'transformers:PhimoeConfig',
            'model_class': 'transformers:PhimoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/Phi-3.5-MoE-instruct',
                'revision': '43688451b462a3351d8580625ebe1931adb3986d',
                'description': ('Pinned PhimoeConfig example and architecture-author checkpoint. The public forward doc'
                    ' erroneously names mistralai/Phimoe-8x7B-v0.1; the matching config documentation names'
                    ' this Microsoft checkpoint.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/phimoe/configuration_phimoe.py#L24',
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 4099},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'pi0': {
        'reference': {
            'config_class': 'transformers:PI0Config',
            'model_class': 'transformers:PI0ForConditionalGeneration',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('Complete pinned HF PI0 defaults, corroborated by author LeRobot pi0_base config; '
                    'author config is not HF model format. Nested child dictionaries require explicitly '
                    'retaining PI0 layer/head/patch defaults, otherwise HF constructs generic Gemma/SigLIP '
                    'defaults.'),
            },
        },
        'dimension_overrides': {
            'vlm_config': {
                'text_config': {'hidden_size': 64, 'intermediate_size': 128, 'head_dim': 16, 'vocab_size': 1024},
                'vision_config': {'hidden_size': 64, 'intermediate_size': 128, 'num_attention_heads': 4, 'image_size': 56},
                'projection_dim': 64,
                'image_token_index': 1024,
            },
            'dit_config': {'hidden_size': 64, 'intermediate_size': 128, 'head_dim': 16, 'vocab_size': 1024},
        },
        'reference_backend': 'sdpa',
        'input': {'kind': 'pi0', 'sequence_length': 35, 'camera_mask': [[True, True, False]]},
        'workload': 'sample_actions',
        'outputs': ['actions'],
        'dimension_purpose': ('Preserve27SigLIP/18VLM/18expert '
            'layers,8:1GQA,3cameras(2valid),50actions,32state/actiondims,10Eulersteps; reduce '
            'image56patch14,width64,MLP128,head16,vocab1024. Explicit common native noise input.'),
        'config_overrides': {
            'vlm_config': {
                'text_config': {'num_hidden_layers': 18, 'num_attention_heads': 8, 'num_key_value_heads': 1},
                'vision_config': {'num_hidden_layers': 27, 'patch_size': 14, 'vision_use_head': False},
            },
            'dit_config': {'num_hidden_layers': 18, 'num_attention_heads': 8, 'num_key_value_heads': 1},
        },
    },

    'pix2struct': {
        'reference': {
            'config_class': 'transformers:Pix2StructConfig',
            'model_class': 'transformers:Pix2StructForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'google/pix2struct-textcaps-base',
                'revision': '61bee0d7e2378e601b68f853ceee4f7cf99f1b88',
                'url': 'https://huggingface.co/google/pix2struct-textcaps-base/blob/61bee0d7e2378e601b68f853ceee4f7cf99f1b88/config.json',
                'description': 'Pinned Pix2Struct image-captioning example checkpoint; this case evaluates its complete forward call.',
            },
        },
        'input': {
            'kind': 'image', 'name': 'flattened_patches', 'batch_size': 1, 'shape': [2048, 770],
            'patch_grid_columns': 64, 'decoder_sequence_length': 137, 'attention_mask_length': 2048,
        },
        'workload': 'forward',
        'outputs': ['logits', 'encoder_last_hidden_state'],
        'reference_backend': None,
    },

    'pixio': {
        'reference': {
            'config_class': 'transformers.models.pixio.configuration_pixio:PixioConfig',
            'model_class': 'transformers.models.pixio.modeling_pixio:PixioModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': 'The pinned public configuration example constructs PixioConfig() and PixioModel(config).',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/pixio/configuration_pixio.py#L43',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 256, 256]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'pixtral': {
        'reference': {
            'config_class': 'transformers:PixtralVisionConfig',
            'model_class': 'transformers:PixtralVisionModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'mistral-labs/pixtral-12b',
                'revision': 'c2756cbbb9422eba9f6c5c439a214b0392dfc998',
                'description': 'Pinned HF documented checkpoint; original audit public base-model task retained.',
                'config_key': 'vision_config',
            },
        },
        'dimension_overrides': {'hidden_size': 256, 'intermediate_size': 1024, 'num_attention_heads': 4, 'num_hidden_layers': 2},
        'input': {'kind': 'image', 'shape': [3, 96, 128], 'batch_size': 2},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': 'eager',
        'dimension_purpose': ('Keep head_dim64, image_size1024 positional-frequency domain and patch16; two nonsquare6x8 '
            'patch images exercise both spatial axes and inter-image block mask. Only width/layer/head '
            'counts reduced. Optional image_sizes omitted as native default.'),
    },

    'plbart': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:PLBartConfig',
            'model_class': 'transformers:PLBartForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'uclanlp/plbart-base',
                'revision': 'cf5287241fcff3819f6ade49635dc2d77efee032',
                'description': ('Pinned public PLBartForConditionalGeneration example loads uclanlp/plbart-base; retain'
                    ' default ordinary conditional generation and cache creation.'),
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139, 'decoder_start_token_id': 50003,
            'encoder_suffix_token_ids': [2, 50003],
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'poolformer': {
        'reference': {
            'config_class': 'transformers.models.poolformer.configuration_poolformer:PoolFormerConfig',
            'model_class': 'transformers.models.poolformer.modeling_poolformer:PoolFormerModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs PoolFormerConfig() and '
                    'PoolFormerModel(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/poolformer/configuration_poolformer.py#L42',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'pop2piano': {
        'reference': {
            'config_class': 'transformers:Pop2PianoConfig',
            'model_class': 'transformers:Pop2PianoForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'sweetcocoa/pop2piano',
                'revision': '142e8ed35614bcf77a3515b979e48ed528342349',
                'description': 'Published mel/composer-conditioned MIDI generation, including default composer1 and cached greedy decoding.',
            },
            'generation_config': {
                'decoder_start_token_id': 0,
                'eos_token_id': 1,
                'pad_token_id': 0,
                'return_dict_in_generate': False,
                'max_length': 256,
                'composer_to_feature_token': {
                    'composer1': 2052, 'composer2': 2053, 'composer3': 2054, 'composer4': 2055,
                    'composer5': 2056, 'composer6': 2057, 'composer7': 2058, 'composer8': 2059,
                    'composer9': 2060, 'composer10': 2061, 'composer11': 2062, 'composer12': 2063,
                    'composer13': 2064, 'composer14': 2065, 'composer15': 2066, 'composer16': 2067,
                    'composer17': 2068, 'composer18': 2069, 'composer19': 2070, 'composer20': 2071,
                    'composer21': 2072,
                },
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'spectrogram', 'name': 'input_features', 'batch_size': 1, 'shape': [137, 512],
            'attention_mask_length': 137,
        },
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 17},
        'outputs': ['sequences', 'logits', 'past_key_values'],
    },

    'pp_doclayout_v3': {
        'reference': {
            'config_class': 'transformers:PPDocLayoutV3Config',
            'model_class': 'transformers:PPDocLayoutV3ForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'PaddlePaddle/PP-DocLayoutV3_safetensors',
                'revision': '97d101e6db2642e162a1d05392d1b0231c91033e',
                'description': 'Pinned ordinary public object-detection task example.',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 800, 800]},
        'outputs': [
            'logits', 'pred_boxes', 'order_logits', 'out_masks', 'last_hidden_state',
            'intermediate_hidden_states', 'intermediate_logits', 'intermediate_reference_points',
            'encoder_last_hidden_state', 'init_reference_points', 'enc_topk_logits', 'enc_topk_bboxes',
            'enc_outputs_class', 'enc_outputs_coord_logits',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pp_formulanet': {
        'reference': {
            'config_class': 'transformers:PPFormulaNetConfig',
            'model_class': 'transformers:PPFormulaNetForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'PaddlePaddle/PP-FormulaNet_plus-L_safetensors',
                'revision': '6cc004a8e91ea3f3819b05317f40b290cbe99c2b',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/pp_formulanet.md',
                'description': 'Pinned native documented public task checkpoint; full task architecture retained.',
            },
            'generation_config': {
                'bos_token_id': 0, 'eos_token_id': 2, 'forced_eos_token_id': 2, 'decoder_start_token_id': 2,
                'max_length': 1537, 'pad_token_id': 1, 'use_cache': True,
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 768, 768]},
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'reference_backend': None,
        'generation_kwargs': {'max_new_tokens': 3},
    },

    'pp_lcnet': {
        'reference': {
            'config_class': 'transformers:PPLCNetConfig',
            'model_class': 'transformers:PPLCNetBackbone',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/pp_lcnet/modeling_pp_lcnet.py',
                'description': 'Pinned public backbone task example constructs the architecture using constructor defaults.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['feature_maps'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pp_lcnet_v3': {
        'reference': {
            'config_class': 'transformers:PPLCNetV3Config',
            'model_class': 'transformers:PPLCNetV3Backbone',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py',
                'description': 'Pinned public backbone task example constructs the architecture using constructor defaults.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['feature_maps'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pp_ocrv5_mobile_det': {
        'reference': {
            'config_class': 'transformers:PPOCRV5MobileDetConfig',
            'model_class': 'transformers:PPOCRV5MobileDetForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'PaddlePaddle/PP-OCRv5_mobile_det_safetensors',
                'revision': 'c5041d225cf951ff06900ab81a3c7d543c45e2ad',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/pp_ocrv5_mobile_det.md',
                'description': 'Pinned native documented public task checkpoint; full task architecture retained.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 960, 608]},
        'outputs': ['last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pp_ocrv5_mobile_rec': {
        'reference': {
            'config_class': 'transformers:PPOCRV5MobileRecConfig',
            'model_class': 'transformers:PPOCRV5MobileRecForTextRecognition',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'PaddlePaddle/PP-OCRv5_mobile_rec_safetensors',
                'revision': '485a9a9781535ad477de3b1eea5f96a2e7300d8d',
                'url': 'https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_safetensors/blob/485a9a9781535ad477de3b1eea5f96a2e7300d8d/config.json',
                'description': 'Pinned HF documented OCR task checkpoint and native processor48x320 dimensions.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 48, 320]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'pp_ocrv5_server_det': {
        'reference': {
            'config_class': 'transformers:PPOCRV5ServerDetConfig',
            'model_class': 'transformers:PPOCRV5ServerDetForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'PaddlePaddle/PP-OCRv5_server_det_safetensors',
                'revision': 'cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/pp_ocrv5_server_det.md',
                'description': 'Pinned native documented public task checkpoint; full task architecture retained.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 960, 608]},
        'outputs': ['last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'pp_ocrv5_server_rec': {
        'reference': {
            'config_class': 'transformers:PPOCRV5ServerRecConfig',
            'model_class': 'transformers:PPOCRV5ServerRecForTextRecognition',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'PaddlePaddle/PP-OCRv5_server_rec_safetensors',
                'revision': '542979d7cc3791732bb12af35313a6840952d79f',
                'url': 'https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors/blob/542979d7cc3791732bb12af35313a6840952d79f/config.json',
                'description': 'Pinned HF documented OCR task checkpoint and native processor48x320 dimensions.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 48, 320]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'prompt_depth_anything': {
        'reference': {
            'config_class': 'transformers:PromptDepthAnythingConfig',
            'model_class': 'transformers:PromptDepthAnythingForDepthEstimation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'depth-anything/prompt-depth-anything-vits-hf',
                'revision': '6f3768a5383d95a904ec00d2a7ca3bd186f8797e',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/prompt_depth_anything/modeling_prompt_depth_anything.py#L418',
                'description': 'Pinned public example supplies image and measured prompt depth.',
            },
        },
        'reference_backend': None,
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 560, 756], 'prompt_depth_shape': [1, 192, 256]},
        'outputs': ['predicted_depth'],
        'workload': 'forward',
    },

    'prophetnet': {
        'reference': {
            'config_class': 'transformers:ProphetNetConfig',
            'model_class': 'transformers:ProphetNetForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/prophetnet-large-uncased',
                'revision': 'c5b84da76e7f132c85b5b361e508145eaf2c24cd',
                'description': ('Pinned public conditional-generation example checkpoint; retain default two prediction'
                    ' streams, logits_ngram, encoder output and both cache types.'),
            },
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values', 'logits_ngram'],
        'reference_backend': None,
    },

    'pvt': {
        'reference': {
            'config_class': 'transformers.models.pvt.configuration_pvt:PvtConfig',
            'model_class': 'transformers.models.pvt.modeling_pvt:PvtModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned PvtConfig example explicitly constructs PvtModel(PvtConfig()). Retain that '
                    'public base-model task with all default stages and already computed outputs.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/pvt/configuration_pvt.py#L49',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'pvt_v2': {
        'reference': {
            'config_class': 'transformers.models.pvt_v2.configuration_pvt_v2:PvtV2Config',
            'model_class': 'transformers.models.pvt_v2.modeling_pvt_v2:PvtV2Model',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': 'The pinned public PvtV2Model example constructs PvtV2Config().',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/pvt_v2/configuration_pvt_v2.py#L44',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'qianfan_ocr': {
        'reference': {
            'config_class': 'transformers:QianfanOCRConfig',
            'model_class': 'transformers:QianfanOCRForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'baidu/Qianfan-OCR',
                'revision': '623bf5d20d446abdb36606aa4547cd0c18886fe5',
                'url': 'https://huggingface.co/baidu/Qianfan-OCR/blob/623bf5d20d446abdb36606aa4547cd0c18886fe5/config.json',
                'description': 'Pinned native HF OCR task example; preserve source text_config.use_cache=false and default final patch selection.',
            },
        },
        'dimension_overrides': {
            'image_token_id': 900,
            'vision_config': {
                'hidden_size': 128, 'intermediate_size': 512, 'num_attention_heads': 2,
                'num_hidden_layers': 2, 'image_size': 56,
            },
            'text_config': {
                'hidden_size': 320, 'intermediate_size': 1216, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 1, 'vocab_size': 1024,
                'max_position_embeddings': 1024,
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'shape': [3, 56, 56],
            'sequence_length': 16, 'image_token_positions': [3, 4, 5, 6],
        },
        'workload': 'forward',
        'outputs': ['logits', 'image_hidden_states'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Nativepatch14/head64vision,CLStoken,learnedpositions,LayerNorm+layerscales/GELU;factor.5pixelpacking+GELUprojector.'
            ' Qwen3head128/4:1GQA,attentionwidth1.6xhidden,QKnorm,sourcecachefalse preserved '
            'aswhole-sequenceforward.'),
    },

    'qwen2': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:Qwen2Config',
            'model_class': 'transformers:Qwen2ForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Qwen/Qwen2-7B',
                'revision': '453ed1575b739b5b03ce3758b23befdb0967f40e',
                'url': 'https://huggingface.co/Qwen/Qwen2-7B/resolve/453ed1575b739b5b03ce3758b23befdb0967f40e/config.json',
                'description': ('Pinned causal-LM example identifier meta-qwen2/Qwen2-2-7b-hf returned HTTP404; use the'
                    ' official checkpoint named by the pinned configuration-class documentation. Preserve '
                    'its computational settings.'),
                'invalid_causal_lm_example': 'meta-qwen2/Qwen2-2-7b-hf',
                'invalid_causal_lm_example_access': 'HTTP404 RepositoryNotFoundError',
                'source_locations': [
                    'src/transformers/models/qwen2/configuration_qwen2.py:Qwen2Config',
                    'src/transformers/models/qwen2/modeling_qwen2.py:Qwen2ForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 258},
        'outputs': ['logits', 'past_key_values'],
        'implementation_kwargs': {'native_precision': True},
        'dimension_overrides': {
            'hidden_size': 896, 'num_attention_heads': 7, 'num_key_value_heads': 1, 'num_hidden_layers': 2,
            'intermediate_size': 4736, 'vocab_size': 1024, 'bos_token_id': 1, 'eos_token_id': 1,
        },
        'dimension_purpose': ('Two decoder blocks; 128-wide heads, 7:1 grouped queries, original feed-forward ratio. BOS/EOS '
            'share ID 1 in the smaller vocabulary; supplied-token continuation, not generation.'),
        'configuration_scope': 'reviewed_scaled_v1',
    },

    'qwen2_5_omni': {
        'reference': {
            'config_class': 'transformers:Qwen2_5OmniConfig',
            'model_class': 'transformers:Qwen2_5OmniForConditionalGeneration',
            'load_with_base_class': True,
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'Qwen/Qwen2.5-Omni-7B',
                'revision': 'ae9e1690543ffd5c0221dc27f79834d0294cba00',
                'url': 'https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/ae9e1690543ffd5c0221dc27f79834d0294cba00/config.json',
                'description': 'Pinned HF public full-parent example, default text-plus-waveform output and Chelsie speaker.',
            },
            'speakers': {
                'repo': 'Qwen/Qwen2.5-Omni-7B', 'revision': 'ae9e1690543ffd5c0221dc27f79834d0294cba00',
                'filename': 'spk_dict.pt',
            },
            'generation_output_names': ['sequences', 'waveform'],
        },
        'dimension_overrides': {
            'thinker_config': {
                'text_config': {
                    'hidden_size': 896, 'intermediate_size': 4736, 'num_hidden_layers': 2,
                    'num_attention_heads': 7, 'num_key_value_heads': 1, 'max_position_embeddings': 2048,
                },
                'vision_config': {
                    'depth': 2, 'hidden_size': 128, 'embed_dim': 128, 'intermediate_size': 384,
                    'num_heads': 2, 'out_hidden_size': 896, 'fullatt_block_indexes': [1],
                },
                'audio_config': {
                    'd_model': 128, 'encoder_layers': 2, 'encoder_attention_heads': 2, 'encoder_ffn_dim': 512,
                    'output_dim': 896,
                },
            },
            'talker_config': {
                'hidden_size': 224, 'embedding_size': 896, 'intermediate_size': 4736, 'num_hidden_layers': 2,
                'num_attention_heads': 3, 'num_key_value_heads': 1, 'max_position_embeddings': 2048,
            },
            'token2wav_config': {
                'dit_config': {
                    'hidden_size': 64, 'num_attention_heads': 16, 'head_dim': 4, 'emb_dim': 32, 'enc_dim': 32,
                    'enc_channels': [16, 16, 16, 16, 48], 'enc_attention_channels': 16, 'enc_se_channels': 16,
                },
                'bigvgan_config': {'upsample_initial_channel': 64},
            },
        },
        'input': {
            'kind': 'text_image',
            'text_batch_size': 1,
            'sequence_length': 161,
            'image_batch_size': 1,
            'shape': [144, 1176],
            'video_shape': [32, 1176],
            'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 12, 12]],
            'video_grid_thw': [[2, 4, 4]],
            'video_second_per_grid': [2.0],
            'audio_shape': [128, 404],
            'feature_attention_lengths': [401],
            'input_ids': [
                [
                    151644, 872, 198, 10, 151647, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
                    151646, 151646, 151646, 151646, 151646, 151648, 11, 151652, 151655, 151655, 151655,
                    151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
                    151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
                    151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
                    151653, 12, 151652, 151656, 151656, 151656, 151656, 151656, 151656, 151656, 151656,
                    151653, 13, 151645, 151644, 77091, 198,
                ],
            ],
            'attention_mask': True,
            'pad_token_id': 151643,
            'dtypes': {'video_second_per_grid': 'float32'},
        },
        'workload': 'generate',
        'generation_kwargs': {'thinker_max_new_tokens': 4, 'talker_max_new_tokens': 26},
        'generation_seed': 0,
        'outputs': ['sequences', 'waveform'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Keep full-parent default audio generation: source vocabulary and speaker BOS; 7:1 thinker '
            'GQA/head128,3:1talker GQA/head128 and native attention-to-hidden expansion;2layers each, '
            'audio401frames crosses2n_window chunks, image12x12 crosses native window boundaries and '
            'onefullattentionblock,2framevideo; all22DiTlayers retain lookahead/backward, block24/repeats2 '
            'with25codec tokens; all6vocoder upsamplers and3kernel×3dilation residuals; '
            'native80mel/192speaker width. Four thinker and26talker declared generation steps, native '
            'sampling40/.8/.9/repetition1.05.'),
    },

    'qwen2_5_vl': {
        'reference': {
            'config_class': 'transformers:Qwen2_5_VLConfig',
            'model_class': 'transformers:Qwen2_5_VLForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Qwen/Qwen2.5-VL-7B-Instruct',
                'revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
                'url': 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/cc594898137f460bfe9f0759e9844b3ce807cfb5/config.json',
                'description': ('Pinned HF public conditional-generation example checkpoint; image and video paths, '
                    'native multimodal positions, prefill and continuation.'),
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'second_per_grid_ts'],
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'vision_config': {'depth': 8, 'num_heads': 2, 'hidden_size': 128, 'intermediate_size': 384, 'out_hidden_size': 896},
            'image_token_id': 900,
            'video_token_id': 901,
            'vision_start_token_id': 902,
            'vision_end_token_id': 903,
            'vocab_size': 1024,
            'hidden_size': 896,
            'intermediate_size': 1536,
            'num_hidden_layers': 2,
            'num_attention_heads': 7,
            'num_key_value_heads': 1,
            'max_position_embeddings': 2048,
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'sequence_length': 110, 'image_batch_size': 1,
            'shape': [120, 1176], 'video_shape': [240, 1176], 'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 10, 12]], 'video_grid_thw': [[2, 10, 12]],
            'image_token_positions': list(range(4, 34)), 'video_token_positions': list(range(38, 98)),
            'mm_token_type_ids': True, 'second_per_grid_ts': [2],
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'past_key_values', 'rope_deltas'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduced widths/layers/vocabulary retain checkpoint activation, head dimension128 and '
            'grouped-query ratio, both image/video grids, caches and multimodal rotary sections. Qwen2.5 '
            'retains eight-block window/full pattern with non-square partial windows; Qwen3 retains all27 '
            'vision layers and three DeepStack outputs; MoE retains128 experts/top8.'),
    },

    'qwen2_audio': {
        'reference': {
            'config_class': 'transformers:Qwen2AudioConfig',
            'model_class': 'transformers:Qwen2AudioForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'Qwen/Qwen2-Audio-7B',
                'revision': 'dd84470756e6277a71d4d7188773a43cde92696e',
                'url': 'https://huggingface.co/Qwen/Qwen2-Audio-7B/blob/dd84470756e6277a71d4d7188773a43cde92696e/config.json',
                'description': 'Pinned HF task example checkpoint, padded audio encoder with projected features and cached text continuation.',
            },
            'prefill_input_names': ['input_features', 'feature_attention_mask'],
        },
        'dimension_overrides': {
            'audio_config': {
                'd_model': 128, 'encoder_layers': 2, 'encoder_attention_heads': 2, 'encoder_ffn_dim': 512,
                'max_source_positions': 16,
            },
            'text_config': {
                'vocab_size': 1024, 'hidden_size': 256, 'intermediate_size': 1536, 'num_hidden_layers': 2,
                'num_attention_heads': 2, 'num_key_value_heads': 2, 'max_position_embeddings': 2048,
            },
            'audio_token_index': 900,
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'sequence_length': 24, 'image_batch_size': 1,
            'image_input_name': 'input_features', 'shape': [128, 32], 'feature_attention_lengths': [25],
            'audio_token_positions': [3, 4, 5, 6, 7, 8],
        },
        'outputs': ['logits', 'past_key_values'],
        'dimension_purpose': ('Two full audio/text layers; native128mel bins; padded32-frame audio has25valid frames and six '
            'audio tokens;1:1 multi-head text attention and native128head width; prefill and cached '
            'continuation.'),
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
    },

    'qwen2_moe': {
        'reference': {
            'config_class': 'transformers:Qwen2MoeConfig',
            'model_class': 'transformers:Qwen2MoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned docs/source/en/model_doc/qwen2_moe.md:66-71 loads this causal LM with SDPA. The'
                    ' class-level Qwen2Moe-8x7B example is unavailable (Hub 404). Retain 60 experts, '
                    'top-four unnormalized selected probabilities, the sigmoid-gated shared expert, biased '
                    'QKV, and multi-head full attention. Expert width is reduced to 256 (instead of '
                    'proportional 176) to satisfy the existing Blackwell BF16 kernel requirement of a '
                    'multiple of 128; the initial width 192 passed packing but failed its launch check.'),
                'checkpoint': 'Qwen/Qwen1.5-MoE-A2.7B-Chat',
                'revision': 'ec052fda178e241c7c443468d2fa1db6618996be',
                'url': 'https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/blob/ec052fda178e241c7c443468d2fa1db6618996be/config.json',
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'qwen2_vl': {
        'reference': {
            'config_class': 'transformers:Qwen2VLConfig',
            'model_class': 'transformers:Qwen2VLForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Qwen/Qwen2-VL-7B-Instruct',
                'revision': 'eed13092ef92e448dd6875b2a00151bd3f7db0ac',
                'url': 'https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/eed13092ef92e448dd6875b2a00151bd3f7db0ac/config.json',
                'description': ('Pinned HF public conditional-generation example checkpoint; image and video paths, '
                    'native multimodal positions, prefill and continuation.'),
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'vision_config': {'depth': 2, 'num_heads': 2, 'embed_dim': 128, 'hidden_size': 896},
            'image_token_id': 900,
            'video_token_id': 901,
            'vision_start_token_id': 902,
            'vision_end_token_id': 903,
            'vocab_size': 1024,
            'hidden_size': 896,
            'intermediate_size': 1536,
            'num_hidden_layers': 2,
            'num_attention_heads': 7,
            'num_key_value_heads': 1,
            'max_position_embeddings': 2048,
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'sequence_length': 32, 'image_batch_size': 1,
            'shape': [16, 1176], 'video_shape': [32, 1176], 'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 4, 4]], 'video_grid_thw': [[2, 4, 4]],
            'image_token_positions': [4, 5, 6, 7], 'video_token_positions': list(range(12, 20)),
            'mm_token_type_ids': True,
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'past_key_values', 'rope_deltas'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduced widths/layers/vocabulary retain checkpoint activation, head dimension128 and '
            'grouped-query ratio, both image/video grids, caches and multimodal rotary sections. Qwen2.5 '
            'retains eight-block window/full pattern with non-square partial windows; Qwen3 retains all27 '
            'vision layers and three DeepStack outputs; MoE retains128 experts/top8.'),
    },

    'qwen3': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:Qwen3Config',
            'model_class': 'transformers:Qwen3ForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'Qwen/Qwen3-8B',
                'revision': 'b968826d9c46dd6066d109eabc6255188de91218',
                'description': 'Pinned causal-LM task example checkpoint.',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'qwen3_5': {
        'reference': {
            'config_class': 'transformers:Qwen3_5Config',
            'model_class': 'transformers:Qwen3_5ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Qwen/Qwen3.5-27B',
                'revision': 'fc05daec18b0a78c049392ed2e771dde82bdf654',
                'url': 'https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json',
                'description': ('Pinned HF configuration documentation names Qwen3.5-27B; conditional-generation '
                    'example incorrectly names Qwen3-VL. Preserve full image/video and hybrid '
                    'recurrent/full attention.'),
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
            'prefill_sequence_input_names': ['mm_token_type_ids'],
            'native_cache_defaults': True,
        },
        'input': {
            'kind': 'text_image',
            'text_batch_size': 1,
            'sequence_length': 1746,
            'image_batch_size': 1,
            'shape': [2304, 1536],
            'video_shape': [4608, 1536],
            'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 48, 48]],
            'video_grid_thw': [[2, 48, 48]],
            'image_token_positions': list(range(4, 580)),
            'video_token_positions': [
                584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602,
                603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621,
                622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640,
                641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
                660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678,
                679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
                698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716,
                717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735,
                736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754,
                755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773,
                774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792,
                793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811,
                812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830,
                831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849,
                850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868,
                869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887,
                888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906,
                907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925,
                926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944,
                945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963,
                964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982,
                983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000,
                1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
                1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030,
                1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045,
                1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060,
                1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075,
                1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090,
                1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105,
                1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
                1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135,
                1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150,
                1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1164, 1165, 1166, 1167, 1168, 1169,
                1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184,
                1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199,
                1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214,
                1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229,
                1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244,
                1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259,
                1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274,
                1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289,
                1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304,
                1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319,
                1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334,
                1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349,
                1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364,
                1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379,
                1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394,
                1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409,
                1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424,
                1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439,
                1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454,
                1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469,
                1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484,
                1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499,
                1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514,
                1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529,
                1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544,
                1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559,
                1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574,
                1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589,
                1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604,
                1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619,
                1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634,
                1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649,
                1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664,
                1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679,
                1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694,
                1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709,
                1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724,
                1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739,
            ],
            'mm_token_type_ids': True,
            'fixed_token_ids': {'3': 248053, '580': 248054, '583': 248053, '1160': 248054, '1163': 248053, '1740': 248054},
            'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values', 'rope_deltas'],
        'reference_backend': None,
    },

    'qwen3_5_moe': {
        'reference': {
            'config_class': 'transformers:Qwen3_5MoeConfig',
            'model_class': 'transformers:Qwen3_5MoeForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Qwen/Qwen3.5-35B-A3B',
                'revision': '59d61f3ce65a6d9863b86d2e96597125219dc754',
                'url': 'https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json',
                'description': ('Pinned HF configuration-documentation checkpoint; its full-parent task example adds a '
                    'nonexistent -Instruct suffix (404 recorded).'),
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'text_config': {
                'vocab_size': 1024, 'hidden_size': 1024, 'num_hidden_layers': 4, 'num_attention_heads': 8,
                'num_key_value_heads': 1, 'max_position_embeddings': 2048, 'linear_num_key_heads': 2,
                'linear_num_value_heads': 4,
                'layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
                'num_experts': 16, 'num_experts_per_tok': 8, 'moe_intermediate_size': 256,
                'shared_expert_intermediate_size': 256,
            },
            'vision_config': {
                'depth': 2, 'hidden_size': 128, 'intermediate_size': 384, 'num_heads': 2,
                'out_hidden_size': 1024,
            },
            'image_token_id': 900,
            'video_token_id': 901,
            'vision_start_token_id': 902,
            'vision_end_token_id': 903,
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'sequence_length': 32, 'image_batch_size': 1,
            'shape': [16, 1536], 'video_shape': [32, 1536], 'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 4, 4]], 'video_grid_thw': [[2, 4, 4]],
            'image_token_positions': [4, 5, 6, 7], 'video_token_positions': [12, 13, 14, 15, 20, 21, 22, 23],
            'mm_token_type_ids': True,
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'past_key_values', 'rope_deltas'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Native3:1GDN/full layer pattern, head256 partial64/interleaved3Dpositions,8:1GQA and '
            'attentionwidth2xhidden, linearhead128/key:value1:2; gatedsharedexpert plus '
            'top8-of16developmentroutedexperts in everyblock; nonDeepStackvision, image/two-framevideo, '
            'allrecurrent/conv/KVstates.'),
    },

    'qwen3_moe': {
        'reference': {
            'config_class': 'transformers:Qwen3MoeConfig',
            'model_class': 'transformers:Qwen3MoeForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned docs/source/en/model_doc/qwen3_moe.md:52-55 loads this causal LM. The '
                    'class-level Qwen3-MoE-15B-A2B example is unavailable (Hub 404). Retain 128 experts, '
                    'top-eight normalized routing, per-head Q/K RMSNorm, 8:1 grouped-query attention, query'
                    ' projection width twice the hidden width, and disabled sliding windows. The ordinary '
                    'pinned loader resolves SDPA in the shared runtime.'),
                'checkpoint': 'Qwen/Qwen3-30B-A3B',
                'revision': 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39',
                'url': 'https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json',
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'qwen3_next': {
        'reference': {
            'config_class': 'transformers:Qwen3NextConfig',
            'model_class': 'transformers:Qwen3NextForCausalLM',
            'source': {
                'checkpoint': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
                'revision': '9c7f2fbe84465e40164a94cc16cd30b6999b0cc7',
                'url': 'https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7/config.json',
                'kind': 'example_checkpoint',
                'description': ('Pinned task class and docs example Qwen3-Next-80B-A3B-Instruct. Preserve prefix '
                    'linear/linear/linear/full/linear, active shared-gated MoE at every layer, top10 '
                    'routing, GQA8, GVA2 and partial RoPE25%. Expert pool alone downsized512 to64; GDN '
                    'head128 retained for current existing Blackwell prefill kernel.'),
            },
            'native_cache_defaults': True,
            'native_position_ids': True,
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 515},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'qwen3_omni_moe': {
        'reference': {
            'config_class': 'transformers:Qwen3OmniMoeConfig',
            'model_class': 'transformers:Qwen3OmniMoeForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
                'revision': '26291f793822fb6be9555850f06dfe95f2d7e695',
                'description': 'Matching public Qwen3 Omni parent checkpoint; preserves enabled speech output.',
                'url': 'https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct',
            },
            'generation_output_names': ['sequences', 'waveform'],
        },
        'reference_backend': 'sdpa',
        'dimension_overrides': {
            'thinker_config': {
                'text_config': {
                    'hidden_size': 128, 'intermediate_size': 128, 'moe_intermediate_size': 128,
                    'num_hidden_layers': 4, 'num_attention_heads': 8, 'num_key_value_heads': 1,
                    'num_experts': 16, 'max_position_embeddings': 1024,
                },
                'audio_config': {
                    'd_model': 128, 'downsample_hidden_size': 16, 'encoder_attention_heads': 2,
                    'encoder_ffn_dim': 256, 'encoder_layers': 2, 'output_dim': 128,
                },
                'vision_config': {
                    'depth': 3, 'hidden_size': 128, 'num_heads': 2, 'intermediate_size': 256,
                    'out_hidden_size': 128, 'deepstack_visual_indexes': [0, 1, 2],
                },
            },
            'talker_config': {
                'thinker_hidden_size': 128,
                'accept_hidden_layer': 2,
                'text_config': {
                    'hidden_size': 128, 'intermediate_size': 256, 'moe_intermediate_size': 128,
                    'shared_expert_intermediate_size': 128, 'num_hidden_layers': 2, 'num_attention_heads': 8,
                    'num_key_value_heads': 1, 'num_experts': 16, 'max_position_embeddings': 1024,
                },
                'code_predictor_config': {
                    'hidden_size': 128, 'intermediate_size': 256, 'num_hidden_layers': 2,
                    'num_attention_heads': 2, 'num_key_value_heads': 1, 'max_position_embeddings': 1024,
                    'layer_types': ['full_attention', 'full_attention'],
                },
            },
        },
        'input': {'kind': 'text', 'batch_size': 1, 'sequence_length': 1},
        'workload': 'generate',
        'generation_kwargs': {'thinker_max_new_tokens': 3, 'talker_max_new_tokens': 75},
        'generation_seed': 29,
        'outputs': ['sequences', 'waveform'],
        'dimension_purpose': ('Reduced thinker/talker widths/layers/experts retain thinker MoE top8, talker shared MoE top6, '
            'three DeepStack injections, QK norm/interleaved MRoPE and input audio/image/video. Original '
            'vocabularies,16 code groups, native sampling, all causal waveform upsampling stages;75 talker '
            'steps produce74 code frames to exercise original72-token waveform attention window. Three '
            'thinker steps and seeded sampled audio. Native full Code2Wav dimensions with pretrained '
            'checkpoint state replace near-zero random waveform initialization; thinker/talker remain '
            'common seeded random state.'),
    },

    'qwen3_vl': {
        'reference': {
            'config_class': 'transformers:Qwen3VLConfig',
            'model_class': 'transformers:Qwen3VLForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Qwen/Qwen3-VL-8B-Instruct',
                'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                'url': 'https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json',
                'description': ('Pinned HF public conditional-generation example checkpoint; image and video paths, '
                    'native multimodal positions, prefill and continuation.'),
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'text_config': {
                'vocab_size': 1024, 'hidden_size': 512, 'intermediate_size': 1536, 'num_hidden_layers': 3,
                'num_attention_heads': 4, 'num_key_value_heads': 1, 'max_position_embeddings': 2048,
            },
            'vision_config': {
                'depth': 27, 'num_heads': 2, 'hidden_size': 128, 'intermediate_size': 384,
                'out_hidden_size': 512,
            },
            'image_token_id': 900,
            'video_token_id': 901,
            'vision_start_token_id': 902,
            'vision_end_token_id': 903,
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'sequence_length': 32, 'image_batch_size': 1,
            'shape': [16, 1536], 'video_shape': [32, 1536], 'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 4, 4]], 'video_grid_thw': [[2, 4, 4]],
            'image_token_positions': [4, 5, 6, 7], 'video_token_positions': [12, 13, 14, 15, 20, 21, 22, 23],
            'mm_token_type_ids': True,
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'past_key_values', 'rope_deltas'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduced widths/layers/vocabulary retain checkpoint activation, head dimension128 and '
            'grouped-query ratio, both image/video grids, caches and multimodal rotary sections. Qwen2.5 '
            'retains eight-block window/full pattern with non-square partial windows; Qwen3 retains all27 '
            'vision layers and three DeepStack outputs; MoE retains128 experts/top8.'),
    },

    'qwen3_vl_moe': {
        'reference': {
            'config_class': 'transformers:Qwen3VLMoeConfig',
            'model_class': 'transformers:Qwen3VLMoeForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
                'revision': '9c4b90e1e4ba969fd3b5378b57d966d725f1b86c',
                'url': 'https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c/config.json',
                'description': ('Pinned HF public conditional-generation example checkpoint; image and video paths, '
                    'native multimodal positions, prefill and continuation.'),
            },
            'native_position_ids': True,
            'prefill_input_names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
            'prefill_sequence_input_names': ['mm_token_type_ids'],
        },
        'dimension_overrides': {
            'text_config': {
                'vocab_size': 1024, 'hidden_size': 512, 'intermediate_size': 1536, 'num_hidden_layers': 3,
                'num_attention_heads': 8, 'num_key_value_heads': 1, 'max_position_embeddings': 2048,
                'moe_intermediate_size': 64,
            },
            'vision_config': {
                'depth': 27, 'num_heads': 2, 'hidden_size': 128, 'intermediate_size': 384,
                'out_hidden_size': 512,
            },
            'image_token_id': 900,
            'video_token_id': 901,
            'vision_start_token_id': 902,
            'vision_end_token_id': 903,
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'sequence_length': 32, 'image_batch_size': 1,
            'shape': [16, 1536], 'video_shape': [32, 1536], 'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 4, 4]], 'video_grid_thw': [[2, 4, 4]],
            'image_token_positions': [4, 5, 6, 7], 'video_token_positions': [12, 13, 14, 15, 20, 21, 22, 23],
            'mm_token_type_ids': True,
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'past_key_values', 'rope_deltas'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduced widths/layers/vocabulary retain checkpoint activation, head dimension128 and '
            'grouped-query ratio, both image/video grids, caches and multimodal rotary sections. Qwen2.5 '
            'retains eight-block window/full pattern with non-square partial windows; Qwen3 retains all27 '
            'vision layers and three DeepStack outputs; MoE retains128 experts/top8.'),
    },

    'recurrent_gemma': {
        'reference': {
            'config_class': 'transformers:RecurrentGemmaConfig',
            'model_class': 'transformers:RecurrentGemmaForCausalLM',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/recurrent_gemma/configuration_recurrent_gemma.py',
                'description': ('HF docs construct RecurrentGemmaConfig; named google/recurrentgemma-2b config '
                    'gated403. Author RECURRENT_GEMMA_2B_V1 corroborates constructor '
                    'width2560/intermediate7680/layers26/heads10/RRApattern/window2048/logitcap30. Other '
                    'fields remain pinned HF constructor defaults.'),
                'author_revision': '2efa84dac0e68e63547a27a18fa943c98f1c312e',
                'author_url': 'https://raw.githubusercontent.com/google-deepmind/recurrentgemma/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/common.py',
            },
            'caller_retained_cache': True,
            'conv_cache_history': 3,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 2063},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values', 'hidden_states'],
    },

    'reformer': {
        'reference': {
            'config_class': 'transformers:ReformerConfig',
            'model_class': 'transformers:ReformerForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'hf-internal-testing/tiny-random-reformer',
                'revision': '92d3924b57fe38f8c03ad579f85e7c4b3614e804',
                'url': 'https://huggingface.co/hf-internal-testing/tiny-random-reformer/blob/92d3924b57fe38f8c03ad579f85e7c4b3614e804/config.json',
                'description': ('Pinned HF MLM forward example explicitly uses this illustrative checkpoint because no '
                    'pretrained masked-LM checkpoint is available; all four default layers are local, not '
                    'LSH.'),
            },
        },
        'dimension_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 96},
        'workload': 'masked_lm',
        'reference_backend': 'eager',
        'dimension_purpose': ('Retain the entire tiny illustrative checkpoint, including four local layers, axial factors4x25'
            ' and chunk4; 96 tokens exercise cyclic neighboring chunks.'),
    },

    'regnet': {
        'reference': {
            'config_class': 'transformers.models.regnet.configuration_regnet:RegNetConfig',
            'model_class': 'transformers.models.regnet.modeling_regnet:RegNetModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs RegNetConfig() and '
                    'RegNetModel(configuration).'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/regnet/configuration_regnet.py#L35',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'rembert': {
        'reference': {
            'config_class': 'transformers:RemBertConfig',
            'model_class': 'transformers:RemBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/rembert',
                'revision': '65da5133da36e29dfca67d4f0dd9f7f9db21b563',
                'url': 'https://huggingface.co/google/rembert/blob/65da5133da36e29dfca67d4f0dd9f7f9db21b563/config.json',
                'description': ('Preserve the corpus masked-LM task using the checkpoint named by pinned HF '
                    'src/transformers/models/rembert/configuration_rembert.py:22; complete omitted values '
                    'from the pinned constructor.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'resnet': {
        'reference': {
            'config_class': 'transformers.models.resnet.configuration_resnet:ResNetConfig',
            'model_class': 'transformers.models.resnet.modeling_resnet:ResNetModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs ResNetModel(ResNetConfig()) with '
                    'random weights. This base-model task uses constructor computational defaults; '
                    'checkpoint configuration is not loaded.'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/resnet/configuration_resnet.py#L37-L49',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'roberta': {
        'reference': {
            'config_class': 'transformers:RobertaConfig',
            'model_class': 'transformers:RobertaForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'FacebookAI/roberta-base',
                'revision': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b',
                'url': 'https://huggingface.co/FacebookAI/roberta-base/blob/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/roberta.md:60 masked-LM example; checkpoint '
                    'configuration completed by pinned constructor defaults.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'roberta_prelayernorm': {
        'reference': {
            'config_class': 'transformers:RobertaPreLayerNormConfig',
            'model_class': 'transformers:RobertaPreLayerNormForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'andreasmadsen/efficient_mlm_m0.40',
                'revision': '3924d3fe62a60eb393a809cec55355f2837de173',
                'url': 'https://huggingface.co/andreasmadsen/efficient_mlm_m0.40/blob/3924d3fe62a60eb393a809cec55355f2837de173/config.json',
                'description': ('Preserve the corpus masked-LM task using the checkpoint named by pinned HF '
                    'src/transformers/models/roberta_prelayernorm/configuration_roberta_prelayernorm.py:24;'
                    ' complete omitted values from the pinned constructor.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'roc_bert': {
        'reference': {
            'config_class': 'transformers:RoCBertConfig',
            'model_class': 'transformers:RoCBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned RoCBertForMaskedLM example loads roc-bert-base-zh; retain concatenated word, '
                    'shape, and pronunciation embeddings and the full masked-LM logits.'),
                'checkpoint': 'weiweishi/roc-bert-base-zh',
                'revision': '0348c672af2fc2aea383cd1275e43398e7a1ebd1',
                'url': 'https://huggingface.co/weiweishi/roc-bert-base-zh/blob/0348c672af2fc2aea383cd1275e43398e7a1ebd1/config.json',
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'tokens',
            'batch_size': 1,
            'sequence_length': 512,
            'auxiliary_fields': {'input_shape_ids': 'shape_vocab_size', 'input_pronunciation_ids': 'pronunciation_vocab_size'},
        },
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'roformer': {
        'reference': {
            'config_class': 'transformers:RoFormerConfig',
            'model_class': 'transformers:RoFormerForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Pinned RoFormer masked-LM documentation loads junnyu/roformer_chinese_base; retain '
                    'adjacent-pair rotary Q/K, unrotated V, and equal embedding/hidden width from that '
                    'checkpoint.'),
                'checkpoint': 'junnyu/roformer_chinese_base',
                'revision': 'f780ca57e5eab4627f334bed71a8e5a353b33d69',
                'url': 'https://huggingface.co/junnyu/roformer_chinese_base/blob/f780ca57e5eab4627f334bed71a8e5a353b33d69/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'rt_detr': {
        'reference': {
            'config_class': 'transformers:RTDetrConfig',
            'model_class': 'transformers:RTDetrForObjectDetection',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'PekingU/rtdetr_r50vd',
                'revision': 'df939e661d8c52e80608d1ec566561aabd25a4e7',
            },
        },
        'input': {'kind': 'image', 'shape': [3, 640, 640], 'batch_size': 1},
        'workload': 'forward',
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'intermediate_hidden_states', 'intermediate_logits',
            'intermediate_reference_points', 'encoder_last_hidden_state', 'init_reference_points',
            'enc_topk_bboxes', 'enc_topk_logits', 'enc_outputs_class', 'enc_outputs_coord_logits',
        ],
        'reference_backend': None,
    },

    'rt_detr_resnet': {
        'reference': {
            'config_class': 'transformers:RTDetrResNetConfig',
            'model_class': 'transformers:RTDetrResNetBackbone',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/rt_detr/modeling_rt_detr_resnet.py#L354',
                'description': ('The pinned backbone forward example constructs RTDetrResNetConfig(). Preserve its '
                    'bottleneck stages, three-convolution stem, average-pool shortcuts, and default '
                    'final-stage feature map.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['feature_maps'],
        'reference_backend': None,
    },

    'rt_detr_v2': {
        'reference': {
            'config_class': 'transformers:RTDetrV2Config',
            'model_class': 'transformers:RTDetrV2ForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'PekingU/rtdetr_v2_r18vd',
                'revision': '5650961749fa93567c0d46fc7f43ea4f9e914107',
                'url': 'https://huggingface.co/PekingU/rtdetr_v2_r18vd/blob/5650961749fa93567c0d46fc7f43ea4f9e914107/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/rt_detr_v2.md:52-59 gives the concrete '
                    'object-detection example. Preserve its basic ResNet blocks, hidden expansion 0.5, and '
                    'three decoder layers; complete omitted fields from pinned constructor defaults.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 640, 640]},
        'workload': 'forward',
        'outputs': [
            'logits', 'pred_boxes', 'last_hidden_state', 'intermediate_hidden_states', 'intermediate_logits',
            'intermediate_reference_points', 'encoder_last_hidden_state', 'init_reference_points',
            'enc_topk_bboxes', 'enc_topk_logits', 'enc_outputs_class', 'enc_outputs_coord_logits',
        ],
        'reference_backend': None,
    },

    'rwkv': {
        'reference': {
            'config_class': 'transformers:RwkvConfig',
            'model_class': 'transformers:RwkvForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'RWKV/rwkv-4-169m-pile',
                'revision': '46bdc280eb97b6141d5d51a935e0c4870ecaefcc',
                'url': 'https://huggingface.co/RWKV/rwkv-4-169m-pile/blob/46bdc280eb97b6141d5d51a935e0c4870ecaefcc/config.json',
                'description': ('Pinned RwkvConfig documented checkpoint; preserve primary RwkvForCausalLM task, cached'
                    ' single-prompt inference, untied head, and rescale_every6. The model-doc430m example '
                    'uses the base RwkvModel.'),
            },
            'cache_argument': 'state',
            'cache_output': 'state',
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 131},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'state'],
    },

    'sam': {
        'reference': {
            'config_class': 'transformers:SamConfig',
            'model_class': 'transformers:SamModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'facebook/sam-vit-base',
                'revision': '70c1a07f894ebb5b307fd9eaaee97b9dfc16068f',
                'description': 'Pinned HF public point-prompt segmentation example.',
            },
        },
        'input': {'kind': 'external', 'shape': [3, 1024, 1024], 'batch_size': 1},
        'outputs': ['pred_masks', 'iou_scores'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'sam2': {
        'reference': {
            'config_class': 'transformers:Sam2Config',
            'model_class': 'transformers:Sam2Model',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'danelcsb/sam2.1_hiera_tiny',
                'revision': '5dfafcd6f120afc7acf900e5c6c413187e9f989c',
                'description': ('Pinned image-task source example checkpoint; stored model_type is video, explicit '
                    'corpus Sam2Config selects shared image settings.'),
            },
        },
        'input': {'kind': 'external', 'shape': [3, 1024, 1024], 'batch_size': 1},
        'outputs': ['pred_masks', 'iou_scores', 'object_score_logits', 'image_embeddings'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'sam2_video': {
        'reference': {
            'config_class': 'transformers:Sam2VideoConfig',
            'model_class': 'transformers:Sam2VideoModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'facebook/sam2.1-hiera-tiny',
                'revision': 'de431c4043854a71d8101e17995dfe596bf101a5',
                'description': 'Pinned model documentation video tracking example.',
            },
        },
        'input': {
            'kind': 'video', 'name': 'video', 'shape': [3, 1024, 1024], 'batch_size': 18,
            'input_points': [[[[512.0, 512.0]]]], 'input_labels': [[[1]]],
            'dtypes': {'input_points': 'float32', 'input_labels': 'int64'},
        },
        'outputs': ['pred_masks', 'object_score_logits'],
        'workload': 'sam2_video',
        'reference_backend': None,
    },

    'sam3': {
        'reference': {
            'config_class': 'transformers:Sam3Config',
            'model_class': 'transformers:Sam3Model',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/sam3/configuration_sam3.py#L216-L222',
                'description': ('Explicit full-model constructor example; not a claim of equivalence to the '
                    'inaccessible facebook/sam3 checkpoint.'),
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 32,
            'shape': [3, 1008, 1008], 'eos_positions': [7], 'bos_token_id': 49406, 'eos_token_id': 49407,
            'pad_token_id': 1, 'attention_mask': True,
        },
        'workload': 'forward',
        'outputs': [
            'pred_masks', 'pred_boxes', 'pred_logits', 'presence_logits', 'semantic_seg',
            'decoder_reference_boxes',
        ],
    },

    'sam3_lite_text': {
        'reference': {
            'config_class': 'transformers:Sam3LiteTextConfig',
            'model_class': 'transformers:Sam3LiteTextModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'yonigozlan/sam3-litetext-s0',
                'revision': 'b09766e54f5d2eba021119ec7feff13e74c0f8fc',
                'description': 'Pinned model_doc Sam3LiteTextModel ordinary image/text segmentation example.',
            },
        },
        'input': {
            'kind': 'text_image', 'shape': [3, 1008, 1008], 'text_batch_size': 1, 'image_batch_size': 1,
            'sequence_length': 32,
            'input_ids': [[49406, 86, 2, 118, 11, 120, 59, 89, 73, 81, 13, 115, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            'attention_mask': True, 'pad_token_id': 0, 'batch_size': 1,
        },
        'outputs': [
            'pred_masks', 'pred_boxes', 'pred_logits', 'presence_logits', 'semantic_seg',
            'decoder_reference_boxes',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'sam3_tracker': {
        'reference': {
            'config_class': 'transformers:Sam3TrackerConfig',
            'model_class': 'transformers:Sam3TrackerModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/sam3_tracker/configuration_sam3_tracker.py#L111-L115',
                'description': ('Explicit constructor example for point-prompt image segmentation; retain its four '
                    'feature levels and native three returned embeddings.'),
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'image',
            'batch_size': 1,
            'shape': [3, 1008, 1008],
            'input_points': [[[[504.0, 504.0]]]],
            'dtypes': {'input_points': 'float32'},
        },
        'workload': 'forward',
        'outputs': ['pred_masks', 'iou_scores', 'object_score_logits', 'image_embeddings'],
    },

    'sam3_tracker_video': {'reference': {'config_class': 'transformers:Sam3TrackerVideoConfig',
                   'model_class': 'transformers:Sam3TrackerVideoModel',
                   'source': {'kind': 'constructor_defaults',
                              'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                              'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py#L175-L201',
                              'description': 'Pinned HF full-model constructor example, selected '
                                             'independently of the gated facebook/sam3 checkpoint; '
                                             'point-initialized video propagation.'}},
     'dimension_overrides': {'image_size': 448,
                             'vision_config': {'backbone_config': {'hidden_size': 128,
                                                                   'intermediate_size': 592,
                                                                   'num_hidden_layers': 2,
                                                                   'num_attention_heads': 2,
                                                                   'global_attn_indexes': [1]}}},
     'dimension_purpose': 'Two vision layers retain local/global attention, head width 64 and feed-forward '
                          'ratio 592/128. Patch size 14 and image size 448 produce a 32x32 grid crossing the '
                          'default 24-token window. All pyramid, decoder and memory features retain '
                          'their defaults; 18 frames cross seven memory slots and 16 object pointers.',
     'input': {'kind': 'video',
               'name': 'video',
               'batch_size': 18,
               'shape': [3, 448, 448],
               'input_points': [[[[224.0, 224.0]]]],
               'input_labels': [[[1]]],
               'dtypes': {'input_points': 'float32', 'input_labels': 'int64'}},
     'outputs': ['pred_masks', 'object_score_logits'],
     'workload': 'sam3_tracker_video',
     'reference_backend': None},

    'seed_oss': {
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'reference': {
            'config_class': 'transformers:SeedOssConfig',
            'model_class': 'transformers:SeedOssForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'ByteDance-Seed/Seed-OSS-36B-Instruct',
                'revision': '497f1dca95ebdec98e41d517b9f060ee753c902f',
                'description': 'Pinned causal-LM example checkpoint.',
            },
        },
        'dimension_overrides': {
            'num_hidden_layers': 2, 'vocab_size': 1024, 'max_position_embeddings': 1024, 'hidden_size': 320,
            'intermediate_size': 1728, 'num_attention_heads': 10, 'num_key_value_heads': 1, 'head_dim': 64,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 257},
        'outputs': ['logits'],
        'dimension_purpose': ('Retains GQA10:1, independent attention width2H, MLP/H5.4 and nonzero QKV bias path; theta1e7 '
            'unchanged. Development dimensions only.'),
    },

    'seamless_m4t': {
        'variants': {
            'text': {
                'reference': {
                    'config_class': 'transformers:SeamlessM4TConfig',
                    'model_class': 'transformers:SeamlessM4TModel',
                    'source': {
                        'kind': 'example_checkpoint',
                        'checkpoint': 'facebook/hf-seamless-m4t-medium',
                        'revision': 'ecf60d4df63baaac3f82ae6a7ad7adcb19dcb26c',
                        'url': 'https://huggingface.co/facebook/hf-seamless-m4t-medium/blob/ecf60d4df63baaac3f82ae6a7ad7adcb19dcb26c/config.json',
                        'description': (
                            'Pinned HF docs/source/en/model_doc/seamless_m4t.md demonstrates both '
                            'text-to-speech and audio-to-speech generation with medium checkpoint and '
                            'target rus. Generation IDs/maps from same revision generation_config.json, '
                            'restricted only to evaluated rus language.'
                        ),
                    },
                    'generation_config': {
                        'bos_token_id': 2,
                        'decoder_start_token_id': 3,
                        'eos_token_id': 3,
                        'max_new_tokens': 256,
                        'pad_token_id': 0,
                        't2u_lang_code_to_id': {'rus': 10067},
                        'text_decoder_lang_to_code_id': {'rus': 256147},
                        'transformers_version': '4.35.0.dev0',
                        'vocoder_lang_code_to_id': {'rus': 23},
                    },
                    'generation_output_names': ['waveform', 'waveform_lengths'],
                    'fan_in_normal_modules': ['t2u_model.model.decoder', 'vocoder.dur_predictor', 'vocoder.hifi_gan'],
                },
                'dimension_overrides': {
                    'decoder_attention_heads': 2,
                    'decoder_ffn_dim': 512,
                    'decoder_layers': 2,
                    'encoder_attention_heads': 2,
                    'encoder_ffn_dim': 512,
                    'encoder_layers': 2,
                    'hidden_size': 128,
                    'lang_embed_dim': 32,
                    'speech_encoder_attention_heads': 2,
                    'speech_encoder_intermediate_size': 512,
                    'speech_encoder_layers': 2,
                    'spkr_embed_dim': 32,
                    't2u_decoder_attention_heads': 2,
                    't2u_decoder_ffn_dim': 1024,
                    't2u_decoder_layers': 2,
                    't2u_encoder_attention_heads': 2,
                    't2u_encoder_ffn_dim': 1024,
                    't2u_encoder_layers': 2,
                    'unit_embed_dim': 256,
                    'upsample_initial_channel': 128,
                },
                'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 13},
                'workload': 'generate',
                'outputs': ['waveform', 'waveform_lengths'],
                'generation_kwargs': {'tgt_lang': 'rus', 'text_max_new_tokens': 4, 'speech_max_new_tokens': 4},
                'implementation_kwargs': {
                    'generation_config': {
                        'bos_token_id': 2,
                        'decoder_start_token_id': 3,
                        'eos_token_id': 3,
                        'max_new_tokens': 256,
                        'pad_token_id': 0,
                        't2u_lang_code_to_id': {'rus': 10067},
                        'text_decoder_lang_to_code_id': {'rus': 256147},
                        'transformers_version': '4.35.0.dev0',
                        'vocoder_lang_code_to_id': {'rus': 23},
                    },
                },
                'reference_backend': 'eager',
                'dimension_purpose': (
                    'Reduce transformer depth to2, width to128 with64-wide heads and original FF ratios; '
                    'retain full text/unit/language vocabularies, 160 input features, relative positions,'
                    ' kernel31 speech convolution, stride8 adapter, all5 vocoder upsampling stages and '
                    'all3 residual kernels/dilations. Text13tokens and audio65frames exercise positional '
                    'and temporal kernels;4 generated text/unit steps exercise cached updates. Both '
                    'documented input modalities retain complete speech synthesis. Shared FP32 fan-in '
                    'initialization of complete unitdecoder linear layers prevents untrained tied '
                    'language-token repetition from creating out-of-range vocoder IDs; duration predictor'
                    ' and HiFiGAN fan-in give meaningful duration/output signal. '
                    'Embeddings/tiedhead/LN/biases unchanged; no forced/suppressed tokens or pretrained '
                    'weights.'
                ),
            },
            'audio': {
                'reference': {
                    'config_class': 'transformers:SeamlessM4TConfig',
                    'model_class': 'transformers:SeamlessM4TModel',
                    'source': {
                        'kind': 'example_checkpoint',
                        'checkpoint': 'facebook/hf-seamless-m4t-medium',
                        'revision': 'ecf60d4df63baaac3f82ae6a7ad7adcb19dcb26c',
                        'url': 'https://huggingface.co/facebook/hf-seamless-m4t-medium/blob/ecf60d4df63baaac3f82ae6a7ad7adcb19dcb26c/config.json',
                        'description': (
                            'Pinned HF docs/source/en/model_doc/seamless_m4t.md demonstrates both '
                            'text-to-speech and audio-to-speech generation with medium checkpoint and '
                            'target rus. Generation IDs/maps from same revision generation_config.json, '
                            'restricted only to evaluated rus language.'
                        ),
                    },
                    'generation_config': {
                        'bos_token_id': 2,
                        'decoder_start_token_id': 3,
                        'eos_token_id': 3,
                        'max_new_tokens': 256,
                        'pad_token_id': 0,
                        't2u_lang_code_to_id': {'rus': 10067},
                        'text_decoder_lang_to_code_id': {'rus': 256147},
                        'transformers_version': '4.35.0.dev0',
                        'vocoder_lang_code_to_id': {'rus': 23},
                    },
                    'generation_output_names': ['waveform', 'waveform_lengths'],
                    'fan_in_normal_modules': ['t2u_model.model.decoder', 'vocoder.dur_predictor', 'vocoder.hifi_gan'],
                },
                'dimension_overrides': {
                    'decoder_attention_heads': 2,
                    'decoder_ffn_dim': 512,
                    'decoder_layers': 2,
                    'encoder_attention_heads': 2,
                    'encoder_ffn_dim': 512,
                    'encoder_layers': 2,
                    'hidden_size': 128,
                    'lang_embed_dim': 32,
                    'speech_encoder_attention_heads': 2,
                    'speech_encoder_intermediate_size': 512,
                    'speech_encoder_layers': 2,
                    'spkr_embed_dim': 32,
                    't2u_decoder_attention_heads': 2,
                    't2u_decoder_ffn_dim': 1024,
                    't2u_decoder_layers': 2,
                    't2u_encoder_attention_heads': 2,
                    't2u_encoder_ffn_dim': 1024,
                    't2u_encoder_layers': 2,
                    'unit_embed_dim': 256,
                    'upsample_initial_channel': 128,
                },
                'input': {
                    'kind': 'spectrogram',
                    'name': 'input_features',
                    'batch_size': 1,
                    'shape': [65, 160],
                    'attention_mask_length': 65,
                },
                'workload': 'generate',
                'outputs': ['waveform', 'waveform_lengths'],
                'generation_kwargs': {'tgt_lang': 'rus', 'text_max_new_tokens': 4, 'speech_max_new_tokens': 4},
                'implementation_kwargs': {
                    'generation_config': {
                        'bos_token_id': 2,
                        'decoder_start_token_id': 3,
                        'eos_token_id': 3,
                        'max_new_tokens': 256,
                        'pad_token_id': 0,
                        't2u_lang_code_to_id': {'rus': 10067},
                        'text_decoder_lang_to_code_id': {'rus': 256147},
                        'transformers_version': '4.35.0.dev0',
                        'vocoder_lang_code_to_id': {'rus': 23},
                    },
                },
                'reference_backend': 'eager',
                'dimension_purpose': (
                    'Reduce transformer depth to2, width to128 with64-wide heads and original FF ratios; '
                    'retain full text/unit/language vocabularies, 160 input features, relative positions,'
                    ' kernel31 speech convolution, stride8 adapter, all5 vocoder upsampling stages and '
                    'all3 residual kernels/dilations. Text13tokens and audio65frames exercise positional '
                    'and temporal kernels;4 generated text/unit steps exercise cached updates. Both '
                    'documented input modalities retain complete speech synthesis. Shared FP32 fan-in '
                    'initialization of complete unitdecoder linear layers prevents untrained tied '
                    'language-token repetition from creating out-of-range vocoder IDs; duration predictor'
                    ' and HiFiGAN fan-in give meaningful duration/output signal. '
                    'Embeddings/tiedhead/LN/biases unchanged; no forced/suppressed tokens or pretrained '
                    'weights.'
                ),
            },
        },
    },

    'seamless_m4t_v2': {
        'variants': {
            'text': {
                'reference': {
                    'config_class': 'transformers:SeamlessM4Tv2Config',
                    'model_class': 'transformers:SeamlessM4Tv2Model',
                    'source': {
                        'kind': 'example_checkpoint',
                        'checkpoint': 'facebook/seamless-m4t-v2-large',
                        'revision': '5f8cc790b19fc3f67a61c105133b20b34e3dcb76',
                        'description': (
                            'Pinned HF seamless_m4t_v2 model documentation explicitly demonstrates both '
                            'text/audio-to-speech generation; retain character-conditioned unit synthesis'
                            ' and full vocoder.'
                        ),
                        'url': 'https://huggingface.co/facebook/seamless-m4t-v2-large/blob/5f8cc790b19fc3f67a61c105133b20b34e3dcb76/config.json',
                    },
                    'generation_config_source': {
                        'repo': 'facebook/seamless-m4t-v2-large',
                        'revision': '5f8cc790b19fc3f67a61c105133b20b34e3dcb76',
                    },
                    'generation_output_names': ['waveform', 'waveform_lengths'],
                    'fan_in_normal_modules': ['t2u_model.model.decoder', 'vocoder.dur_predictor', 'vocoder.hifi_gan'],
                },
                'implementation_kwargs': {
                    'generation_config_source': {
                        'repo': 'facebook/seamless-m4t-v2-large',
                        'revision': '5f8cc790b19fc3f67a61c105133b20b34e3dcb76',
                    },
                },
                'dimension_overrides': {
                    'decoder_attention_heads': 2,
                    'decoder_ffn_dim': 1024,
                    'decoder_layers': 2,
                    'encoder_attention_heads': 2,
                    'encoder_ffn_dim': 1024,
                    'encoder_layers': 2,
                    'hidden_size': 128,
                    'lang_embed_dim': 32,
                    'speech_encoder_attention_heads': 2,
                    'speech_encoder_chunk_size': 32,
                    'speech_encoder_intermediate_size': 512,
                    'speech_encoder_layers': 2,
                    'speech_encoder_left_chunk_num': 2,
                    'spkr_embed_dim': 32,
                    't2u_decoder_attention_heads': 2,
                    't2u_decoder_ffn_dim': 1024,
                    't2u_decoder_layers': 2,
                    't2u_encoder_attention_heads': 2,
                    't2u_encoder_ffn_dim': 1024,
                    't2u_encoder_layers': 2,
                    't2u_variance_predictor_embed_dim': 128,
                    't2u_variance_predictor_hidden_dim': 32,
                    'unit_embed_dim': 160,
                    'upsample_initial_channel': 128,
                },
                'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 13},
                'workload': 'generate',
                'outputs': ['waveform', 'waveform_lengths'],
                'generation_kwargs': {'tgt_lang': 'rus', 'text_max_new_tokens': 4},
                'reference_backend': 'eager',
                'dimension_purpose': (
                    'Use2repeated transformer layers, hidden128 and2heads preservinghead64; text/unit '
                    'FF1024 preserve8:1, speechFF512 preserve4:1. Retain vocabulary/character/language '
                    'maps, both character and vocoder learned-duration expansions, kernel31 causal speech'
                    ' convolution, relative clipping64/8, stride8 adapter, all5 HiFiGAN stages/residual '
                    'kernels. Character predictor input128/hidden32 preserves native4:1; vocoder '
                    'unit/lang/speaker widths scaled uniformly8x to160/32/32. Audio129frames with '
                    'explicit chunk32 and2prior chunks exercises partialfinalchunk, historycutoff, '
                    'futurechunk masking and visible relative distances95>64 and31>8. Text13tokens '
                    'and4generatedsteps preservecached decoding and several character groups. Shared '
                    'standard fan-in random weights in declared subtrees expose duration/vocoder '
                    'computations without token forcing/suppression; BF16 waveform signal still requires '
                    'review.'
                ),
            },
            'audio': {
                'reference': {
                    'config_class': 'transformers:SeamlessM4Tv2Config',
                    'model_class': 'transformers:SeamlessM4Tv2Model',
                    'source': {
                        'kind': 'example_checkpoint',
                        'checkpoint': 'facebook/seamless-m4t-v2-large',
                        'revision': '5f8cc790b19fc3f67a61c105133b20b34e3dcb76',
                        'description': (
                            'Pinned HF seamless_m4t_v2 model documentation explicitly demonstrates both '
                            'text/audio-to-speech generation; retain character-conditioned unit synthesis'
                            ' and full vocoder.'
                        ),
                        'url': 'https://huggingface.co/facebook/seamless-m4t-v2-large/blob/5f8cc790b19fc3f67a61c105133b20b34e3dcb76/config.json',
                    },
                    'generation_config_source': {
                        'repo': 'facebook/seamless-m4t-v2-large',
                        'revision': '5f8cc790b19fc3f67a61c105133b20b34e3dcb76',
                    },
                    'generation_output_names': ['waveform', 'waveform_lengths'],
                    'fan_in_normal_modules': ['t2u_model.model.decoder', 'vocoder.dur_predictor', 'vocoder.hifi_gan'],
                },
                'implementation_kwargs': {
                    'generation_config_source': {
                        'repo': 'facebook/seamless-m4t-v2-large',
                        'revision': '5f8cc790b19fc3f67a61c105133b20b34e3dcb76',
                    },
                },
                'dimension_overrides': {
                    'decoder_attention_heads': 2,
                    'decoder_ffn_dim': 1024,
                    'decoder_layers': 2,
                    'encoder_attention_heads': 2,
                    'encoder_ffn_dim': 1024,
                    'encoder_layers': 2,
                    'hidden_size': 128,
                    'lang_embed_dim': 32,
                    'speech_encoder_attention_heads': 2,
                    'speech_encoder_chunk_size': 32,
                    'speech_encoder_intermediate_size': 512,
                    'speech_encoder_layers': 2,
                    'speech_encoder_left_chunk_num': 2,
                    'spkr_embed_dim': 32,
                    't2u_decoder_attention_heads': 2,
                    't2u_decoder_ffn_dim': 1024,
                    't2u_decoder_layers': 2,
                    't2u_encoder_attention_heads': 2,
                    't2u_encoder_ffn_dim': 1024,
                    't2u_encoder_layers': 2,
                    't2u_variance_predictor_embed_dim': 128,
                    't2u_variance_predictor_hidden_dim': 32,
                    'unit_embed_dim': 160,
                    'upsample_initial_channel': 128,
                },
                'input': {
                    'kind': 'spectrogram',
                    'name': 'input_features',
                    'batch_size': 1,
                    'shape': [129, 160],
                    'attention_mask_length': 129,
                },
                'workload': 'generate',
                'outputs': ['waveform', 'waveform_lengths'],
                'generation_kwargs': {'tgt_lang': 'rus', 'text_max_new_tokens': 4},
                'reference_backend': 'eager',
                'dimension_purpose': (
                    'Use2repeated transformer layers, hidden128 and2heads preservinghead64; text/unit '
                    'FF1024 preserve8:1, speechFF512 preserve4:1. Retain vocabulary/character/language '
                    'maps, both character and vocoder learned-duration expansions, kernel31 causal speech'
                    ' convolution, relative clipping64/8, stride8 adapter, all5 HiFiGAN stages/residual '
                    'kernels. Character predictor input128/hidden32 preserves native4:1; vocoder '
                    'unit/lang/speaker widths scaled uniformly8x to160/32/32. Audio129frames with '
                    'explicit chunk32 and2prior chunks exercises partialfinalchunk, historycutoff, '
                    'futurechunk masking and visible relative distances95>64 and31>8. Text13tokens '
                    'and4generatedsteps preservecached decoding and several character groups. Shared '
                    'standard fan-in random weights in declared subtrees expose duration/vocoder '
                    'computations without token forcing/suppression; BF16 waveform signal still requires '
                    'review.'
                ),
            },
        },
    },

    'segformer': {
        'reference': {
            'config_class': 'transformers.models.segformer.configuration_segformer:SegformerConfig',
            'model_class': 'transformers.models.segformer.modeling_segformer:SegformerForSemanticSegmentation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'nvidia/segformer-b0-finetuned-ade-512-512',
                'revision': '489d5cd81a0b59fab9b7ea758d3548ebe99677da',
                'hf_source_revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned SegformerForSemanticSegmentation.forward example names this 150-label '
                    'ADE20K checkpoint. Keep its complete four-scale decoder and ordinary logits.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/segformer/modeling_segformer.py#L627',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 512, 512]},
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'seggpt': {
        'reference': {
            'config_class': 'transformers:SegGptConfig',
            'model_class': 'transformers:SegGptForImageSegmentation',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'BAAI/seggpt-vit-large',
                'revision': '77b179b5a1b50a48bd04ef3138804cfacc2ced73',
                'description': 'Pinned HF public task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'shape': [3, 448, 448], 'batch_size': 1, 'segmentation_prompt': True},
        'outputs': ['pred_masks'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'sew': {
        'reference': {
            'config_class': 'transformers:SEWConfig',
            'model_class': 'transformers:SEWModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/sew/configuration_sew.py#L106',
                'description': ('The explicit example constructs configuration defaults; the unrelated SegGPT '
                    'checkpoint in the decorator conflicts with that example. Preserve the group-norm '
                    'frontend and squeeze/upsampling paths.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [164480]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'sew_d': {
        'reference': {
            'config_class': 'transformers:SEWDConfig',
            'model_class': 'transformers:SEWDModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/sew_d/configuration_sew_d.py#L114',
                'description': ('The explicit example constructs configuration defaults; the unrelated SegGPT '
                    'checkpoint in the decorator conflicts with that example. Preserve the group-norm '
                    'frontend and squeeze/upsampling paths. Preserve both relative-attention terms, '
                    'normalized relative embeddings, and separately rounded Python GELU.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [164480]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'siglip': {
        'reference': {
            'config_class': 'transformers:SiglipConfig',
            'model_class': 'transformers:SiglipModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/siglip-base-patch16-224',
                'revision': '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed',
                'url': 'https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/config.json',
                'description': ('Pinned SiglipModel.forward names this paired text/image checkpoint. Preserve both '
                    'complete towers, learned attention pooling, normalized embeddings, and learned '
                    'contrastive scale/bias. Text attention is bidirectional and pooling selects the last '
                    'token.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 1, 'sequence_length': 64,
            'shape': [3, 224, 224], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'siglip2': {
        'reference': {
            'config_class': 'transformers:Siglip2Config',
            'model_class': 'transformers:Siglip2Model',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/siglip2/configuration_siglip2.py',
                'description': ('Pinned Siglip2Config example constructs paired Siglip2Model with defaults; retain '
                    'NaFlex patch positions, masked vision attention, attention pooling and both '
                    'similarities.'),
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 1, 'sequence_length': 64,
            'shape': [256, 768], 'spatial_shapes': [[14, 18]], 'batch_size': 1,
        },
        'outputs': [
            'logits_per_text', 'logits_per_image', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'slanet': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:SLANetConfig',
            'model_class': 'transformers:SLANetForTableRecognition',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'PaddlePaddle/SLANet_plus_safetensors',
                'revision': 'e44d12b3fe695792170dd46623075f63d8544461',
                'url': 'https://huggingface.co/PaddlePaddle/SLANet_plus_safetensors/blob/e44d12b3fe695792170dd46623075f63d8544461/config.json',
                'description': 'Pinned HF table-recognition task checkpoint with its native processor image size and full greedy recurrent structure decoding.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 488, 488]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'slanext': {
        'reference': {
            'config_class': 'transformers:SLANeXtConfig',
            'model_class': 'transformers:SLANeXtForTableRecognition',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'PaddlePaddle/SLANeXt_wired_safetensors',
                'revision': 'd97b77b5078951c313807747f77ec07beab07e4e',
                'url': 'https://huggingface.co/PaddlePaddle/SLANeXt_wired_safetensors/blob/d97b77b5078951c313807747f77ec07beab07e4e/config.json',
                'description': 'Pinned HF table-recognition task checkpoint with its native processor image size and full greedy recurrent structure decoding.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 512, 512]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'smollm3': {
        'reference': {
            'config_class': 'transformers:SmolLM3Config',
            'model_class': 'transformers:SmolLM3ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'HuggingFaceTB/SmolLM3-3B',
                'revision': 'a07cc9a04f16550a088caea529712d1d335b0ac1',
                'url': 'https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/config.json',
                'description': 'Pinned HF model documentation causal-LM example checkpoint; preserve its computational settings.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 193},
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'smolvlm': {
        'reference': {
            'config_class': 'transformers:SmolVLMConfig',
            'model_class': 'transformers:SmolVLMForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
                'revision': '482adb537c021c86670beed01cd58990d01e72e4',
                'description': ('Pinned public video-conditioned text generation; native processor represents frames as'
                    ' image blocks.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/smolvlm/modeling_smolvlm.py',
            },
            'generation_config': {
                '_from_model_config': True, 'bos_token_id': 0, 'eos_token_id': 49279, 'pad_token_id': 2,
                'transformers_version': '4.47.1',
            },
        },
        'input': {'kind': 'external', 'shape': [2, 3, 384, 384], 'sequence_length': 235, 'batch_size': 1},
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 4, 'do_sample': False},
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'reference_backend': None,
    },

    'solar_open': {
        'reference': {
            'config_class': 'transformers:SolarOpenConfig',
            'model_class': 'transformers:SolarOpenForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'upstage/Solar-Open-100B',
                'revision': '1f591439b14055004d1a5d1a975608953a022fea',
                'url': 'https://huggingface.co/upstage/Solar-Open-100B/blob/1f591439b14055004d1a5d1a975608953a022fea/config.json',
                'description': 'Pinned HF task documentation official checkpoint; preserve enabled computation.',
            },
        },
        'dimension_overrides': {
            'hidden_size': 256, 'intermediate_size': 256, 'moe_intermediate_size': 128,
            'num_hidden_layers': 3, 'num_attention_heads': 8, 'num_key_value_heads': 1, 'head_dim': 64,
            'vocab_size': 1024,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Retain attention width2x hidden/8:1 GQA, every layer routed with128 experts/top8/shared1; '
            'full-head YaRN factor2 and native context boundaries. Reduce layer/model/expert intermediate '
            'widths.'
            ' Keep the prompt and first continuation; add a second continuation and compare all logical caches.'),
    },

    'speech_encoder_decoder': {
        'reference': {
            'config_class': 'transformers:SpeechEncoderDecoderConfig',
            'model_class': 'transformers:SpeechEncoderDecoderModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/wav2vec2-xls-r-300m-en-to-15',
                'revision': '929af1e6ad84856a39e90b8ac56d7339bcf1985a',
                'url': 'https://huggingface.co/facebook/wav2vec2-xls-r-300m-en-to-15/blob/929af1e6ad84856a39e90b8ac56d7339bcf1985a/config.json',
                'description': ('Pinned model_doc/speech-encoder-decoder.md inference checkpoint selects '
                    'Wav2Vec2+MBART, not old audit BERT decoder. Preserve seven convolution frontend '
                    'stages, all three strided GLU adapters, pre-norm encoder, ReLU decoder, scaled tied '
                    'embeddings, cross-attention and all default caches.'),
            },
            'native_cache_defaults': True,
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [32000], 'decoder_sequence_length': 33},
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'speech_to_text': {
        'reference': {
            'config_class': 'transformers:Speech2TextConfig',
            'model_class': 'transformers:Speech2TextForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/s2t-small-librispeech-asr',
                'revision': '256e91a603c013ae09aa1e4c09cfc3f8acdf8d19',
                'description': ('Pinned public Speech2Text conditional-generation ASR example uses '
                    'facebook/s2t-small-librispeech-asr. Retain both stride2 GLU convolutions, scaled '
                    'embeddings, sinusoidal positions and final norms.'),
            },
        },
        'input': {
            'kind': 'spectrogram', 'name': 'input_features', 'shape': [401, 80], 'batch_size': 1,
            'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'speecht5': {
        'reference': {
            'config_class': 'transformers:SpeechT5Config',
            'model_class': 'transformers:SpeechT5ForSpeechToText',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/speecht5_asr',
                'revision': '53615c10408485422e09a12cda191a747f4bbe34',
                'description': 'Published ASR configuration; preserve waveform frontend, post-norm relative encoder and cached text decoder.',
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'waveform', 'name': 'input_values', 'batch_size': 1, 'shape': [55200],
            'attention_mask_length': 55200, 'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'splinter': {
        'reference': {
            'config_class': 'transformers:SplinterConfig',
            'model_class': 'transformers:SplinterModel',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Preserve the bare SplinterModel task without a question-selection head. Use the '
                    'official tau/splinter-base checkpoint named in the pinned configuration documentation;'
                    ' its constructor example separately uses defaults.'),
                'checkpoint': 'tau/splinter-base',
                'revision': 'd6bc929405a27b7502bbab767f615c89b0e52373',
                'url': 'https://huggingface.co/tau/splinter-base/blob/d6bc929405a27b7502bbab767f615c89b0e52373/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'squeezebert': {
        'reference': {
            'config_class': 'transformers:SqueezeBertConfig',
            'model_class': 'transformers:SqueezeBertForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Preserve SqueezeBertForMaskedLM with the checkpoint named by pinned SqueezeBertConfig;'
                    ' retain all grouped convolutions and the pooler computed by its base transformer.'),
                'checkpoint': 'squeezebert/squeezebert-uncased',
                'revision': '7978b0c163f11850ec35d5cd541828159313ac41',
                'url': 'https://huggingface.co/squeezebert/squeezebert-uncased/blob/7978b0c163f11850ec35d5cd541828159313ac41/config.json',
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'stablelm': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:StableLmConfig',
            'model_class': 'transformers:StableLmForCausalLM',
            'forward_kwargs': {'logits_to_keep': 0},
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'stabilityai/stablelm-3b-4e1t',
                'revision': 'fa4a6a92fca83c3b4223a3c9bf792887090ebfba',
                'url': 'https://huggingface.co/stabilityai/stablelm-3b-4e1t/resolve/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/config.json',
                'description': ('The causal-LM example resolves to model_type=persimmon, a different architecture; '
                    'preserve the StableLM public task and use its pinned configuration-documentation '
                    'checkpoint.'),
                'causal_lm_example_evidence': {
                    'checkpoint': 'adept/persimmon-8b-base',
                    'revision': '94dc4e0bb7eeb26ec521eb3f78c36c91f6fe866b', 'model_type': 'persimmon',
                    'accessible': True,
                },
                'source_locations': [
                    'src/transformers/models/stablelm/configuration_stablelm.py:StableLmConfig',
                    'src/transformers/models/stablelm/modeling_stablelm.py:StableLmForCausalLM.forward',
                ],
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'outputs': ['logits', 'past_key_values'],
    },

    'starcoder2': {
        'reference': {
            'config_class': 'transformers:Starcoder2Config',
            'model_class': 'transformers:Starcoder2ForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'bigcode/starcoder2-7b',
                'revision': 'bb9afde76d7945da5745592525db122d4d729eb1',
                'url': 'https://huggingface.co/bigcode/starcoder2-7b/blob/bb9afde76d7945da5745592525db122d4d729eb1/config.json',
                'description': 'Pinned HF documentation example checkpoint.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 4099},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'superglue': {
        'reference': {
            'config_class': 'transformers:SuperGlueConfig',
            'model_class': 'transformers:SuperGlueForKeypointMatching',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'magic-leap-community/superglue_outdoor',
                'revision': 'f4041f88aa6789c46558efaafc98316c6b58a382',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/superglue/modeling_superglue.py',
                'description': 'Pinned native public image-pair matching example, including the invoked SuperPoint detector.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [2, 3, 480, 640]},
        'outputs': ['matches', 'matching_scores', 'keypoints', 'mask'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'superpoint': {
        'reference': {
            'config_class': 'transformers:SuperPointConfig',
            'model_class': 'transformers:SuperPointForKeypointDetection',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'magic-leap-community/superpoint',
                'revision': '734450e9ffe229074f5998494ddc615475cdb20a',
                'description': 'Pinned public task example checkpoint and processor.',
            },
        },
        'input': {'kind': 'image', 'shape': [3, 480, 640], 'batch_size': 1},
        'outputs': ['keypoints', 'scores', 'descriptors', 'mask'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'swiftformer': {
        'reference': {
            'config_class': 'transformers.models.swiftformer.configuration_swiftformer:SwiftFormerConfig',
            'model_class': 'transformers.models.swiftformer.modeling_swiftformer:SwiftFormerModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned public SwiftFormerConfig example constructs '
                    'SwiftFormerModel(SwiftFormerConfig()). Retain its actual singleton-axis attention '
                    'softmax, all four stages and layer scales.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/swiftformer/configuration_swiftformer.py#L46',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'swin': {
        'reference': {
            'config_class': 'transformers:SwinConfig',
            'model_class': 'transformers:SwinModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/swin/configuration_swin.py',
                'description': ('The pinned SwinConfig public example explicitly constructs SwinModel(SwinConfig()) '
                    'with random weights. Use complete constructor computational defaults.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'swin2sr': {
        'reference': {
            'config_class': 'transformers:Swin2SRConfig',
            'model_class': 'transformers:Swin2SRModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/swin2sr/configuration_swin2sr.py#L45-L55',
                'description': ('The pinned base-model example constructs Swin2SRConfig() and '
                    'Swin2SRModel(configuration); no superresolution head.'),
            },
        },
        'input': {'kind': 'image', 'shape': [3, 64, 64], 'batch_size': 1},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'swinv2': {
        'reference': {
            'config_class': 'transformers:Swinv2Config',
            'model_class': 'transformers:Swinv2Model',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/swinv2/configuration_swinv2.py',
                'description': ('The pinned Swinv2Config public example explicitly constructs '
                    'Swinv2Model(Swinv2Config()) with random weights. Use complete constructor '
                    'computational defaults. The example comment names a window8/256 checkpoint, but '
                    'executed constructor defaults are window7/image224; follow the executable example '
                    'without loading that checkpoint.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'switch_transformers': {
        'reference': {
            'config_class': 'transformers:SwitchTransformersConfig',
            'model_class': 'transformers:SwitchTransformersForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/switch-base-8',
                'revision': '92fe2d22b024d9937146fe097ba3d3a7ba146e1b',
                'description': ('Pinned Switch conditional-generation example checkpoint; preserve FP32 router, BF16 '
                    'probabilities, alternating sparse layers and eight experts.'),
            },
            'native_cache_defaults': True,
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    't5': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:T5Config',
            'model_class': 'transformers:T5ForConditionalGeneration',
            'forward_kwargs': {},
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'google-t5/t5-small',
                'revision': 'df1b051c49625cf57a3d0d8d3863ed4d13564fe4',
                'description': 'Pinned conditional-generation inference example checkpoint.',
            },
        },
        'config_overrides': {},
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'table_transformer': {
        'reference': {
            'config_class': 'transformers:TableTransformerConfig',
            'model_class': 'transformers:TableTransformerForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/table-transformer-detection',
                'revision': '2357cbe2b5a5d1c03e54f32764f06058933b65ab',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/table_transformer/modeling_table_transformer.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 600, 800]},
        'outputs': ['logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'tapas': {
        'reference': {
            'config_class': 'transformers:TapasConfig',
            'model_class': 'transformers:TapasModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'google/tapas-base',
                'revision': '00456266840bb0a319cd6748ebf7da3caf98816b',
                'url': 'https://huggingface.co/google/tapas-base/blob/00456266840bb0a319cd6748ebf7da3caf98816b/config.json',
                'description': 'Pinned HF public base-model forward example checkpoint, retaining ordinary document/table metadata.',
            },
        },
        'input': {
            'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512, 'table_columns': 3, 'query_length': 8,
            'tokens_per_cell': 3,
        },
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'textnet': {
        'reference': {
            'config_class': 'transformers:TextNetConfig',
            'model_class': 'transformers:TextNetModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'czczup/textnet-base',
                'revision': '6d2531fd3c7cb367bbfd570ca94ea7a0784b3099',
                'description': 'Pinned HF documented checkpoint; original audit public base-model task retained.',
            },
        },
        'input': {'kind': 'image', 'shape': [3, 640, 640], 'batch_size': 1},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'timesfm': {
        'reference': {
            'config_class': 'transformers:TimesFmConfig',
            'model_class': 'transformers:TimesFmModelForPrediction',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/timesfm-2.0-500m-pytorch',
                'revision': 'dc2443792ce5516872b89b37cf1bc058c3bf0c10',
                'description': ('Pinned public prediction example, variable-length series and three frequency buckets; '
                    'preserve all quantile heads and disabled optional decomposition/context '
                    'forecasts/truncation.'),
            },
        },
        'input': {'kind': 'timeseries_list', 'lengths': [100, 200, 400], 'frequencies': [0, 1, 2]},
        'workload': 'forward',
        'outputs': ['mean_predictions', 'full_predictions', 'last_hidden_state'],
        'reference_backend': None,
    },

    'timesfm2_5': {
        'reference': {
            'config_class': 'transformers:TimesFm2_5Config',
            'model_class': 'transformers:TimesFm2_5ModelForPrediction',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/timesfm-2.5-200m-transformers',
                'revision': '5a9806b9b291fad9233b5249d88263f1846304d3',
                'description': ('Pinned TimesFM2.5 documented prediction checkpoint. Preserve default flip invariance, '
                    'continuous quantile head and input-dependent nonnegative clipping.'),
            },
        },
        'input': {'kind': 'timeseries_list', 'lengths': [100, 200, 400]},
        'workload': 'forward',
        'outputs': ['mean_predictions', 'full_predictions', 'last_hidden_state'],
        'reference_backend': None,
    },

    'timesformer': {
        'reference': {
            'config_class': 'transformers:TimesformerConfig',
            'model_class': 'transformers:TimesformerModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/timesformer/configuration_timesformer.py',
                'description': ('Pinned HF configuration documentation explicitly constructs this public task with '
                    'constructor defaults.'),
            },
        },
        'input': {'kind': 'video', 'batch_size': 1, 'shape': [8, 3, 224, 224]},
        'outputs': ['last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'timm_backbone': {
        'reference': {
            'config_class': 'transformers:TimmBackboneConfig',
            'model_class': 'transformers:TimmBackbone',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/timm_backbone/configuration_timm_backbone.py#L40',
                'description': ('Pinned public task example constructs this backbone using constructor defaults; '
                    'TimmBackboneConfig example explicitly selects resnet50.'),
            },
            'load_with_base_class': True,
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['feature_maps'],
        'workload': 'forward',
        'config_overrides': {'backbone': 'resnet50'},
        'reference_backend': None,
    },

    'timm_wrapper': {
        'reference': {
            'config_class': 'transformers:TimmWrapperConfig',
            'model_class': 'transformers:TimmWrapperModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'timm/resnet50.a1_in1k',
                'revision': '767268603ca0cb0bfe326fa87277f19c419566ef',
                'url': 'https://huggingface.co/timm/resnet50.a1_in1k/blob/767268603ca0cb0bfe326fa87277f19c419566ef/config.json',
                'description': ('Pinned TimmWrapperModel.forward example explicitly selects timm/resnet50.a1_in1k. This'
                    ' public task example takes precedence over config-class example resnet18; witness is '
                    'this concrete ResNet50, not all timm models.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'outputs': ['last_hidden_state', 'pooler_output'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'trocr': {
        'reference': {
            'config_class': 'transformers:TrOCRConfig',
            'model_class': 'transformers:TrOCRForCausalLM',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('First pinned TrOCRForCausalLM forward example explicitly constructs TrOCRConfig() as a'
                    ' VisionEncoderDecoder decoder; preserve cross-attention to supplied projected vision '
                    'states.'),
            },
            'continuation_input_names': ['encoder_hidden_states'],
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 66, 'encoder_sequence_length': 197},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'past_key_values'],
        'reference_backend': None,
    },

    'tvp': {
        'reference': {
            'config_class': 'transformers:TvpConfig',
            'model_class': 'transformers:TvpForVideoGrounding',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Intel/tvp-base',
                'revision': '22c974b53227d1a874a8392f139d5d3d720c3ee8',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/tvp.md#L127-L142',
                'description': 'Documented video grounding example, retaining all 48 frames and temporal averaging.',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 100,
            'shape': [48, 3, 448, 448], 'attention_mask': True, 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'udop': {
        'reference': {
            'config_class': 'transformers:UdopConfig',
            'model_class': 'transformers:UdopForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/udop-large',
                'revision': '2d5f5b18aa2a3ad6b551933582ca49265f45afcf',
                'url': 'https://huggingface.co/microsoft/udop-large/blob/2d5f5b18aa2a3ad6b551933582ca49265f45afcf/config.json',
                'description': ('Pinned conditional-generation task processor path: OCR text, documented 0..1000 OCR '
                    'boxes, image, attention mask, decoder tokens.'),
            },
            'encoder_input_names': ['input_ids', 'bbox', 'pixel_values', 'attention_mask'],
        },
        'input': {
            'kind': 'document', 'batch_size': 2, 'sequence_length': 193, 'normalized_bbox': False,
            'image_shape': [3, 224, 224], 'decoder_length': 139, 'attention_mask': True,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'umt5': {
        'workload': 'seq2seq_continuation',
        'reference_backend': None,
        'reference': {
            'config_class': 'transformers:UMT5Config',
            'model_class': 'transformers:UMT5ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'google/umt5-small',
                'revision': '8c63c2b77efbf8e41206a2c8d994846cc9392360',
                'description': ('Pinned conditional-generation example checkpoint. Native pinned configuration ties '
                    'embeddings despite the serialized false field.'),
            },
        },
        'input': {
            'kind': 'seq2seq_tokens', 'batch_size': 1, 'encoder_sequence_length': 193,
            'decoder_sequence_length': 139, 'encoder_prefix_token_ids': [], 'encoder_suffix_token_ids': [1],
            'content_token_id_min': 3,
        },
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
    },

    'unispeech': {
        'reference': {
            'config_class': 'transformers:UniSpeechConfig',
            'model_class': 'transformers:UniSpeechModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/unispeech/configuration_unispeech.py#L121',
                'description': ('Preserve constructor-example defaults: group normalization in the first frontend '
                    'convolution and a post-normalized encoder. The example names a style but explicitly '
                    'constructs UniSpeechConfig rather than loading checkpoint settings.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [48000]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'extract_features'],
        'reference_backend': None,
    },

    'unispeech_sat': {
        'reference': {
            'config_class': 'transformers:UniSpeechSatConfig',
            'model_class': 'transformers:UniSpeechSatModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/unispeech_sat/configuration_unispeech_sat.py#L131',
                'description': ('Preserve constructor-example defaults: group normalization in the first frontend '
                    'convolution and a post-normalized encoder; include the unconditional learned mask '
                    'parameter.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [48000]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'extract_features'],
        'reference_backend': None,
    },

    'univnet': {
        'reference': {
            'config_class': 'transformers:UnivNetConfig',
            'model_class': 'transformers:UnivNetModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'dg845/univnet-dev',
                'revision': 'fbfdd9ac17e7708deb785156e18131e7aee10ee3',
                'description': 'Native documented UnivNet waveform task, explicit standard Gaussian noise_sequence accepted by native API.',
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'continuous', 'name': 'input_features', 'shape': [17, 100], 'batch_size': 1,
            'noise_sequence_shape': [1, 17, 64],
        },
        'workload': 'forward',
        'outputs': ['waveforms'],
    },

    'upernet': {
        'reference': {
            'config_class': 'transformers:UperNetConfig',
            'model_class': 'transformers:UperNetForSemanticSegmentation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openmmlab/upernet-convnext-tiny',
                'revision': '876ffc56b819a829448f7e81e9f8606deef6fb65',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/upernet/modeling_upernet.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 512, 512]},
        'outputs': ['logits'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'uvdoc': {
        'reference': {
            'config_class': 'transformers:UVDocConfig',
            'model_class': 'transformers:UVDocModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'PaddlePaddle/UVDoc_safetensors',
                'revision': 'bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4',
                'description': 'HF documented PaddlePaddle checkpoint with all six selected parallel bridge branches.',
            },
        },
        'input': {'kind': 'image', 'shape': [3, 712, 488], 'batch_size': 1},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'vibevoice_acoustic_tokenizer': {
        'reference': {
            'config_class': 'transformers:VibeVoiceAcousticTokenizerConfig',
            'model_class': 'transformers:VibeVoiceAcousticTokenizerModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'microsoft/VibeVoice-AcousticTokenizer',
                'revision': 'f0643be7cab5b9c91c6f0038e3b8dde13e3e107e',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/vibevoice_acoustic_tokenizer.md#L54-L84',
                'description': ('Pinned standalone tokenizer basic example selects '
                    'microsoft/VibeVoice-AcousticTokenizer and explicitly uses deterministic '
                    'encode(sample=False) then decode. Full forward with sample=False preserves both '
                    'returned latents and audio.'),
            },
            'forward_kwargs': {'sample': False},
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [1, 224000]},
        'workload': 'forward',
        'outputs': ['audio', 'latents'],
        'reference_backend': None,
    },

    'vibevoice_asr': {
        'reference': {
            'config_class': 'transformers:VibeVoiceAsrConfig',
            'model_class': 'transformers:VibeVoiceAsrForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/VibeVoice-ASR-HF',
                'revision': 'f22241c2062b3b25272bf117397e03d73381037a',
                'description': 'Pinned native VibeVoice ASR example; complete dual encoder and greedy transcription generation.',
                'url': 'https://huggingface.co/microsoft/VibeVoice-ASR-HF',
            },
            'generation_config': {
                '_from_model_config': True, 'do_sample': False, 'eos_token_id': 151643, 'max_length': 32768,
                'max_new_tokens': 32768, 'output_attentions': False, 'output_hidden_states': False,
                'pad_token_id': 151655, 'transformers_version': '5.3.0.dev0', 'use_cache': True,
            },
        },
        'reference_backend': {
            '': 'sdpa', 'acoustic_tokenizer_encoder_config': 'eager',
            'semantic_tokenizer_encoder_config': 'eager', 'text_config': 'sdpa',
        },
        'dimension_overrides': {
            'acoustic_tokenizer_encoder_config': {'depths': [1, 1, 1, 1, 1, 1, 1], 'num_filters': 2, 'hidden_size': 16},
            'semantic_tokenizer_encoder_config': {'depths': [1, 1, 1, 1, 1, 1, 1], 'num_filters': 2, 'hidden_size': 32},
            'text_config': {
                'hidden_size': 112, 'intermediate_size': 256, 'num_hidden_layers': 2,
                'num_attention_heads': 7, 'num_key_value_heads': 1,
                'layer_types': ['full_attention', 'full_attention'], 'max_position_embeddings': 1024,
            },
        },
        'input': {'kind': 'text', 'batch_size': 1, 'sequence_length': 1},
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 3},
        'generation_seed': 29,
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'dimension_purpose': ('All7 convolutional stages/six native downsample ratios, both encoders and projector branches, '
            'native two-draw VAE sampling,7:1 textGQA retained. Input exceeds original1440000-sample chunk '
            'boundary by2 native3200-sample hops, retaining convolution caches across chunks;3 greedy text '
            'steps with decoder cache. Width/depth reductions only; original specialIDs/vocabulary '
            'preserved.'),
    },

    'video_llama_3': {
        'reference': {
            'config_class': 'transformers:VideoLlama3Config',
            'model_class': 'transformers:VideoLlama3ForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'lkhl/VideoLLaMA3-2B-Image-HF',
                'revision': '8694f8ccf5df92ac306ed46149e1e96cb319d90e',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/video_llama_3.md',
                'description': ('Pinned model_doc/video_llama_3.md documents image and video using this checkpoint. '
                    'Preserve variable-grid2DvisionRoPE, per-frame attention, checkpoint image merge1 and '
                    'video merge2 bilinear resizing, projector and tied Qwen2 decoder. Public processor '
                    'compression mask retained as input.'),
            },
            'prefill_input_names': [
                'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'image_merge_sizes',
                'video_grid_thw', 'video_merge_sizes', 'video_compression_mask',
            ],
        },
        'dimension_overrides': {
            'vision_config': {'hidden_size': 192, 'intermediate_size': 768, 'num_hidden_layers': 2, 'num_attention_heads': 3},
            'text_config': {
                'hidden_size': 384, 'intermediate_size': 1536, 'num_hidden_layers': 2,
                'num_attention_heads': 6, 'num_key_value_heads': 1,
                'layer_types': ['full_attention', 'full_attention'],
            },
        },
        'input': {
            'kind': 'text_image', 'image_batch_size': 1, 'text_batch_size': 1, 'shape': [24, 588],
            'sequence_length': 74, 'image_token_positions': list(range(3, 27)), 'video_shape': [48, 588],
            'video_token_positions': list(range(35, 47)), 'flatten_pixel_batch': True,
            'image_grid_thw': [[1, 4, 6]], 'image_merge_sizes': [1], 'video_grid_thw': [[2, 4, 6]],
            'video_merge_sizes': [2],
            'video_compression_mask': [True, True, True, True, True, True, True, True, True, True, True, True],
        },
        'workload': 'causal_lm',
        'outputs': ['logits', 'image_hidden_states', 'video_hidden_states'],
        'decode_outputs': ['logits'],
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Rectangular4x6patch grids retain2Dpositions. Checkpoint image merge1 retains24features and '
            'executes same-sizebilinear; video merge2 '
            'rearranges/bilinear4x6to2x3,2videoframes48patches→12features, independentperframeattention. '
            'Defaultcompression maskalltrue for generated changingframes, independentlychecked '
            'bynativeprocessor. Native6:1QwenGQA,tiedhead,vocab retained; two layers and cacheddecode.'),
    },

    'video_llava': {
        'reference': {
            'native_cache_defaults': True,
            'native_position_ids': True,
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:VideoLlavaConfig',
            'model_class': 'transformers:VideoLlavaForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'LanguageBind/Video-LLaVA-7B-hf',
                'revision': '4cf9d8cfc76a54f46a4cb43be5368b46b7f0d736',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/docs/source/en/model_doc/video_llava.md#L125-L140',
                'description': ('Pinned HF model_doc/video_llava.md:125-140 simultaneous image and video inference '
                    'example. Retain separate towers, default penultimate features, image CLS removal and '
                    'video CLS retention, shared GELU projector and Llama conditional decoder.'),
            },
            'prefill_input_names': ['pixel_values_images', 'pixel_values_videos'],
        },
        'input': {
            'kind': 'text_image', 'image_batch_size': 1, 'text_batch_size': 1, 'shape': [3, 224, 224],
            'sequence_length': 2378, 'image_token_positions': list(range(3, 259)),
            'image_input_name': 'pixel_values_images', 'video_shape': [8, 3, 224, 224],
            'video_token_positions': list(range(265, 2321)), 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'video_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'videomae': {
        'reference': {
            'config_class': 'transformers:VideoMAEConfig',
            'model_class': 'transformers:VideoMAEModel',
            'source': {
                'checkpoint': 'MCG-NJU/videomae-base',
                'revision': 'dc740ceda42fce44faed2ea03c6d447db72f6af9',
                'url': 'https://huggingface.co/MCG-NJU/videomae-base/resolve/dc740ceda42fce44faed2ea03c6d447db72f6af9/config.json',
                'kind': 'example_checkpoint',
                'description': ('The pinned base-model forward example loads this checkpoint without a mask. Its '
                    'use_mean_pooling=False setting retains final normalization; reconstruction is outside '
                    'this public class.'),
            },
        },
        'input': {'kind': 'video', 'batch_size': 1, 'shape': [16, 3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
        'dimension_overrides': {'hidden_size': 256, 'num_attention_heads': 4, 'num_hidden_layers': 2, 'intermediate_size': 1024},
        'dimension_purpose': ('Two encoder blocks; 64-wide heads and 4x feed-forward width; all 16 frames at 224x224 and '
            '2x16x16 tubelets retained. Random video tests computation, not tracking quality.'),
        'configuration_scope': 'reviewed_scaled_v1',
    },

    'videomt': {
        'reference': {
            'config_class': 'transformers:VideomtConfig',
            'model_class': 'transformers:VideomtForUniversalSegmentation',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'tue-mps/videomt-dinov2-small-ytvis2019',
                'revision': 'a8c9855154c4825bbdc738e5496840cbfa48987c',
                'description': 'Pinned HF universal video segmentation task example.',
            },
        },
        'input': {'kind': 'continuous', 'name': 'pixel_values_videos', 'shape': [3, 3, 640, 640], 'batch_size': 1},
        'outputs': ['masks_queries_logits', 'class_queries_logits', 'last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'vilt': {
        'reference': {
            'config_class': 'transformers:ViltConfig',
            'model_class': 'transformers:ViltForImageAndTextRetrieval',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'dandelin/vilt-b32-finetuned-coco',
                'revision': '2f3f7f3f62a3f4f429c26309256485bbd9b0e40a',
                'description': ('Pinned public image/text retrieval task and matching documented checkpoint. Native '
                    'reference without old deterministic sampling patch. Complete rectangular images with '
                    'omitted all-valid pixel masks.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/vilt/modeling_vilt.py#L958',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 37,
            'shape': [3, 384, 512], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'vipllava': {
        'reference': {
            'native_cache_defaults': True,
            'native_position_ids': True,
            'continuation_outputs': ['logits', 'past_key_values'],
            'config_class': 'transformers:VipLlavaConfig',
            'model_class': 'transformers:VipLlavaForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'llava-hf/vip-llava-7b-hf',
                'revision': 'd060fe36b7e550f6884342432561ea6afaea8482',
                'url': 'https://huggingface.co/llava-hf/vip-llava-7b-hf/blob/d060fe36b7e550f6884342432561ea6afaea8482/config.json',
                'description': ('Pinned public VipLLaVA conditional generation example. Preserve constructor default '
                    'feature layers [-2,-5,-8,-11,6] since official checkpoint does not override them; '
                    'concatenated multi-layer CLIP patches, projector LayerNorm, GELU projection, ordinary '
                    'Llama decoder.'),
            },
            'prefill_input_names': ['pixel_values'],
        },
        'input': {
            'kind': 'text_image', 'image_batch_size': 1, 'text_batch_size': 1, 'shape': [3, 336, 336],
            'sequence_length': 642, 'image_token_positions': list(range(3, 579)), 'batch_size': 1,
        },
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'image_hidden_states', 'past_key_values'],
        'reference_backend': None,
    },

    'vision_encoder_decoder': {
        'reference': {
            'config_class': 'transformers:VisionEncoderDecoderConfig',
            'model_class': 'transformers:VisionEncoderDecoderModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'nlpconnect/vit-gpt2-image-captioning',
                'revision': 'dc68f91c06a1ba6f15268e5b9c13ae7a7c514084',
                'url': 'https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/blob/dc68f91c06a1ba6f15268e5b9c13ae7a7c514084/config.json',
                'description': ('Pinned model_doc/vision-encoder-decoder.md inference example selects this ViT/GPT2 '
                    'captioning checkpoint. Preserve same-width towers, encoder pooler computation, enabled'
                    ' decoder cross-attention, GELU-new, tied vocabulary output and default cache; native '
                    'vocab50257 retains specialtokens.'),
            },
            'native_cache_defaults': True,
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224], 'decoder_sequence_length': 33},
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'vision_text_dual_encoder': {
        'reference': {
            'config_class': 'transformers:VisionTextDualEncoderConfig',
            'model_class': 'transformers:VisionTextDualEncoderModel',
            'source': {
                'kind': 'composite_checkpoints',
                'components': {
                    'vision_config': {
                        'checkpoint': 'google/vit-base-patch16-224',
                        'revision': '3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3',
                    },
                    'text_config': {
                        'checkpoint': 'google-bert/bert-base-uncased',
                        'revision': '86b5e0934494bd15c9632b12f734a8a67f723594',
                    },
                },
                'description': ('Pinned modeling_vision_text_dual_encoder.py forward example: ViT base patch16 + BERT '
                    'base uncased, paired inference with both poolers. Wrapper has no universal tower '
                    'default; this evaluates the explicit documented pair.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py#L191-L240',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 2, 'image_batch_size': 2, 'sequence_length': 512,
            'shape': [3, 224, 224], 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_image', 'logits_per_text', 'text_embeds', 'image_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'visual_bert': {
        'reference': {
            'config_class': 'transformers:VisualBertConfig',
            'model_class': 'transformers:VisualBertModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'uclanlp/visualbert-vqa-coco-pre',
                'revision': '884aaef1fb6bed1429cae8c3abc314011a3a429f',
                'description': ('Pinned public base-model forward example selects this checkpoint and includes '
                    'externally supplied visual features.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/visual_bert/modeling_visual_bert.py',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 37, 'visual_sequence_length': 9},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'vit': {
        'reference': {
            'config_class': 'transformers.models.vit.configuration_vit:ViTConfig',
            'model_class': 'transformers.models.vit.modeling_vit:ViTModel',
            'source': {
                'kind': 'constructor_defaults',
                'description': ('The pinned public configuration example constructs ViTModel(ViTConfig()) with random '
                    'weights. This base-model task uses constructor computational defaults; checkpoint '
                    'configuration is not loaded.'),
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/vit/configuration_vit.py#L33-L46',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
        'dimension_overrides': {
            'hidden_size': 256, 'num_attention_heads': 4, 'num_hidden_layers': 2, 'intermediate_size': 1024,
            'pooler_output_size': 256,
        },
        'dimension_purpose': ('Two encoder blocks; 64-wide heads and 4x feed-forward width; original 224x224 image and 16x16 '
            'patches; learned pooler remains checked.'),
        'configuration_scope': 'reviewed_scaled_v1',
    },

    'vit_mae': {
        'reference': {
            'config_class': 'transformers:ViTMAEConfig',
            'model_class': 'transformers:ViTMAEForPreTraining',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/vit-mae-base',
                'revision': '25b184bea5538bf5c4c852c79d221195fdd2778d',
                'url': 'https://huggingface.co/facebook/vit-mae-base/blob/25b184bea5538bf5c4c852c79d221195fdd2778d/config.json',
                'description': ('Pinned ViTMAEForPreTraining.forward example explicitly loads facebook/vit-mae-base '
                    '(modeling_vit_mae.py910); preserve original public pretraining task including '
                    'unconditional loss. Config doc constructor example is for the different base encoder '
                    'class.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224], 'noise_shape': [1, 196]},
        'outputs': ['loss', 'logits', 'mask', 'ids_restore'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'vit_msn': {
        'reference': {
            'config_class': 'transformers.models.vit_msn.configuration_vit_msn:ViTMSNConfig',
            'model_class': 'transformers.models.vit_msn.modeling_vit_msn:ViTMSNModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/vit-msn-small',
                'revision': 'a50267de5e56f5b66a7fec6d2d8a8c60db24a704',
                'hf_source_revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': ('The pinned ViTMSNModel.forward example names facebook/vit-msn-small. Configuration '
                    'only; weights are initialized by the pinned HF model.'),
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/vit_msn/modeling_vit_msn.py#L441-L459',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'vitdet': {
        'reference': {
            'config_class': 'transformers.models.vitdet.configuration_vitdet:VitDetConfig',
            'model_class': 'transformers.models.vitdet.modeling_vitdet:VitDetModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'description': 'The pinned public VitDetModel example constructs VitDetConfig().',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/vitdet/modeling_vitdet.py#L635',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state'],
        'reference_backend': None,
    },

    'vits': {'reference': {'config_class': 'transformers:VitsConfig',
                        'model_class': 'transformers:VitsModel',
                        'source': {'kind': 'example_checkpoint',
                                   'checkpoint': 'facebook/mms-tts-eng',
                                   'revision': 'c71de0fe7204c83f1c10820a7d696d0b450048ba',
                                   'url': 'https://huggingface.co/facebook/mms-tts-eng/blob/c71de0fe7204c83f1c10820a7d696d0b450048ba/config.json',
                                   'description': 'Pinned VitsModel.forward text-to-speech example selects '
                                                  'facebook/mms-tts-eng; retain single-speaker stochastic '
                                                  'duration and nonzero native noise.'}},
          'dimension_overrides': {'num_hidden_layers': 2, 'upsample_initial_channel': 64},
          'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 13},
          'workload': 'forward',
          'outputs': ['waveform', 'sequence_lengths', 'spectrogram'],
          'reference_backend': 'eager',
          'generation_seed': 555,
          'dimension_purpose': 'Reduce repeated text layers6 to2 and decoder initial channels512 to64. '
                               'Preserve native hidden192, head96, FF768, flow192, vocabulary38, all4 '
                               'duration-flow definitions (native inference skips the first convolutional '
                               'flow), all4 prior flows with4 WaveNet layers, all3 depthwise dilations, '
                               'all10 spline bins, native window4 and all4 decoder upsample stages8/8/2/2 '
                               'with3 residual kernels3/7/11 and dilations1/3/5. Thirteen tokens exceed '
                               'both sides of relative window4 and exercise noncentral dilated kernels; '
                               'stochastic generated lengths retain dynamic alignment. Native '
                               'duration/prior noise scales remain0.8/0.667. Seed555 fixes common draws, '
                               'including typed/layout-matched prior sampling; no noise suppression.'},
    'vitmatte': {
        'reference': {
            'config_class': 'transformers:VitMatteConfig',
            'model_class': 'transformers:VitMatteForImageMatting',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'hustvl/vitmatte-small-composition-1k',
                'revision': '6a58ad7646403c1df626fbd746900aec7361ea1d',
                'description': 'Pinned HF public task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'shape': [4, 640, 960], 'batch_size': 1},
        'outputs': ['alphas'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'vitpose': {
        'reference': {
            'config_class': 'transformers:VitPoseConfig',
            'model_class': 'transformers:VitPoseForPoseEstimation',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'usyd-community/vitpose-base-simple',
                'revision': 'a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45',
                'description': 'Pinned HF public task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'shape': [3, 256, 192], 'batch_size': 1},
        'outputs': ['heatmaps'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'vitpose_backbone': {
        'reference': {
            'config_class': 'transformers:VitPoseBackboneConfig',
            'model_class': 'transformers:VitPoseBackbone',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py',
                'description': ('Pinned HF configuration documentation explicitly constructs this public task with '
                    'constructor defaults.'),
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 256, 192]},
        'outputs': ['feature_maps'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'vivit': {
        'reference': {
            'config_class': 'transformers:VivitConfig',
            'model_class': 'transformers:VivitModel',
            'source': {
                'checkpoint': 'google/vivit-b-16x2-kinetics400',
                'revision': '8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42',
                'url': 'https://huggingface.co/google/vivit-b-16x2-kinetics400/resolve/8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42/config.json',
                'kind': 'example_checkpoint',
                'description': ('The pinned base-model forward example loads this checkpoint. Retain learned '
                    'CLS/position embeddings, FastGELU and the default CLS pooler. Its legacy '
                    'video_size=[32,224,224] agrees with current num_frames/image_size constructor '
                    'defaults.'),
            },
        },
        'input': {'kind': 'video', 'batch_size': 1, 'shape': [32, 3, 224, 224]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'pooler_output'],
        'reference_backend': None,
    },

    'vjepa2': {
        'reference': {
            'config_class': 'transformers:VJEPA2Config',
            'model_class': 'transformers:VJEPA2Model',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/vjepa2/configuration_vjepa2.py',
                'description': ('Pinned HF configuration documentation explicitly constructs this public task with '
                    'constructor defaults.'),
            },
        },
        'input': {'kind': 'video', 'name': 'pixel_values_videos', 'batch_size': 1, 'shape': [64, 3, 256, 256]},
        'outputs': [
            'last_hidden_state', 'masked_hidden_state', 'predictor_output.last_hidden_state',
            'predictor_output.target_hidden_state',
        ],
        'workload': 'forward',
        'reference_backend': None,
    },

    'voxtral': {
        'reference': {
            'config_class': 'transformers:VoxtralConfig',
            'model_class': 'transformers:VoxtralForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'mistralai/Voxtral-Mini-3B-2507',
                'revision': '3060fe34b35ba5d44202ce9ff3c097642914f8f3',
                'url': 'https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/blob/3060fe34b35ba5d44202ce9ff3c097642914f8f3/config.json',
                'description': 'Pinned public speech example generates text from audio; retain full audio encoder and native greedy token generation.',
            },
            'generation_config': {'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 11, 'transformers_version': '4.54.0.dev0'},
        },
        'dimension_overrides': {
            'audio_config': {
                'hidden_size': 64, 'intermediate_size': 256, 'num_attention_heads': 2,
                'num_key_value_heads': 2, 'head_dim': 32, 'num_hidden_layers': 2, 'max_source_positions': 32,
            },
            'text_config': {
                'hidden_size': 96, 'intermediate_size': 256, 'num_attention_heads': 4,
                'num_key_value_heads': 1, 'head_dim': 32, 'num_hidden_layers': 2,
                'max_position_embeddings': 256,
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 13,
            'shape': [128, 64], 'image_input_name': 'input_features',
            'audio_token_positions': list(range(2, 10)), 'attention_mask': True, 'pad_token_id': 11,
        },
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 3},
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Reduce widths/depth/frame counts, retain128mel bins,4frameaudio '
            'concatenation,GQA4:1,attentionwidth4/3modelwidth,FP32audio position weights and native '
            'vocabulary;13prompttokens include8audio slots.'),
    },

    'voxtral_realtime': {
        'reference': {
            'config_class': 'transformers:VoxtralRealtimeConfig',
            'model_class': 'transformers:VoxtralRealtimeForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'mistralai/Voxtral-Mini-4B-Realtime-2602',
                'revision': '2769294da9567371363522aac9bbcfdd19447add',
                'url': 'https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/blob/2769294da9567371363522aac9bbcfdd19447add/config.json',
                'description': ('Pinned ordinary array-input speech generation, with native causal convolution '
                    'embedding, incremental four-frame audio encoding, two sliding caches and '
                    'time-conditioned tied text decoder.'),
            },
            'generation_config': {
                'bos_token_id': 1, 'eos_token_id': 2, 'output_attentions': False,
                'output_hidden_states': False, 'pad_token_id': 11, 'transformers_version': '5.2.0.dev0',
                'use_cache': True,
            },
        },
        'dimension_overrides': {
            'audio_config': {
                'hidden_size': 80, 'intermediate_size': 320, 'num_attention_heads': 8,
                'num_key_value_heads': 8, 'head_dim': 16, 'num_hidden_layers': 2,
                'max_position_embeddings': 128, 'sliding_window': 8,
            },
            'text_config': {
                'hidden_size': 96, 'intermediate_size': 288, 'num_attention_heads': 4,
                'num_key_value_heads': 1, 'head_dim': 32, 'num_hidden_layers': 2,
                'max_position_embeddings': 128, 'sliding_window': 4,
            },
        },
        'input': {
            'kind': 'text_audio', 'text_batch_size': 1, 'image_batch_size': 1, 'sequence_length': 7,
            'shape': [128, 80], 'image_input_name': 'input_features', 'attention_mask': True,
            'pad_token_id': 11,
        },
        'outputs': ['sequences', 'logits', 'past_key_values'],
        'workload': 'generate',
        'generation_kwargs': {'max_new_tokens': 3},
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Preserve128mel bins,eightaudioframes/token,fourframeprojection,delay6,timeMLP32,tiedhead and '
            'native vocab. Preserve encoder attention/model width8:5 and text4:3; reduce '
            'widths/layers/windows. Prefill28audioframes and7texttokens cross both windows8/4; two '
            'continuation steps exercise retained caches. Whole feature array convolutions remain measured.'),
    },

    'wav2vec2': {
        'reference': {
            'config_class': 'transformers:Wav2Vec2Config',
            'model_class': 'transformers:Wav2Vec2Model',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/wav2vec2-large-960h-lv60-self',
                'revision': '54074b1c16f4de6a5ad59affb4caa8f2ea03a119',
                'url': 'https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self/blob/54074b1c16f4de6a5ad59affb4caa8f2ea03a119/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/wav2vec2.md:73 loads this checkpoint as '
                    'Wav2Vec2Model. Preserve its biased layer-normalized feature convolutions and '
                    'pre-normalized encoder; SDPA is the explicitly selected comparison backend.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [48000]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'extract_features'],
        'reference_backend': None,
    },

    'wav2vec2_bert': {
        'reference': {
            'config_class': 'transformers:Wav2Vec2BertConfig',
            'model_class': 'transformers:Wav2Vec2BertModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/w2v-bert-2.0',
                'revision': 'da985ba0987f70aaeb84a80f2851cfac8c697a7b',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/tests/models/wav2vec2_bert/test_modeling_wav2vec2_bert.py#L610',
                'description': ('Pinned HF base-model loading test selects this checkpoint; preserve its full encoder '
                    'and ordinary outputs.'),
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'spectrogram', 'name': 'input_features', 'batch_size': 1, 'shape': [137, 160],
            'attention_mask_length': 137,
        },
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'extract_features'],
    },

    'wav2vec2_conformer': {
        'reference': {
            'config_class': 'transformers:Wav2Vec2ConformerConfig',
            'model_class': 'transformers:Wav2Vec2ConformerModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/wav2vec2-conformer-rel-pos-large',
                'revision': '1afaab48b41d924fbbcae05d8c5d88836c4a5719',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/tests/models/wav2vec2_conformer/test_modeling_wav2vec2_conformer.py#L617',
                'description': ('Pinned HF base-model loading test selects this checkpoint. Preserve its full encoder '
                    'and ordinary outputs; the constructor example has different dimensions.'),
            },
        },
        'reference_backend': None,
        'input': {
            'kind': 'waveform', 'name': 'input_values', 'batch_size': 1, 'shape': [48000],
            'attention_mask_length': 48000,
        },
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'extract_features'],
    },

    'wavlm': {
        'reference': {
            'config_class': 'transformers:WavLMConfig',
            'model_class': 'transformers:WavLMModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/wavlm/configuration_wavlm.py#L142',
                'description': ('Preserve constructor-example defaults, including activation-dependent gated relative '
                    'attention bias and 320 buckets with maximum distance 800. HF uses its torch multi-head'
                    ' attention wrapper, whose default no-attention-weights path calls SDPA.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [258000]},
        'workload': 'forward',
        'outputs': ['last_hidden_state', 'extract_features'],
        'reference_backend': None,
    },

    'whisper': {
        'reference': {
            'config_class': 'transformers:WhisperConfig',
            'model_class': 'transformers:WhisperForConditionalGeneration',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'openai/whisper-tiny.en',
                'revision': '87c7102498dcde7456f24cfd30239ca606ed9063',
                'url': 'https://huggingface.co/openai/whisper-tiny.en/blob/87c7102498dcde7456f24cfd30239ca606ed9063/config.json',
                'description': ('Pinned HF conditional-generation example selects openai/whisper-tiny.en; retain '
                    'log-mel frontend, GELU, prescaled attention, tied head and default encoder-decoder '
                    'cache.'),
            },
        },
        'input': {
            'kind': 'spectrogram', 'name': 'input_features', 'batch_size': 1, 'shape': [80, 3000],
            'decoder_sequence_length': 139,
        },
        'workload': 'seq2seq_continuation',
        'outputs': ['logits', 'encoder_last_hidden_state', 'past_key_values'],
        'reference_backend': None,
    },

    'x_clip': {
        'reference': {
            'config_class': 'transformers:XCLIPConfig',
            'model_class': 'transformers:XCLIPModel',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'microsoft/xclip-base-patch32',
                'revision': 'a2e27a78a2b5d802e894b8a1ef14f3a8ce490963',
                'url': 'https://huggingface.co/microsoft/xclip-base-patch32/blob/a2e27a78a2b5d802e894b8a1ef14f3a8ce490963/config.json',
                'description': 'Native paired text/video forward example, including all8frames and full temporal/prompt blocks.',
            },
        },
        'input': {
            'kind': 'text_image', 'text_batch_size': 3, 'image_batch_size': 1, 'sequence_length': 77,
            'shape': [8, 3, 224, 224], 'eos_positions': [76, 76, 76], 'bos_token_id': 49406,
            'eos_token_id': 49407, 'pad_token_id': 49407, 'batch_size': 1,
        },
        'workload': 'forward',
        'outputs': [
            'logits_per_video', 'logits_per_text', 'text_embeds', 'video_embeds',
            'text_model_output.last_hidden_state', 'text_model_output.pooler_output',
            'vision_model_output.last_hidden_state', 'vision_model_output.pooler_output',
            'mit_output.last_hidden_state', 'mit_output.pooler_output',
        ],
        'reference_backend': None,
    },

    'xcodec': {
        'reference': {
            'config_class': 'transformers:XcodecConfig',
            'model_class': 'transformers:XcodecModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'hf-audio/xcodec-hubert-librispeech',
                'revision': '2344bd08701239506b61e597ec7910f98bc86e2b',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/xcodec/modeling_xcodec.py#L594',
                'description': ('Pinned HF public forward selects this joint HuBERT/DAC codec. Preserve all semantic '
                    'hidden-state averaging, default highest bandwidth, acoustic decode.'),
            },
        },
        'input': {'kind': 'waveform', 'batch_size': 1, 'shape': [1, 3200]},
        'workload': 'forward',
        'outputs': ['audio_codes', 'audio_values'],
        'reference_backend': None,
    },

    'xglm': {
        'reference': {
            'config_class': 'transformers:XGLMConfig',
            'model_class': 'transformers:XGLMForCausalLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/xglm-564M',
                'revision': 'f3059f01b98ccc877c673149e0178c0e957660f9',
                'url': 'https://huggingface.co/facebook/xglm-564M/blob/f3059f01b98ccc877c673149e0178c0e957660f9/config.json',
                'description': ('Pinned HF task example or configuration documentation checkpoint; preserve checkpoint '
                    'computation flags, completed by pinned constructor defaults.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'xlm': {
        'reference': {
            'config_class': 'transformers:XLMConfig',
            'model_class': 'transformers:XLMWithLMHeadModel',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'FacebookAI/xlm-mlm-en-2048',
                'revision': '6eb6401a142611ae90f3d6bc606b97384f1c9961',
                'url': 'https://huggingface.co/FacebookAI/xlm-mlm-en-2048/blob/6eb6401a142611ae90f3d6bc606b97384f1c9961/config.json',
                'description': ('Pinned HF task example or configuration documentation checkpoint; preserve checkpoint '
                    'computation flags, completed by pinned constructor defaults.'),
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'reference_backend': None,
        'outputs': ['logits'],
    },

    'xlm_roberta': {
        'reference': {
            'config_class': 'transformers:XLMRobertaConfig',
            'model_class': 'transformers:XLMRobertaForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'FacebookAI/xlm-roberta-base',
                'revision': 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',
                'url': 'https://huggingface.co/FacebookAI/xlm-roberta-base/blob/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/config.json',
                'description': ('Pinned HF docs/source/en/model_doc/xlm-roberta.md:61 masked-LM example; checkpoint '
                    'configuration completed by pinned constructor defaults.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'xlm_roberta_xl': {
        'reference': {
            'config_class': 'transformers:XLMRobertaXLConfig',
            'model_class': 'transformers:XLMRobertaXLForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'facebook/xlm-roberta-xl',
                'revision': 'aa5d120255845efeebc9b7f42822a1dd0f9ece9d',
                'url': 'https://huggingface.co/facebook/xlm-roberta-xl/blob/aa5d120255845efeebc9b7f42822a1dd0f9ece9d/config.json',
                'description': ('Preserve the corpus masked-LM task using the checkpoint named by pinned HF '
                    'docs/source/en/model_doc/xlm-roberta-xl.md:57-68; complete omitted values from the '
                    'pinned constructor.'),
            },
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'xlnet': {
        'default_dtype': 'float32',
        'reference': {
            'config_class': 'transformers:XLNetConfig',
            'model_class': 'transformers:XLNetModel',
            'source': {
                'kind': 'constructor_defaults',
                'revision': 'da6c53e431f7c9ef0691239d4ce89b0f711ecad7',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/xlnet/configuration_xlnet.py',
                'description': ('Pinned XLNetConfig example constructs XLNetModel(XLNetConfig()); preserve its '
                    'constructor computation.'),
            },
            'cache_argument': 'mems',
            'cache_output': 'mems',
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 514},
        'workload': 'memory_continuation',
        'outputs': ['last_hidden_state', 'mems'],
        'reference_backend': None,
    },

    'xlstm': {
        'reference': {
            'config_class': 'transformers:xLSTMConfig',
            'model_class': 'transformers:xLSTMForCausalLM',
            'cache_argument': 'cache_params',
            'cache_output': 'cache_params',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'NX-AI/xLSTM-7b',
                'revision': '9dc507bd0939cf372a4a4f667335651d8e49dddb',
                'url': 'https://huggingface.co/NX-AI/xLSTM-7b/blob/9dc507bd0939cf372a4a4f667335651d8e49dddb/config.json',
                'description': 'Pinned HF documented checkpoint; original full dimensions and Triton recurrence backends retained.',
            },
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 130},
        'workload': 'causal_lm_continuation',
        'outputs': ['logits', 'cache_params.rnn_state'],
    },

    'xmod': {
        'reference': {
            'config_class': 'transformers:XmodConfig',
            'model_class': 'transformers:XmodForMaskedLM',
            'source': {
                'kind': 'example_checkpoint',
                'description': ('Preserve XmodForMaskedLM and all 81 checkpoint language adapters. Apply the pinned '
                    'documentation\'s explicit set_default_language("en_XX") choice through '
                    'default_language; retain its default adapter normalization flags.'),
                'checkpoint': 'facebook/xmod-base',
                'revision': '1ff23836a9ee8b9656553630c33506a9a8a59c4f',
                'url': 'https://huggingface.co/facebook/xmod-base/blob/1ff23836a9ee8b9656553630c33506a9a8a59c4f/config.json',
            },
        },
        'config_overrides': {'default_language': 'en_XX'},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 512},
        'workload': 'masked_lm',
        'outputs': ['logits'],
        'reference_backend': None,
    },

    'yolos': {
        'reference': {
            'config_class': 'transformers:YolosConfig',
            'model_class': 'transformers:YolosForObjectDetection',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'hustvl/yolos-tiny',
                'revision': '95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/yolos/modeling_yolos.py',
                'description': 'Pinned public-task example checkpoint.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 512, 672]},
        'outputs': ['logits', 'pred_boxes', 'last_hidden_state'],
        'workload': 'forward',
        'reference_backend': None,
    },

    'youtu': {
        'reference': {
            'config_class': 'transformers:YoutuConfig',
            'model_class': 'transformers:YoutuForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'tencent/Youtu-LLM-2B',
                'revision': '8b0e73594661a945a7f5aafc737b63e3d8ad3f75',
            },
        },
        'dimension_overrides': {
            'hidden_size': 512, 'intermediate_size': 1024, 'num_hidden_layers': 2, 'num_attention_heads': 4,
            'num_key_value_heads': 4, 'q_lora_rank': 256, 'vocab_size': 1024, 'bos_token_id': 1,
            'eos_token_id': 2,
        },
        'input': {'kind': 'tokens', 'batch_size': 2, 'sequence_length': 270},
        'workload': 'causal_lm',
        'reference_backend': 'sdpa',
        'dimension_purpose': ('Two dense decoder layers, compressed query/KV projections, native192-wide Q/K and128-wide '
            'values;269-token prompt crosses attention tiles and includes cached continuation. Head '
            'count,width/queryrank/vocabulary reduced; nativeKVrank512 and plain interleavedRoPE retained.'),
    },

    'zamba': {
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'config_overrides': {},
        'reference': {
            'config_class': 'transformers:ZambaConfig',
            'model_class': 'transformers:ZambaForCausalLM',
            'source': {
                'kind': 'example_checkpoint', 'checkpoint': 'Zyphra/Zamba-7B-v1',
                'revision': 'b8c9e7ede2f60ef36c21328bcf578932318db771',
                'url': 'https://huggingface.co/Zyphra/Zamba-7B-v1/resolve/b8c9e7ede2f60ef36c21328bcf578932318db771/config.json',
                'description': 'Pinned Transformers docs/source/en/model_doc/zamba.md causal LM example; computational settings from this checkpoint config.',
            },
            'forward_kwargs': {'logits_to_keep': 0},
            'conv_cache_history': 3,
        },
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 271},
        'outputs': ['logits', 'past_key_values'],
    },

    'zamba2': {
        'reference': {
            'config_class': 'transformers:Zamba2Config',
            'model_class': 'transformers:Zamba2ForCausalLM',
            'source': {
                'kind': 'constructor_defaults',
                'checkpoint': 'Zyphra/Zamba2-7B',
                'revision': '0ed988f366ddf3293f2dbe8a4843884215e0afdb',
                'description': ('Explicit pinned constructor defaults, not equivalent to the named 7B checkpoint. The '
                    'full 54-layer configuration preserves memory sharing, nine MLP adapter uses, no '
                    'attention adapters/RoPE, and separate Mamba kernels. Earlier checkpoint-access '
                    'investigation remains in scratch.'),
                'url': 'https://huggingface.co/Zyphra/Zamba2-7B',
            },
            'conv_cache_history': 3,
        },
        'config_overrides': {},
        'input': {'kind': 'tokens', 'batch_size': 1, 'sequence_length': 271},
        'workload': 'causal_lm_continuation',
        'reference_backend': None,
        'outputs': ['logits', 'past_key_values'],
    },

    'zoedepth': {
        'reference': {
            'config_class': 'transformers:ZoeDepthConfig',
            'model_class': 'transformers:ZoeDepthForDepthEstimation',
            'source': {
                'kind': 'example_checkpoint',
                'checkpoint': 'Intel/zoedepth-nyu-kitti',
                'revision': 'f364d4c7936e91f465abba182208dd68142bf0ca',
                'url': 'https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/zoedepth/modeling_zoedepth.py',
                'description': 'Pinned public depth-estimation checkpoint, native NYU/KITTI routing and both output tensors.',
            },
        },
        'input': {'kind': 'image', 'batch_size': 1, 'shape': [3, 384, 512]},
        'outputs': ['predicted_depth', 'domain_logits'],
        'workload': 'forward',
        'reference_backend': None,
    },
}
