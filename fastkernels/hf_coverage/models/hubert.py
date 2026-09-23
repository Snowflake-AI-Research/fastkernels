"""HubertModel's documented large encoder, sharing Wav2Vec2 operations."""

from .wav2vec2 import build_waveform_model, load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    return build_waveform_model(config, device, dtype, return_features=False)
