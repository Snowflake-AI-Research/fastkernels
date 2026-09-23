"""OWLv2 retains OWL-ViT computation and adds the default objectness head."""

from .owlvit import build_detector, load_detector, make_workloads


def build_from_config(config, device, dtype):
    return build_detector(config, device, dtype, objectness=True)


def load_state_dict_into(model, state_dict, config):
    load_detector(model, state_dict, config, prefix="owlv2")
