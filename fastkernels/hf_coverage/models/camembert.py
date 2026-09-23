"""CamemBERT shares the RoBERTa encoder computation and masked-LM head."""

from .roberta import build_from_config, load_state_dict_into, make_workloads
