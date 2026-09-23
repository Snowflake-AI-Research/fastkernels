"""Reuse the complete V-JEPA2 encoder and its default predictor."""

from fastkernels.tasks.baseline.L4.vjepa2 import VJEPA2Model

from ..runner import Workload


def build_from_config(config, device, dtype):
    return VJEPA2Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {name.replace("patch_embeddings.proj.", "patch_embeddings.proj.conv."): value
              for name, value in state_dict.items()}
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    def forward():
        output = model(**inputs)
        return {
            "last_hidden_state": output.last_hidden_state,
            "masked_hidden_state": output.masked_hidden_state,
            "predictor_output.last_hidden_state": output.predictor_output.last_hidden_state,
            "predictor_output.target_hidden_state": output.predictor_output.target_hidden_state,
        }
    return {"forward": Workload(run=forward)}
