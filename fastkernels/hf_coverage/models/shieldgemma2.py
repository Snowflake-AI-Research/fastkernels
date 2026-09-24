"""ShieldGemma2's full Gemma3 forward followed by its two-token classifier."""

from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.softmax import Softmax
from . import gemma3
from ..runner import Workload


class ShieldGemma2(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.model = gemma3.build_from_config(config, device, dtype)
        # Match the selected native SDPA dispatch with an existing operation.
        # Gemma3's cuDNN preference forces MATH for head256, introducing BF16
        # drift before the small two-score classifier even on identical Q/K/V.
        for layer in self.model.model.language_model.layers:
            layer.self_attn.core = DenseAttention(backend='sdpa')
        self.indices = [getattr(config, 'yes_token_index', 10784),
                        getattr(config, 'no_token_index', 3771)]
        if any(index < 0 or index >= config.text_config.vocab_size for index in self.indices):
            raise ValueError('ShieldGemma2 vocabulary must contain both classifier tokens')
        self.softmax = Softmax(dim=-1)

    def forward(self, **inputs):
        # Preserve the complete vocabulary projection and internal cache work.
        output = self.model(**inputs)
        logits = output['logits'][:, -1, self.indices]
        return {'logits': logits, 'probabilities': self.softmax(logits)}


def build_from_config(config, device, dtype):
    return ShieldGemma2(config, device, dtype).eval()


def load_state_dict_into(model, state_dict, config):
    if any(not name.startswith('model.') for name in state_dict):
        raise ValueError('Expected ShieldGemma2 weights under model.')
    gemma3.load_state_dict_into(model.model, {
        name.removeprefix('model.'): value for name, value in state_dict.items()
    }, config)


def make_workloads(model, inputs, config, case=None):
    return {'forward': Workload(run=lambda: model(**inputs))}
