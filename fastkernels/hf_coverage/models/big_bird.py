"""BigBird MLM with the pinned HF evaluation block-sparse attention pattern."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from .bert import MaskedLMHead, make_workloads


class EvalBlockSparseAttention(nn.Module):
    """Q/K/V [B,H,T,D], already block padded; mask [B,T] marks real tokens.

    Both pinned HF random-block planners return block zero in evaluation.
    Repeated entries remain repeated, as required by that reference computation.
    """
    def __init__(self, block_size=64, num_random_blocks=3):
        super().__init__()
        self.block_size, self.num_random_blocks = block_size, num_random_blocks
        self.matmul, self.softmax = BatchMatMul(), Softmax()

    def _bmm(self, a, b):
        shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a, b = a.expand(*shape, *a.shape[-2:]), b.expand(*shape, *b.shape[-2:])
        return self.matmul(a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:])).reshape(*shape, a.shape[-2], b.shape[-1])

    def forward(self, query, key, value, mask):
        batch, heads, length, width = query.shape
        block, random = self.block_size, self.num_random_blocks
        count = length // block
        if self.training or length % block or count <= 5 + 2 * random:
            raise ValueError('Sparse evaluation requires a block-padded sequence above the HF dense fallback length')
        q, k, v = [x.reshape(batch, heads, count, block, width) for x in (query, key, value)]
        valid = mask.reshape(batch, count, block)

        def grouped(indices, key_blocks, split_middle=False):
            selected_q = q[:, :, indices]
            selected_k = k[:, :, key_blocks].flatten(3, 4)
            selected_v = v[:, :, key_blocks].flatten(3, 4)
            selected_mask = valid[:, key_blocks].flatten(2, 3)
            scores = self._bmm(selected_q, selected_k.transpose(-1, -2)) * width**-0.5
            scores = scores + (~selected_mask[:, None, :, None, :]).to(scores.dtype) * -10000.0
            weights = self.softmax(scores)
            if not split_middle:
                return self._bmm(weights, selected_v)
            # Preserve HF's four separately rounded value reductions and additions.
            ranges = ((block, 4 * block), (4 * block, (4 + random) * block),
                      (0, block), ((4 + random) * block, (5 + random) * block))
            parts = [self._bmm(weights[..., start:end], selected_v[..., start:end, :]) for start, end in ranges]
            return ((parts[0] + parts[1]) + parts[2]) + parts[3]

        device = query.device
        global_keys = torch.arange(count, device=device)[None, :]
        first = grouped([0], global_keys)
        second_keys = torch.tensor([[0, 1, 2, count - 1] + [0] * random], device=device)
        second = grouped([1], second_keys)
        middle_ids = torch.arange(2, count - 2, device=device)
        middle_keys = torch.stack([torch.zeros_like(middle_ids), middle_ids - 1, middle_ids,
                                   middle_ids + 1] + [torch.zeros_like(middle_ids)] * random
                                  + [torch.full_like(middle_ids, count - 1)], dim=1)
        middle = grouped(middle_ids, middle_keys, split_middle=True)
        penultimate_keys = torch.tensor([[0, count - 3, count - 2, count - 1] + [0] * random], device=device)
        penultimate = grouped([count - 2], penultimate_keys)
        last = grouped([count - 1], global_keys)
        result = torch.cat((first, second, middle, penultimate, last), dim=2).reshape(batch, heads, length, width)
        return result.masked_fill(~mask[:, None, :, None], 0)


class BigBirdLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size, bias=config.use_bias)
        self.attention = EvalBlockSparseAttention(config.block_size, config.num_random_blocks)
        self.proj = Linear(config.hidden_size, config.hidden_size)
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size, act_approximate='tanh')
        self.mlp.act = NewGELUActivation()

    def forward(self, hidden, mask):
        batch, length = hidden.shape[:2]
        q, k, v = [x.reshape(batch, length, self.heads, self.width).transpose(1, 2) for x in self.qkv(hidden).chunk(3, -1)]
        context = self.attention(q, k, v, mask).transpose(1, 2).reshape(batch, length, -1)
        hidden = self.norm1(hidden + self.proj(context))
        return self.norm2(hidden + self.mlp(hidden))


class BigBirdForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size, self.padding_idx = config.block_size, config.pad_token_id
        self.pad = Pad()
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList([BigBirdLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = Tanh()
        self.lm_head = MaskedLMHead(config)
        self.lm_head.activation = NewGELUActivation()
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        length = input_ids.shape[1]
        ids = self.pad(input_ids, (0, -length % self.block_size), value=self.padding_idx)
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        mask = (positions < length).expand(ids.shape[0], -1)
        hidden = self.embeddings.forward_with_token_type_ids(ids, positions)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        # HF executes this backbone branch even though the MLM head discards it.
        self.pooler_activation(self.pooler(hidden[:, 0]))
        return self.lm_head(hidden[:, :length])


def build_from_config(config, device, dtype):
    if config.attention_type != 'block_sparse' or config.hidden_act != 'gelu_new' or config.rescale_embeddings:
        raise ValueError('The selected BigBird checkpoint requires block-sparse attention, GELU-new and unscaled embeddings')
    return BigBirdForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    if not torch.equal(remaining['cls.predictions.bias'], remaining['cls.predictions.decoder.bias']):
        raise ValueError('Masked-LM bias aliases disagree')
    remaining.pop('cls.predictions.bias')
    names = {'proj': 'attention.output.dense', 'norm1': 'attention.output.LayerNorm',
             'norm2': 'output.LayerNorm', 'mlp.fc1': 'intermediate.dense', 'mlp.fc2': 'output.dense'}
    for name in model.state_dict():
        if name.startswith('layers.'):
            _, index, rest = name.split('.', 2)
            part, field = rest.rsplit('.', 1)
            prefix = f'bert.encoder.layer.{index}.'
            if part == 'qkv':
                weights[name] = torch.cat([remaining.pop(prefix + f'attention.self.{p}.{field}') for p in ('query', 'key', 'value')])
                continue
            source = prefix + names[part] + '.' + field
        elif name.startswith('lm_head.'):
            rest = name[len('lm_head.'):]
            source = ('cls.predictions.' if rest.startswith('decoder.') else 'cls.predictions.transform.') + rest
        else:
            source = 'bert.' + name.replace('.emb.weight', '.weight')
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped BigBird parameters: {sorted(remaining)}')
    if config.tie_word_embeddings and not torch.equal(weights['lm_head.decoder.weight'], weights['embeddings.word_embeddings.emb.weight']):
        raise ValueError('Tied BigBird weights disagree')
    model.load_state_dict(weights, strict=True)
