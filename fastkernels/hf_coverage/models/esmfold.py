"""ESMFold ordinary protein folding: ESM-2, triangular trunk, and rigid geometry.

Numerical carriers are existing operations, including unchanged AF3 internal
quaternion conversion. Fixed residue data below are from pinned HF/OpenFold.
"""

import math
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.alphafold3_triangle_attention import TriangleAttention
from fastkernels.tasks.baseline.L2.alphafold3_triangle_multiplication import TriangleMultiplicativeUpdate
from fastkernels.tasks.baseline.L2.alphafold3_of3_attention import _attention
from fastkernels.tasks.baseline.L3.alphafold3_diffusion_module import _quat_to_rot
from ..patches.codec_top1 import CodecTop1
from ..patches.gelu_python import PythonGELU
from ..patches.esmfold_l2_norm import EsmFoldL2Norm
from ..patches.product_gate import ProductGate
from ..runner import Workload


def norm(width, eps=1e-5):
    return LayerNorm(width, eps=eps, promote_fp32=False)


def group(**children):
    module = nn.Module()
    for name, child in children.items():
        module.add_module(name, child)
    return module


def residue_weight_scale(length, dtype):
    """Shape-only constant matching the native unit-weight reduction dtype."""
    count = torch.tensor(length, dtype=dtype)
    return float(torch.tensor(1.0, dtype=dtype) / (count + 1e-8))


class Arithmetic(nn.Module):
    """Linear-storage products/reductions and positive-domain square roots."""
    def __init__(self):
        super().__init__()
        self.product, self.matmul, self.select = ProductGate(), BMM(), CodecTop1()
        self.batch_norm = BatchNorm2d(1, eps=1e-8, affine=False).eval()
        self.batch_norm._non_persistent_buffers_set.update(self.batch_norm._buffers)

    def mul(self, x, y):
        x, y = torch.broadcast_tensors(x, y)
        return self.product(torch.cat((x, y), -1))

    def sum(self, x):
        # Explicit FP32 accumulation, followed by native reduction output dtype.
        return self.matmul(x.float(), torch.ones(x.shape[-1], 1, device=x.device)).squeeze(-1).to(x.dtype)

    def sqrt(self, x, epsilon):
        # (x + eps) / sqrt(x + eps), using unchanged inference BatchNorm.
        # Here x >= 0 and epsilon > 0; zero inputs therefore remain defined.
        y = x.float() + epsilon
        self.batch_norm.eps = epsilon
        self.batch_norm.running_mean = torch.zeros_like(y).flatten()
        self.batch_norm.running_var = x.float().flatten()
        return self.batch_norm(y.reshape(1, -1, 1, 1)).reshape_as(x).to(x.dtype)

    def rotate(self, rotation, points):
        return self.matmul(rotation.float(), points.float().unsqueeze(-1)).squeeze(-1)

    def compose(self, first, second):
        r1, t1 = first
        r2, t2 = second
        return self.matmul(r1, r2), self.rotate(r1, t2) + t1

    def update(self, quat, translation, update):
        # Quaternion left multiplication is a genuine small dense linear map.
        # Its 4x3 matrix has the same fixed work as native quaternion products.
        w, x, y, z = quat.unbind(-1)
        matrix = torch.stack((-x, -y, -z, w, -z, y, z, w, -x, -y, x, w), -1).reshape(*quat.shape[:-1], 4, 3)
        next_quat = quat + self.matmul(matrix, update[..., :3].float().unsqueeze(-1)).squeeze(-1)
        next_translation = translation + self.rotate(_quat_to_rot(quat), update[..., 3:])
        return L2Norm(eps=0)(next_quat), next_translation


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h = config.hidden_size, config.num_attention_heads
        self.heads, self.width = h, d // h
        self.attention = group(**{'self': group(query=Linear(d, d), key=Linear(d, d), value=Linear(d, d)),
                                 'output': group(dense=Linear(d, d)), 'LayerNorm': norm(d, config.layer_norm_eps)})
        self.intermediate, self.output = group(dense=Linear(d, config.intermediate_size)), group(dense=Linear(config.intermediate_size, d))
        self.LayerNorm, self.gelu = norm(d, config.layer_norm_eps), PythonGELU()

    def forward(self, hidden, mask, rotary):
        normalized = self.attention.LayerNorm(hidden)
        projection = getattr(self.attention, 'self')
        q, k, v = [getattr(projection, name)(normalized) for name in ('query', 'key', 'value')]
        q = q * self.width**-0.5
        shape = (*hidden.shape[:2], self.heads, self.width)
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(hidden.shape[0], -1).reshape(-1)
        q, k = RotaryEmbedding.forward_native(positions, q.reshape(-1, q.shape[-1]).float(),
            k.reshape(-1, k.shape[-1]).float(), self.width, rotary.cos_sin_cache.to(q.dtype).float())
        bias = hidden.new_zeros(mask.shape).masked_fill(~mask, torch.finfo(hidden.dtype).min)[:, None, None, :]
        output = _attention(q.to(hidden.dtype).view(shape).transpose(1, 2),
                            k.to(hidden.dtype).view(shape).transpose(1, 2), v.view(shape).transpose(1, 2), [bias]).transpose(1, 2)
        hidden = hidden + self.attention.output.dense(output.reshape_as(hidden))
        return hidden + self.output.dense(self.gelu(self.intermediate.dense(self.LayerNorm(hidden))))


class EsmStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = group(word_embeddings=Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id))
        self.encoder = group(layer=nn.ModuleList(EsmLayer(config) for _ in range(config.num_hidden_layers)),
                             emb_layer_norm_after=norm(config.hidden_size, config.layer_norm_eps))
        self.contact_head = group(regression=Linear(config.num_hidden_layers * config.num_attention_heads, 1))
        self.rotary_embeddings = group()
        width = config.hidden_size // config.num_attention_heads
        self.rotary_embeddings.register_buffer('inv_freq', 1.0 / config.rope_theta ** (torch.arange(0, width, 2).float() / width))
        self.rotary = RotaryEmbedding(width, config.max_position_embeddings, config.rope_theta)

    def forward(self, ids):
        mask = ids != self.config.pad_token_id
        hidden = self.embeddings.word_embeddings(ids)
        # Ordinary folding has no MLM mask pattern: token-dropout compensation
        # is the configured training keep fraction, even at inference.
        hidden = (hidden * 0.88).masked_fill(~mask[..., None], 0)
        states = []
        for layer in self.encoder.layer:
            states.append(hidden)
            hidden = layer(hidden, mask, self.rotary)
        hidden = self.encoder.emb_layer_norm_after(hidden)
        states.append(hidden)
        return torch.stack(states, dim=2)


class SequenceToPair(nn.Module):
    def __init__(self, s, z):
        super().__init__()
        self.layernorm, self.proj, self.o_proj = norm(s), Linear(s, z), Linear(z, z)
        self.arithmetic = Arithmetic()

    def forward(self, sequence):
        q, k = self.proj(self.layernorm(sequence)).chunk(2, -1)
        return self.o_proj(torch.cat((self.arithmetic.mul(q[:, None], k[:, :, None]), q[:, None] - k[:, :, None]), -1))


class PairToSequence(nn.Module):
    def __init__(self, z, heads):
        super().__init__()
        self.layernorm, self.linear = norm(z), Linear(z, heads, bias=False)

    def forward(self, pairs):
        return self.linear(self.layernorm(pairs))


class SequenceAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.width = heads, dim // heads
        self.proj, self.o_proj, self.g_proj = Linear(dim, 3 * dim, bias=False), Linear(dim, dim), Linear(dim, dim)
        self.arithmetic, self.softmax, self.sigmoid = Arithmetic(), Softmax(), Sigmoid()

    def forward(self, x, mask, bias):
        q, k, v = self.proj(x).view(*x.shape[:2], self.heads, -1).permute(0, 2, 1, 3).chunk(3, -1)
        a = self.arithmetic.matmul(q * self.width**-0.5, k.transpose(-1, -2)) + bias.permute(0, 3, 1, 2)
        a = self.softmax(a.masked_fill(~mask[:, None, None].bool(), -torch.inf))
        output = self.arithmetic.matmul(a, v).transpose(1, 2).reshape_as(x)
        return self.o_proj(self.arithmetic.mul(output, self.sigmoid(self.g_proj(x))))


class ResidueMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(norm(dim), Linear(dim, 4 * dim), ReLU(), Linear(4 * dim, dim))

    def forward(self, x):
        return x + self.mlp(x)


class TriangleBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        s, z = config.sequence_state_dim, config.pairwise_state_dim
        self.layernorm_1 = norm(s)
        self.sequence_to_pair, self.pair_to_sequence = SequenceToPair(s, z), PairToSequence(z, s // config.sequence_head_width)
        self.seq_attention = SequenceAttention(s, s // config.sequence_head_width)
        for name, outgoing in [('tri_mul_out', True), ('tri_mul_in', False)]:
            operation = TriangleMultiplicativeUpdate(z, z, outgoing)
            for child in ('linear_a_p', 'linear_a_g', 'linear_b_p', 'linear_b_g', 'linear_g', 'linear_z'):
                setattr(operation, child, Linear(z, z))
            operation.layer_norm_in, operation.layer_norm_out = norm(z), norm(z)
            setattr(self, name, operation)
        for name, starting in [('tri_att_start', True), ('tri_att_end', False)]:
            operation = TriangleAttention(z, config.pairwise_head_width, z // config.pairwise_head_width, starting)
            operation.layer_norm = norm(z)
            operation.mha.linear_o, operation.mha.linear_g = Linear(z, z), Linear(z, z)
            setattr(self, name, operation)
        self.mlp_seq, self.mlp_pair = ResidueMLP(s), ResidueMLP(z)

    def forward(self, s, z, mask):
        s = s + self.seq_attention(self.layernorm_1(s), mask, self.pair_to_sequence(z))
        s = self.mlp_seq(s)
        z = z + self.sequence_to_pair(s)
        pair_mask = (mask[:, :, None] * mask[:, None, :]).to(z.dtype)
        for operation in (self.tri_mul_out, self.tri_mul_in, self.tri_att_start, self.tri_att_end):
            z = z + operation(z, mask=pair_mask)
        return s, self.mlp_pair(z)


class AngleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1, self.linear_2, self.relu = Linear(dim, dim), Linear(dim, dim), ReLU()

    def forward(self, x):
        return x + self.linear_2(self.relu(self.linear_1(self.relu(x))))


class Angles(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_in, self.linear_initial = Linear(config.sequence_dim, config.resnet_dim), Linear(config.sequence_dim, config.resnet_dim)
        self.layers = nn.ModuleList(AngleBlock(config.resnet_dim) for _ in range(config.num_resnet_blocks))
        self.linear_out, self.relu = Linear(config.resnet_dim, config.num_angles * 2), ReLU()
        self.normalize = EsmFoldL2Norm(eps=config.epsilon)

    def forward(self, x, initial):
        x = self.linear_in(self.relu(x)) + self.linear_initial(self.relu(initial))
        for layer in self.layers:
            x = layer(x)
        x = self.linear_out(self.relu(x)).reshape(*x.shape[:-1], -1, 2)
        return x, self.normalize(x)


class TransitionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1, self.linear_2, self.linear_3 = Linear(dim, dim), Linear(dim, dim), Linear(dim, dim)
        self.relu = ReLU()

    def forward(self, x):
        return x + self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(x)))))


class Transition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(TransitionLayer(config.sequence_dim) for _ in range(config.num_transition_layers))
        self.layer_norm = norm(config.sequence_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class PointAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        s, z, h, d, q, v = config.sequence_dim, config.pairwise_dim, config.num_heads_ipa, config.ipa_dim, config.num_qk_points, config.num_v_points
        self.linear_q, self.linear_kv = Linear(s, h * d), Linear(s, 2 * h * d)
        self.linear_q_points, self.linear_kv_points = Linear(s, h * q * 3), Linear(s, h * (q + v) * 3)
        self.linear_b, self.linear_out = Linear(z, h), Linear(h * (z + d + 4 * v), s)
        self.head_weights = nn.Parameter(torch.empty(h))
        self.register_buffer('prepared_head_weights', torch.empty(h), persistent=False)
        self.arithmetic, self.softmax = Arithmetic(), Softmax()

    def forward(self, s, z, quaternion, translation, mask):
        c, op = self.config, self.arithmetic
        h, d, nq, nv = c.num_heads_ipa, c.ipa_dim, c.num_qk_points, c.num_v_points
        q = self.linear_q(s).reshape(*s.shape[:-1], h, d)
        k, v = self.linear_kv(s).reshape(*s.shape[:-1], h, 2 * d).chunk(2, -1)
        rotation = _quat_to_rot(quaternion)
        def points(projection):
            xyz = torch.stack(projection.chunk(3, -1), -1)
            xyz = op.rotate(rotation[..., None, :, :], xyz) + translation[..., None, :]
            return xyz.reshape(*s.shape[:-1], h, -1, 3)
        qp = points(self.linear_q_points(s))
        kp, vp = points(self.linear_kv_points(s)).split((nq, nv), -2)
        a = op.matmul(q.transpose(-2, -3), k.permute(0, 2, 3, 1)) * math.sqrt(1 / (3 * d))
        a = a + self.linear_b(z).permute(0, 3, 1, 2) * math.sqrt(1 / 3)
        delta = qp.unsqueeze(-4) - kp.unsqueeze(-5)
        distance = op.sum(op.mul(delta, delta))
        distance = op.mul(distance, self.prepared_head_weights[:, None])
        distance = op.sum(distance) * -0.5
        a = a + distance.permute(0, 3, 1, 2)
        a = self.softmax(a + (c.inf * (mask[:, :, None] * mask[:, None, :] - 1))[:, None])
        scalar = op.matmul(a, v.transpose(-2, -3).to(a.dtype)).transpose(-2, -3).flatten(-2)
        # Natural attention matrix product over residue keys for point values.
        weighted = op.matmul(a, vp.permute(0, 2, 1, 3, 4).flatten(-2)).reshape(*a.shape[:2], s.shape[1], nv, 3)
        weighted = weighted.permute(0, 2, 1, 3, 4)
        local = op.rotate(rotation.transpose(-1, -2)[..., None, None, :, :], weighted - translation[..., None, None, :])
        length = op.sqrt(op.sum(op.mul(local, local)), c.epsilon).flatten(-2)
        local = local.flatten(-3, -2)
        pair = op.matmul(a.transpose(-2, -3), z.to(a.dtype)).flatten(-2)
        return self.linear_out(torch.cat((scalar, *local.unbind(-1), length, pair), -1).to(z.dtype))


class Structure(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm_s, self.layer_norm_z = norm(config.sequence_dim), norm(config.pairwise_dim)
        self.linear_in, self.ipa = Linear(config.sequence_dim, config.sequence_dim), PointAttention(config)
        self.layer_norm_ipa, self.transition = norm(config.sequence_dim), Transition(config)
        self.bb_update, self.angle_resnet = group(linear=Linear(config.sequence_dim, 6)), Angles(config)
        self.arithmetic = Arithmetic()

    def geometry(self, quaternion, translation, angles, aa):
        op = self.arithmetic
        # Native default-frame constants first round to angle dtype; all rigid
        # matrices then promote to FP32 in Rotation's constructor.
        frames = torch.tensor(_DEFAULT_FRAMES, device=aa.device, dtype=angles.dtype)[aa].float()
        backbone_angle = angles.new_zeros(*angles.shape[:-2], 1, 2)
        backbone_angle[..., 1] = 1
        alpha = torch.cat((backbone_angle, angles), -2)
        rotations = alpha.new_zeros(*alpha.shape[:-1], 3, 3)
        rotations[..., 0, 0] = 1
        rotations[..., 1, 1], rotations[..., 1, 2] = alpha[..., 1], -alpha[..., 0]
        rotations[..., 2, 1:] = alpha
        rotations = op.matmul(frames[..., :3, :3], rotations.float())
        translations = frames[..., :3, 3]
        pieces = [(rotations[..., :5, :, :], translations[..., :5, :])]
        previous = rotations[..., 4, :, :], translations[..., 4, :]
        for index in (5, 6, 7):
            previous = op.compose(previous, (rotations[..., index, :, :], translations[..., index, :]))
            pieces.append((previous[0].unsqueeze(-3), previous[1].unsqueeze(-2)))
        rotations, translations = torch.cat([p[0] for p in pieces], -3), torch.cat([p[1] for p in pieces], -2)
        rotations, translations = op.compose((_quat_to_rot(quaternion)[..., None, :, :], translation[..., None, :] * self.config.trans_scale_factor), (rotations, translations))
        sidechain = rotations.new_zeros(*rotations.shape[:-2], 4, 4)
        sidechain[..., :3, :3], sidechain[..., :3, 3], sidechain[..., 3, 3] = rotations, translations, 1
        indices = torch.tensor(_GROUP_IDX, device=aa.device)[aa]
        atom_r = torch.gather(rotations, -3, indices[..., None, None].expand(*indices.shape, 3, 3))
        atom_t = torch.gather(translations, -2, indices[..., None].expand(*indices.shape, 3))
        # The native lazily-created literature positions retain angle dtype.
        literature = torch.tensor(_LIT_POSITIONS, device=aa.device, dtype=angles.dtype)[aa].float()
        xyz = op.rotate(atom_r, literature) + atom_t
        atom_mask = torch.tensor(_ATOM14_MASK, device=aa.device, dtype=torch.bool)[aa]
        return sidechain, xyz.masked_fill(~atom_mask[..., None], 0)

    def forward(self, s, z, aa, mask):
        s, z = self.layer_norm_s(s), self.layer_norm_z(z)
        initial, s = s, self.linear_in(s)
        quaternion = torch.zeros(*s.shape[:-1], 4, device=s.device)
        quaternion[..., 0] = 1
        translation = torch.zeros(*s.shape[:-1], 3, device=s.device)
        results = []
        for _ in range(self.config.num_blocks):
            s = s + self.ipa(s, z, quaternion, translation, mask.float())
            s = self.transition(self.layer_norm_ipa(s))
            quaternion, translation = self.arithmetic.update(quaternion, translation, self.bb_update.linear(s))
            unnormalized, angles = self.angle_resnet(s, initial)
            sidechain, xyz = self.geometry(quaternion, translation, angles, aa)
            results.append(dict(frames=torch.cat((quaternion, translation * self.config.trans_scale_factor), -1),
                                sidechain_frames=sidechain, unnormalized_angles=unnormalized, angles=angles, positions=xyz, states=s))
        return {key: torch.stack([r[key] for r in results]) for key in results[0]}


class Trunk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        s, z = config.sequence_state_dim, config.pairwise_state_dim
        self.pairwise_positional_embedding = group(embedding=Embedding(2 * config.position_bins + 2, z))
        self.blocks = nn.ModuleList(TriangleBlock(config) for _ in range(config.num_blocks))
        self.recycle_s_norm, self.recycle_z_norm, self.recycle_disto = norm(s), norm(z), Embedding(15, z)
        self.structure_module = Structure(config.structure_module)
        self.trunk2sm_s, self.trunk2sm_z = Linear(s, config.structure_module.sequence_dim), Linear(z, config.structure_module.pairwise_dim)
        self.arithmetic = Arithmetic()

    def distogram(self, coordinates):
        op = self.arithmetic
        n, ca, c = coordinates.unbind(-2)
        b, c = ca - n, c - ca
        # Two three-coordinate packed products are the native cross product.
        cross = op.mul(b[..., [1, 2, 0]], c[..., [2, 0, 1]]) - op.mul(b[..., [2, 0, 1]], c[..., [1, 2, 0]])
        cb = -0.58273431 * cross + 0.56802827 * b - 0.54067466 * c + ca
        delta = cb[..., None, :, :] - cb[..., :, None, :]
        distance = op.sum(op.mul(delta, delta))
        boundaries = torch.linspace(3.375, 21.375, 14, device=cb.device).square()
        thresholds, distance = torch.broadcast_tensors(boundaries, distance[..., None])
        # Threshold first gives strict > with first-index tie breaking.
        predicates = op.select(torch.stack((thresholds, distance), -1))
        return op.sum(predicates.float()).long()

    def forward(self, initial_s, initial_z, aa, positions, mask):
        s, z = torch.zeros_like(initial_s), torch.zeros_like(initial_z)
        bins = torch.zeros(initial_z.shape[:-1], device=aa.device, dtype=torch.long)
        delta = (positions[:, None, :] - positions[:, :, None]).clamp(-self.config.position_bins, self.config.position_bins)
        delta = delta + self.config.position_bins + 1
        delta = delta.masked_fill(~(mask[:, None, :].bool() & mask[:, :, None].bool()), 0)
        for _ in range(self.config.max_recycles):
            s = initial_s + self.recycle_s_norm(s)
            z = initial_z + (self.recycle_z_norm(z) + self.recycle_disto(bins))
            z = z + self.pairwise_positional_embedding.embedding(delta)
            for block in self.blocks:
                s, z = block(s, z, mask)
            structure = self.structure_module(self.trunk2sm_s(s), self.trunk2sm_z(z), aa, mask)
            bins = self.distogram(structure['positions'][-1, :, :, :3])
        structure.update(s_s=s, s_z=z)
        return structure


class EsmFold(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        fold = config.esmfold_config
        s, z = fold.trunk.sequence_state_dim, fold.trunk.pairwise_state_dim
        self.esm = EsmStem(config)
        self.esm_s_combine = nn.Parameter(torch.empty(config.num_hidden_layers + 1))
        self.register_buffer('af2_to_esm', torch.empty(22, dtype=torch.long))
        self.esm_s_mlp = nn.Sequential(norm(config.hidden_size), Linear(config.hidden_size, s), ReLU(), Linear(s, s))
        self.embedding, self.trunk = Embedding(23, s, padding_idx=0), Trunk(fold.trunk)
        self.distogram_head, self.ptm_head, self.lm_head = Linear(z, 64), Linear(z, 64), Linear(s, 23)
        d = fold.lddt_head_hid_dim
        self.lddt_head = nn.Sequential(norm(fold.trunk.structure_module.sequence_dim), Linear(fold.trunk.structure_module.sequence_dim, d), Linear(d, d), Linear(d, 37 * 50))
        self.arithmetic, self.softmax = Arithmetic(), Softmax()

    def forward(self, input_ids, attention_mask=None):
        aa, op = input_ids, self.arithmetic
        mask = torch.ones_like(aa) if attention_mask is None else attention_mask
        positions = torch.arange(aa.shape[1], device=aa.device)[None].expand_as(aa)
        ids = self.af2_to_esm[(aa + 1).masked_fill(mask != 1, 0)]
        ids = torch.cat((torch.zeros_like(ids[:, :1]), ids, torch.ones_like(ids[:, :1])), -1)
        ids[torch.arange(aa.shape[0], device=aa.device), (ids != 1).sum(1)] = 2
        language = self.esm(ids)[:, 1:-1].to(self.esm_s_combine.dtype)
        language = op.matmul(self.softmax(self.esm_s_combine)[None], language).squeeze(2)
        s = self.esm_s_mlp(language) + self.embedding(aa)
        z = s.new_zeros(*aa.shape, aa.shape[1], self.config.esmfold_config.trunk.pairwise_state_dim)
        out = self.trunk(s, z, aa, positions, mask)
        distogram = self.distogram_head(out['s_z'])
        out.update(distogram_logits=(distogram + distogram.transpose(1, 2)) * 0.5, lm_logits=self.lm_head(out['s_s']),
                   aatype=aa, residue_index=positions)
        for key, table in [('atom14_atom_exists', _ATOM14_MASK), ('atom37_atom_exists', _ATOM37_MASK),
                           ('residx_atom14_to_atom37', _ATOM14_TO_37), ('residx_atom37_to_atom14', _ATOM37_TO_14)]:
            dtype = torch.float32 if key.endswith('exists') else torch.long
            value = torch.tensor(table, device=aa.device, dtype=dtype)[aa]
            out[key] = value.masked_fill(~mask[..., None].bool(), 0) if key.endswith('exists') else value
        lddt = self.lddt_head(out['states']).reshape(*out['states'].shape[:-1], 37, 50)
        edges = torch.linspace(0, 1, 51, device=aa.device, dtype=lddt.dtype)
        out['lddt_head'] = lddt
        out['plddt'] = op.matmul(self.softmax(lddt[-1]), ((edges[:-1] + edges[1:]) * 0.5)[:, None]).squeeze(-1)
        logits = self.ptm_head(out['s_z'])
        out['ptm_logits'] = logits
        probabilities = self.softmax(logits)
        centers = torch.arange(64, device=aa.device).float() * 0.5 + 0.25
        out['aligned_confidence_probs'] = probabilities
        out['predicted_aligned_error'] = op.sum(op.mul(probabilities.float(), centers))
        out['max_predicted_aligned_error'] = centers[-1]
        # All residue weights are one in native compute_tm's ordinary call.
        d0 = 1.24 * (max(aa.shape[1], 19) - 15) ** (1 / 3) - 1.8
        tm_kernel = 1.0 / (1 + centers.square() / d0**2)
        term = op.sum(op.mul(probabilities.float(), tm_kernel))
        # Shape metadata follows native residue-weight reduction rounding too:
        # e.g. BF16 length257 sums to256 before the reciprocal is computed.
        normalization = residue_weight_scale(aa.shape[1], logits.dtype)
        alignment = op.sum(term * normalization)
        out['ptm'] = alignment.flatten()[op.select(alignment.flatten())]
        return out


def build_from_config(config, device, dtype):
    fold = config.esmfold_config
    if (config.position_embedding_type != 'rotary' or not config.token_dropout or config.emb_layer_norm_before
            or config.is_decoder or config.add_cross_attention or fold.fp16_esm or fold.use_esm_attn_map
            or fold.esm_ablate_sequence or fold.esm_ablate_pairwise or fold.bypass_lm or not fold.embed_aa
            or fold.trunk.chunk_size is not None):
        raise ValueError('ESMFold construction requires the documented esmfold_v1 active computation')
    model = EsmFold(config).to(device=device, dtype=dtype).eval()
    # Native loading retains inv_freq in FP32; positional trig rounds at use.
    width = config.hidden_size // config.num_attention_heads
    model.esm.rotary = RotaryEmbedding(width, config.max_position_embeddings, config.rope_theta).to(device=device)
    model.esm.rotary_embeddings.inv_freq = model.esm.rotary_embeddings.inv_freq.float()
    return model


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if '.tri_att_' in source:
            source = source.replace('.linear_z.', '.linear.')
        mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped ESMFold state: {sorted(remaining)}')
    model.load_state_dict(mapped, strict=True)
    positions = torch.arange(config.max_position_embeddings, device=model.esm.rotary_embeddings.inv_freq.device).float()
    frequencies = positions[:, None] * model.esm.rotary_embeddings.inv_freq[None].float()
    model.esm.rotary.cos_sin_cache = torch.cat((frequencies.cos(), frequencies.sin()), -1)
    point = model.trunk.structure_module.ipa
    # Inference-constant learned scalar/vector transform, excluded from timing.
    point.prepared_head_weights = torch.nn.functional.softplus(point.head_weights.detach()) * math.sqrt(1 / (3 * (point.config.num_qk_points * 9 / 2)))


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}


# Fixed residue geometry/mapping data only: HF da6c53e431f7c9ef0691239d4ce89b0f711ecad7
# models/esm/openfold_utils/residue_constants.py and data_transforms.make_atom14_masks.
# OpenFold / DeepMind, Apache-2.0. No reference numerical functions are imported.
_DEFAULT_FRAMES = [
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.35943782329559326, 0.9331690669059753, 0.0, -0.5249999761581421], [0.9331690669059753, 0.35943782329559326, 0.0, 1.3630000352859497], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.5260000228881836], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.35907092690467834, 0.9333102703094482, 0.0, -0.5239999890327454], [0.9333102703094482, 0.35907092690467834, 0.0, 1.3619999885559082], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.524999976158142], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3424367904663086, -0.5121516585350037, 0.7876787185668945, -0.5239999890327454], [-0.5084271430969238, 0.8060193657875061, 0.30304232239723206, -0.777999997138977], [-0.7900879383087158, -0.29670438170433044, -0.5364024639129639, -1.2089999914169312], [0.0, 0.0, 0.0, 1.0]], [[0.4051617980003357, -0.9142450094223022, 0.0, 0.6159999966621399], [0.9142450094223022, 0.4051617980003357, 0.0, 1.3899999856948853], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.37048444151878357, -0.9288386702537537, 0.0, 0.5640000104904175], [0.9288386702537537, 0.37048444151878357, -0.0, 1.4140000343322754], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.36914604902267456, -0.9293714165687561, 0.0, 0.5389999747276306], [0.9293714165687561, 0.36914604902267456, 0.0, 1.3569999933242798], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3673693835735321, 0.9300751090049744, 0.0, -0.5360000133514404], [0.9300751090049744, 0.3673693835735321, 0.0, 1.3569999933242798], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.5260000228881836], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3470269441604614, -0.5223447680473328, 0.7789276242256165, -0.531000018119812], [-0.5143318772315979, 0.8005018830299377, 0.3076677918434143, -0.7870000004768372], [-0.7842416763305664, -0.29385825991630554, -0.5464542508125305, -1.2000000476837158], [0.0, 0.0, 0.0, 1.0]], [[0.38522419333457947, -0.9228230118751526, 0.0, 0.5839999914169312], [0.9228230118751526, 0.38522419333457947, -0.0, 1.3990000486373901], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3596675992012024, 0.9330804944038391, 0.0, -0.5249999761581421], [0.9330804944038391, 0.3596675992012024, 0.0, 1.3619999885559082], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.5269999504089355], [0.0, -1.0, -0.0, 0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.34376704692840576, -0.5128490328788757, 0.7866448163986206, -0.5260000228881836], [-0.5084615349769592, 0.8059300184249878, 0.303222119808197, -0.777999997138977], [-0.7894878387451172, -0.29574087262153625, -0.5378162264823914, -1.2079999446868896], [0.0, 0.0, 0.0, 1.0]], [[0.3904991149902344, -0.9206032752990723, 0.0, 0.5929999947547913], [0.9206032752990723, 0.3904991149902344, 0.0, 1.3980000019073486], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3578762412071228, 0.9337690472602844, 0.0, -0.5220000147819519], [0.9337690472602844, 0.3578762412071228, 0.0, 1.3619999885559082], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.5240000486373901], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3395833373069763, -0.5093438625335693, 0.790728747844696, -0.5189999938011169], [-0.5057763457298279, 0.8076807260513306, 0.3030546307563782, -0.7730000019073486], [-0.7930154204368591, -0.2970196008682251, -0.531889021396637, -1.2120000123977661], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.36049413681030273, 0.9327614903450012, 0.0, -0.5260000228881836], [0.9327614903450012, 0.36049413681030273, 0.0, 1.3609999418258667], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.5260000228881836], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.343253493309021, -0.5138495564460754, 0.7862160801887512, -0.5249999761581421], [-0.5093227624893188, 0.8051466345787048, 0.3038572072982788, -0.7789999842643738], [-0.7891560792922974, -0.2961377203464508, -0.5380846858024597, -1.2070000171661377], [0.0, 0.0, 0.0, 1.0]], [[0.4038827121257782, -0.9148107767105103, 0.0, 0.6150000095367432], [0.9148107767105103, 0.4038827121257782, -0.0, 1.3930000066757202], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.3869074881076813, -0.9221185445785522, 0.0, 0.5870000123977661], [0.9221185445785522, 0.3869074881076813, 0.0, 1.3990000486373901], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3616858422756195, 0.9323000311851501, 0.0, -0.527999997138977], [0.9323000311851501, 0.3616858422756195, 0.0, 1.3609999418258667], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.5260000228881836], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3436011075973511, -0.5152674317359924, 0.7851355075836182, -0.5260000228881836], [-0.5101758241653442, 0.8043280243873596, 0.30459335446357727, -0.781000018119812], [-0.7884535193443298, -0.29589852690696716, -0.5392449498176575, -1.2070000171661377], [0.0, 0.0, 0.0, 1.0]], [[0.40412548184394836, -0.9147035479545593, 0.0, 0.6150000095367432], [0.9147035479545593, 0.40412548184394836, -0.0, 1.3919999599456787], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.39463359117507935, -0.9188385605812073, 0.0, 0.6000000238418579], [0.9188385605812073, 0.39463359117507935, -0.0, 1.3969999551773071], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.39333826303482056, 0.9193938374519348, 0.0, -0.5720000267028809], [0.9193938374519348, 0.39333826303482056, 0.0, 1.3370000123977661], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.5169999599456787], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3613210618495941, 0.9324414730072021, 0.0, -0.5270000100135803], [0.9324414730072021, 0.3613210618495941, 0.0, 1.3600000143051147], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.524999976158142], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3431905508041382, -0.5140678882598877, 0.7861008048057556, -0.5249999761581421], [-0.5085757374763489, 0.8053328394889832, 0.3046140670776367, -0.777999997138977], [-0.7896651029586792, -0.29525110125541687, -0.5378250479698181, -1.2079999446868896], [0.0, 0.0, 0.0, 1.0]], [[0.4011695683002472, -0.9160038232803345, 0.0, 0.6000000238418579], [0.9160038232803345, 0.4011695683002472, 0.0, 1.3700000047683716], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.33794260025024414, 0.9411666989326477, 0.0, -0.49300000071525574], [0.9411666989326477, 0.33794260025024414, 0.0, 1.3730000257492065], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.5269999504089355], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.34689003229141235, -0.49944600462913513, 0.7938646078109741, -0.5360000133514404], [-0.5132160186767578, 0.8095400929450989, 0.285051167011261, -0.7929999828338623], [-0.7850328683853149, -0.3085426390171051, -0.5371450781822205, -1.2130000591278076], [0.0, 0.0, 0.0, 1.0]], [[0.3483339250087738, -0.9373705387115479, 0.0, 0.5339999794960022], [0.9373705387115479, 0.3483339250087738, 0.0, 1.437000036239624], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3564513623714447, 0.9343138933181763, 0.0, -0.5199999809265137], [0.9343138933181763, 0.3564513623714447, 0.0, 1.3630000352859497], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.524999976158142], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.340964674949646, -0.5080348253250122, 0.7909764051437378, -0.5220000147819519], [-0.5049151182174683, 0.80870121717453, 0.3017664849758148, -0.7730000019073486], [-0.7929714918136597, -0.29648423194885254, -0.5322530269622803, -1.2139999866485596], [0.0, 0.0, 0.0, 1.0]], [[0.4432864189147949, -0.8963800072669983, 0.0, 0.6779999732971191], [0.8963800072669983, 0.4432864189147949, -0.0, 1.371000051498413], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3602638244628906, 0.9328504800796509, 0.0, -0.5260000228881836], [0.9328504800796509, 0.3602638244628906, 0.0, 1.3619999885559082], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.5260000228881836], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3426136374473572, -0.5132287740707397, 0.7869003415107727, -0.5239999890327454], [-0.5086897611618042, 0.8055312037467957, 0.3038983643054962, -0.777999997138977], [-0.7898421883583069, -0.29616838693618774, -0.5370602011680603, -1.2079999446868896], [0.0, 0.0, 0.0, 1.0]], [[0.40680912137031555, -0.91351318359375, 0.0, 0.6190000176429749], [0.91351318359375, 0.40680912137031555, -0.0, 1.3899999856948853], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.3669722080230713, -0.9302319288253784, 0.0, 0.5590000152587891], [0.9302319288253784, 0.3669722080230713, -0.0, 1.4170000553131104], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.36776456236839294, -0.9299189448356628, 0.0, 0.5600000023841858], [0.9299189448356628, 0.36776456236839294, -0.0, 1.4160000085830688], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.35682111978530884, 0.9341727495193481, 0.0, -0.5210000276565552], [0.9341727495193481, 0.35682111978530884, 0.0, 1.3639999628067017], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.524999976158142], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3419100344181061, -0.5097509622573853, 0.7894628047943115, -0.5230000019073486], [-0.5073081851005554, 0.8072841763496399, 0.30154699087142944, -0.7760000228881836], [-0.7910346984863281, -0.29739901423454285, -0.5346194505691528, -1.2100000381469727], [0.0, 0.0, 0.0, 1.0]], [[0.40326765179634094, -0.9150820970535278, 0.0, 0.6129999756813049], [0.9150820970535278, 0.40326765179634094, 0.0, 1.3910000324249268], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.3831057548522949, -0.9237045049667358, 0.0, 0.703000009059906], [0.9237045049667358, 0.3831057548522949, -0.0, 1.6950000524520874], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3552537262439728, 0.9347699284553528, 0.0, -0.5180000066757202], [0.9347699284553528, 0.3552537262439728, 0.0, 1.3630000352859497], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.5240000486373901], [0.0, -1.0, -0.0, 0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.34270966053009033, -0.5082933306694031, 0.7900556921958923, -0.5249999761581421], [-0.5065575242042542, 0.8082362413406372, 0.3002559542655945, -0.7760000228881836], [-0.7911697626113892, -0.2973080575466156, -0.5344702005386353, -1.2120000123977661], [0.0, 0.0, 0.0, 1.0]], [[0.40336206555366516, -0.9150404334068298, 0.0, 0.6069999933242798], [0.9150404334068298, 0.40336206555366516, -0.0, 1.3769999742507935], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.386408269405365, 0.9223278164863586, 0.0, -0.5659999847412109], [0.9223278164863586, 0.386408269405365, 0.0, 1.3509999513626099], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.5269999504089355], [-0.0, -1.0, 0.0, -0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3566810190677643, -0.4815024733543396, 0.80058354139328, -0.5460000038146973], [-0.3991430401802063, 0.8533400297164917, 0.3354036211967468, -0.6110000014305115], [-0.8446676731109619, -0.19991524517536163, -0.4965585768222809, -1.2929999828338623], [0.0, 0.0, 0.0, 1.0]], [[0.2555799186229706, -0.9667879343032837, 0.0, 0.38199999928474426], [0.9667879343032837, 0.2555799186229706, -0.0, 1.4450000524520874], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.36251240968704224, 0.9319789409637451, 0.0, -0.5289999842643738], [0.9319789409637451, 0.36251240968704224, 0.0, 1.3600000143051147], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.524999976158142], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.33873042464256287, -0.5139699578285217, 0.7880967855453491, -0.5180000066757202], [-0.5080956220626831, 0.8049025535583496, 0.30654647946357727, -0.7770000100135803], [-0.7918968200683594, -0.2965919077396393, -0.5337908864021301, -1.2109999656677246], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.35442689061164856, 0.9350837469100952, 0.0, -0.5170000195503235], [0.9350837469100952, 0.35442689061164856, 0.0, 1.3639999628067017], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.5260000228881836], [0.0, -1.0, -0.0, 0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3350840210914612, -0.5107815265655518, 0.7917202711105347, -0.515999972820282], [-0.5149644017219543, 0.8029689788818359, 0.30008751153945923, -0.7929999828338623], [-0.7890059351921082, -0.30715319514274597, -0.5320963263511658, -1.215000033378601], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.35704952478408813, 0.9340854287147522, 0.0, -0.5210000276565552], [0.9340854287147522, 0.35704952478408813, 0.0, 1.3630000352859497], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.524999976158142], [-0.0, -1.0, 0.0, -0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3415566682815552, -0.509585440158844, 0.7897225618362427, -0.5230000019073486], [-0.5067839026451111, 0.8074936866760254, 0.3018675446510315, -0.7760000228881836], [-0.791523277759552, -0.2971138060092926, -0.5340545773506165, -1.2120000123977661], [0.0, 0.0, 0.0, 1.0]], [[0.4062003493309021, -0.9137840270996094, 0.0, 0.609000027179718], [0.9137840270996094, 0.4062003493309021, 0.0, 1.3700000047683716], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.3578762412071228, 0.9337690472602844, 0.0, -0.5220000147819519], [0.9337690472602844, 0.3578762412071228, 0.0, 1.3619999885559082], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.5240000486373901], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.340803325176239, -0.5100081562995911, 0.7897751331329346, -0.5220000147819519], [-0.5066348314285278, 0.807279646396637, 0.30268916487693787, -0.7760000228881836], [-0.791943371295929, -0.2969701290130615, -0.5335114598274231, -1.2130000591278076], [0.0, 0.0, 0.0, 1.0]], [[0.4021390676498413, -0.9155786037445068, 0.0, 0.6069999933242798], [0.9155786037445068, 0.4021390676498413, 0.0, 1.3819999694824219], [-0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.33854958415031433, 0.9409485459327698, 0.0, -0.49399998784065247], [0.9409485459327698, 0.33854958415031433, 0.0, 1.3730000257492065], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, -0.0, 1.5269999504089355], [-0.0, -1.0, -0.0, -0.0], [-0.0, 0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]], [[-0.34495073556900024, -0.5002416968345642, 0.7942085862159729, -0.5329999923706055], [-0.5145137310028076, 0.8084681630134583, 0.28575313091278076, -0.7950000166893005], [-0.7850379347801208, -0.3100604712963104, -0.5362629294395447, -1.2130000591278076], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
]
_GROUP_IDX = [
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 6, 7, 7, 7, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 4, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 6, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 3, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
_LIT_POSITIONS = [
    [[-0.5249999761581421, 1.3630000352859497, 0.0], [0.0, 0.0, 0.0], [1.5260000228881836, -0.0, -0.0], [0.6269999742507935, 1.062000036239624, 0.0], [-0.5289999842643738, -0.7739999890327454, -1.2050000429153442], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5239999890327454, 1.3619999885559082, -0.0], [0.0, 0.0, 0.0], [1.524999976158142, -0.0, -0.0], [0.6259999871253967, 1.062000036239624, 0.0], [-0.5239999890327454, -0.777999997138977, -1.2089999914169312], [0.6159999966621399, 1.3899999856948853, -0.0], [0.5640000104904175, 1.4140000343322754, 0.0], [0.5389999747276306, 1.3569999933242798, -0.0], [0.7580000162124634, 1.093000054359436, -0.0], [0.20600000023841858, 2.3010001182556152, 0.0], [2.078000068664551, 0.9779999852180481, -0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5360000133514404, 1.3569999933242798, 0.0], [0.0, 0.0, 0.0], [1.5260000228881836, -0.0, -0.0], [0.625, 1.062000036239624, 0.0], [-0.531000018119812, -0.7870000004768372, -1.2000000476837158], [0.5839999914169312, 1.3990000486373901, 0.0], [0.6330000162124634, 1.059000015258789, 0.0], [0.5929999947547913, -1.187999963760376, 0.0010000000474974513], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5249999761581421, 1.3619999885559082, -0.0], [0.0, 0.0, 0.0], [1.5269999504089355, 0.0, -0.0], [0.6259999871253967, 1.062000036239624, -0.0], [-0.5260000228881836, -0.777999997138977, -1.2079999446868896], [0.5929999947547913, 1.3980000019073486, -0.0], [0.6100000143051147, 1.090999960899353, 0.0], [0.5920000076293945, -1.1009999513626099, -0.003000000026077032], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5220000147819519, 1.3619999885559082, -0.0], [0.0, 0.0, 0.0], [1.5240000486373901, 0.0, 0.0], [0.625, 1.062000036239624, -0.0], [-0.5189999938011169, -0.7730000019073486, -1.2120000123977661], [0.7279999852180481, 1.652999997138977, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5260000228881836, 1.3609999418258667, -0.0], [0.0, 0.0, 0.0], [1.5260000228881836, 0.0, 0.0], [0.6259999871253967, 1.062000036239624, -0.0], [-0.5249999761581421, -0.7789999842643738, -1.2070000171661377], [0.6150000095367432, 1.3930000066757202, 0.0], [0.5870000123977661, 1.3990000486373901, -0.0], [0.6340000033378601, 1.059999942779541, 0.0], [0.5929999947547913, -1.1890000104904175, -0.0010000000474974513], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.527999997138977, 1.3609999418258667, 0.0], [0.0, 0.0, 0.0], [1.5260000228881836, -0.0, -0.0], [0.6259999871253967, 1.062000036239624, 0.0], [-0.5260000228881836, -0.781000018119812, -1.2070000171661377], [0.6150000095367432, 1.3919999599456787, 0.0], [0.6000000238418579, 1.3969999551773071, 0.0], [0.6069999933242798, 1.0950000286102295, -0.0], [0.5889999866485596, -1.1039999723434448, -0.0010000000474974513], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5720000267028809, 1.3370000123977661, 0.0], [0.0, 0.0, 0.0], [1.5169999599456787, -0.0, -0.0], [0.6259999871253967, 1.062000036239624, -0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5270000100135803, 1.3600000143051147, 0.0], [0.0, 0.0, 0.0], [1.524999976158142, 0.0, 0.0], [0.625, 1.062999963760376, 0.0], [-0.5249999761581421, -0.777999997138977, -1.2079999446868896], [0.6000000238418579, 1.3700000047683716, -0.0], [0.7440000176429749, 1.159999966621399, -0.0], [0.8889999985694885, -1.0210000276565552, 0.003000000026077032], [2.0299999713897705, 0.8510000109672546, 0.0020000000949949026], [2.1449999809265137, -0.4659999907016754, 0.004000000189989805], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.49300000071525574, 1.3730000257492065, -0.0], [0.0, 0.0, 0.0], [1.5269999504089355, -0.0, -0.0], [0.6269999742507935, 1.062000036239624, -0.0], [-0.5360000133514404, -0.7929999828338623, -1.2130000591278076], [0.5339999794960022, 1.437000036239624, -0.0], [0.5400000214576721, -0.7850000262260437, -1.1990000009536743], [0.6190000176429749, 1.3910000324249268, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5199999809265137, 1.3630000352859497, 0.0], [0.0, 0.0, 0.0], [1.524999976158142, -0.0, -0.0], [0.625, 1.062999963760376, -0.0], [-0.5220000147819519, -0.7730000019073486, -1.2139999866485596], [0.6779999732971191, 1.371000051498413, 0.0], [0.5299999713897705, 1.4299999475479126, -0.0], [0.5350000262260437, -0.7739999890327454, 1.2000000476837158], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5260000228881836, 1.3619999885559082, -0.0], [0.0, 0.0, 0.0], [1.5260000228881836, 0.0, 0.0], [0.6259999871253967, 1.062000036239624, -0.0], [-0.5239999890327454, -0.777999997138977, -1.2079999446868896], [0.6190000176429749, 1.3899999856948853, 0.0], [0.5590000152587891, 1.4170000553131104, 0.0], [0.5600000023841858, 1.4160000085830688, 0.0], [0.5540000200271606, 1.3869999647140503, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5210000276565552, 1.3639999628067017, -0.0], [0.0, 0.0, 0.0], [1.524999976158142, 0.0, 0.0], [0.625, 1.062000036239624, -0.0], [-0.5230000019073486, -0.7760000228881836, -1.2100000381469727], [0.6129999756813049, 1.3910000324249268, -0.0], [0.703000009059906, 1.6950000524520874, 0.0], [0.3199999928474426, 1.7860000133514404, -0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5180000066757202, 1.3630000352859497, 0.0], [0.0, 0.0, 0.0], [1.5240000486373901, 0.0, -0.0], [0.6259999871253967, 1.062000036239624, -0.0], [-0.5249999761581421, -0.7760000228881836, -1.2120000123977661], [0.6069999933242798, 1.3769999742507935, 0.0], [0.7089999914169312, 1.1950000524520874, -0.0], [0.7059999704360962, -1.1959999799728394, 0.0], [2.1019999980926514, 1.1979999542236328, -0.0], [2.0980000495910645, -1.2009999752044678, -0.0], [2.7939999103546143, -0.003000000026077032, -0.0010000000474974513], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5659999847412109, 1.3509999513626099, -0.0], [0.0, 0.0, 0.0], [1.5269999504089355, -0.0, 0.0], [0.6209999918937683, 1.065999984741211, 0.0], [-0.5460000038146973, -0.6110000014305115, -1.2929999828338623], [0.38199999928474426, 1.4450000524520874, 0.0], [0.47699999809265137, 1.4240000247955322, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5289999842643738, 1.3600000143051147, -0.0], [0.0, 0.0, 0.0], [1.524999976158142, -0.0, -0.0], [0.6259999871253967, 1.062000036239624, -0.0], [-0.5180000066757202, -0.7770000100135803, -1.2109999656677246], [0.503000020980835, 1.3250000476837158, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5170000195503235, 1.3639999628067017, 0.0], [0.0, 0.0, 0.0], [1.5260000228881836, 0.0, -0.0], [0.6259999871253967, 1.062000036239624, 0.0], [-0.515999972820282, -0.7929999828338623, -1.215000033378601], [0.47200000286102295, 1.3530000448226929, 0.0], [0.550000011920929, -0.7179999947547913, -1.2280000448226929], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.5210000276565552, 1.3630000352859497, 0.0], [0.0, 0.0, 0.0], [1.524999976158142, -0.0, 0.0], [0.6269999742507935, 1.062000036239624, 0.0], [-0.5230000019073486, -0.7760000228881836, -1.2120000123977661], [0.609000027179718, 1.3700000047683716, -0.0], [0.8240000009536743, 1.090999960899353, 0.0], [0.8539999723434448, -1.1480000019073486, -0.004999999888241291], [2.140000104904175, 0.6899999976158142, -0.004000000189989805], [2.186000108718872, -0.6779999732971191, -0.007000000216066837], [0.621999979019165, -2.5299999713897705, -0.007000000216066837], [3.2829999923706055, -1.5429999828338623, -0.010999999940395355], [1.715000033378601, -3.3889999389648438, -0.010999999940395355], [3.0280001163482666, -2.890000104904175, -0.013000000268220901]],
    [[-0.5220000147819519, 1.3619999885559082, 0.0], [0.0, 0.0, 0.0], [1.5240000486373901, -0.0, -0.0], [0.6269999742507935, 1.062000036239624, -0.0], [-0.5220000147819519, -0.7760000228881836, -1.2130000591278076], [0.6069999933242798, 1.3819999694824219, -0.0], [0.7160000205039978, 1.1950000524520874, -0.0], [0.7129999995231628, -1.194000005722046, -0.0010000000474974513], [2.1070001125335693, 1.2000000476837158, -0.0020000000949949026], [2.1040000915527344, -1.2009999752044678, -0.003000000026077032], [2.7909998893737793, -0.0010000000474974513, -0.003000000026077032], [4.168000221252441, -0.0020000000949949026, -0.004999999888241291], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[-0.49399998784065247, 1.3730000257492065, -0.0], [0.0, 0.0, 0.0], [1.5269999504089355, -0.0, -0.0], [0.6269999742507935, 1.062000036239624, -0.0], [-0.5329999923706055, -0.7950000166893005, -1.2130000591278076], [0.5400000214576721, 1.4290000200271606, -0.0], [0.5329999923706055, -0.7760000228881836, 1.2029999494552612], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
]
_ATOM14_MASK = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
_ATOM37_MASK = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
_ATOM14_TO_37 = [
    [0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 11, 23, 32, 29, 30, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 16, 15, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 16, 17, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 11, 26, 25, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 11, 26, 27, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 14, 13, 20, 25, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 6, 7, 12, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 12, 13, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 11, 19, 35, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 18, 19, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 11, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 9, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 12, 13, 24, 21, 22, 33, 34, 28],
    [0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32, 31, 0, 0],
    [0, 1, 2, 4, 3, 6, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
_ATOM37_TO_14 = [
    [0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 9, 10, 0, 8, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 0, 7, 6, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 0, 5, 6, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 0, 0, 6, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 8, 0, 0, 0, 13, 0, 0, 0, 0, 11, 12, 0, 0],
    [0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 10, 0, 0, 0, 0],
    [0, 1, 2, 4, 3, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
