"""RT-DETRv2 detection and proposal outputs using the existing detector stages."""

from torch import nn

from fastkernels.tasks.baseline.L1.linear import BMM
from fastkernels.tasks.baseline.L2.rtdetrv2_multihead_attention import RTDetrV2MultiheadAttention
from ..patches.rt_detr_v2 import RTDetrV2ProposalOutputs
from ..runner import Workload, config_values


class RTDetrV2ForObjectDetection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = RTDetrV2ProposalOutputs(config)

    def forward(self, pixel_values, pixel_mask=None):
        return self.model(pixel_values, pixel_mask=pixel_mask)


class ScoreScaledAttention(RTDetrV2MultiheadAttention):
    """Compose existing operations with HF's scaling after the score matmul."""

    def __init__(self, width, heads):
        super().__init__(width, heads)
        self.matmul = BMM()

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                output_attentions=False):
        batch, length, width = hidden_states.shape
        positioned = hidden_states if position_embeddings is None else hidden_states + position_embeddings
        def heads(value):
            return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        query, key, value = heads(self.q_proj(positioned)), heads(self.k_proj(positioned)), heads(self.v_proj(hidden_states))
        scores = self.matmul(query, key.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = self._softmax(scores)
        output = self.matmul(probabilities, value).transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output), probabilities if output_attentions else None


def build_from_config(config, device, dtype):
    config = config_values(config.to_dict())
    # HF exposes this as a property; serialized configurations keep the labels.
    config.num_labels = len(config.id2label)
    if (config.backbone_config.model_type != "rt_detr_resnet"
            or config.learn_initial_query or not config.with_box_refine
            or config.anchor_image_size is not None
            or config.num_feature_levels != len(config.decoder_in_channels)
            or config.num_feature_levels != len(config.encoder_in_channels)):
        raise ValueError("RT-DETRv2 requires the documented ResNet detector with proposal queries and box refinement")
    model = RTDetrV2ForObjectDetection(config)
    for layer in tuple(model.modules()):
        attention = getattr(layer, "self_attn", None)
        if isinstance(attention, RTDetrV2MultiheadAttention):
            layer.self_attn = ScoreScaledAttention(attention.embed_dim, attention.num_heads)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        source = source.replace("encoder.encoder.", "encoder.aifi.")
        source = source.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        source = source.replace(".fc1.", ".mlp.fc1.")
        source = source.replace(".fc2.", ".mlp.fc2.")
        weights[name] = state_dict[source]
    # Keep the reference's parameter and buffer dtypes, including frozen norms.
    model.load_state_dict(weights, strict=True, assign=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
