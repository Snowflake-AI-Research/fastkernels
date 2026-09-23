"""The HF example's concrete timm ResNet50 through existing residual blocks."""

from .resnet import ResNetModel
from ..runner import Config, Workload


def resnet50(device, dtype):
    config = Config(num_channels=3, embedding_size=64, hidden_sizes=[256, 512, 1024, 2048],
                    depths=[3, 4, 6, 3], downsample_in_first_stage=False,
                    downsample_in_bottleneck=False)
    return ResNetModel(config).to(device=device, dtype=dtype).eval()


def build_from_config(config, device, dtype):
    if config.backbone != "resnet50" or not config.features_only or tuple(config.out_indices) != (-1,):
        raise ValueError("This case preserves the HF example's ResNet50 final-stage backbone")
    if config.freeze_batch_norm_2d or config.output_stride not in (None, 32):
        raise ValueError("The declared backbone preserves ordinary BatchNorm and stride32")
    return resnet50(device, dtype)


def load_resnet50(model, state_dict, prefix):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        if name.startswith("embedder.embedder."):
            source = name.removeprefix("embedder.embedder.")
            source = source.replace("convolution.", "conv1.").replace("normalization.", "bn1.")
        elif name.startswith("encoder.stages."):
            parts = name.split(".")
            stage, block, branch = int(parts[2]), parts[4], parts[5:]
            source = f"layer{stage + 1}.{block}."
            if branch[0] == "shortcut":
                source += "downsample." + ("0." if branch[1] == "convolution" else "1.") + ".".join(branch[2:])
            else:
                index = int(branch[1]) + 1
                source += ("conv" if branch[2] == "convolution" else "bn") + str(index) + "." + ".".join(branch[3:])
        else:
            raise KeyError(name)
        mapped[name] = remaining.pop(prefix + source)
    if remaining:
        raise KeyError(f"Unmapped concrete timm ResNet50 state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def load_state_dict_into(model, state_dict, config):
    load_resnet50(model, state_dict, "_backbone.")


def make_workloads(model, inputs, config):
    def forward():
        hidden = model.encoder(model.embedder(inputs["pixel_values"]))
        return {"feature_maps.0": hidden}
    return {"forward": Workload(run=forward)}
