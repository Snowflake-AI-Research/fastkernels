"""DP3 (3D Diffusion Policy) — L4 pipeline.

3-D visuomotor policy from "3D Diffusion Policy: Generalizable Visuomotor
Policy Learning via Simple 3D Representations" (Ze et al., RSS 2024).

Reference implementation: https://github.com/YanjieZe/3D-Diffusion-Policy

Architecture (inference):
    point_cloud (B, T_obs, N, 3)         agent_pos (B, T_obs, state_dim)
        │                                  │
        └─ PointNetEncoderXYZ ─ 64         └─ state_mlp ─ 64
                    │                                │
                    └────── concat ──────────────────┘
                                  │
                            (B, T_obs * 128) = global_cond
                                  │
                       ┌──────────▼──────────┐
                       │ ConditionalUnet1D    │
                       │ (1-D U-Net + FiLM)   │
                       └──────────┬──────────┘
                          DDIM scheduler (10 steps)
                                  │
                          predicted x_0 (action chunk)
                                  │
                          unnormalize → slice [To-1 : To-1+n_action_steps]

Two flavours are wired up via ``DP3Config.variant``:
    "dp3"        : down_dims=[512,1024,2048], dsed=128 (full)
    "simple_dp3" : down_dims=[128, 256, 384], dsed=128 (faster)

Both share the same encoder, the same DDIM scheduler with
``prediction_type="sample"``, and the same horizon=16 / n_obs_steps=2 /
n_action_steps=8 layout (matches released ``dp3.yaml`` and ``simple_dp3.yaml``).

L4 wiring/configuration only — computation lives in L1-L3 tasks. Uses
``diffusers.DDIMScheduler`` (consistent with how SDXL / FLUX / HunyuanVideo
L4 pipelines pull schedulers from diffusers).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ..L2.dp3_pointnet_encoder import PointNetEncoderXYZ, PointNetEncoderXYZRGB
from ..L2.dp3_state_encoder import build_state_mlp
from ..L3.dp3_conditional_unet1d import ConditionalUnet1D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DP3Config:
    """DP3 / Simple-DP3 configuration.

    Defaults match Simple-DP3 on MetaWorld-reach: 4-D action, 9-D state,
    512 points (XYZ only), horizon=16, To=2, n_action_steps=8.
    """

    # Encoder
    num_points: int = 512
    use_pc_color: bool = False
    state_dim: int = 9
    encoder_output_dim: int = 64
    state_mlp_hidden_sizes: tuple[int, ...] = (64, 64)
    pointnet_use_layernorm: bool = True
    pointnet_final_norm: str = "layernorm"

    # Action / horizon
    action_dim: int = 4
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8

    # U-Net
    variant: Literal["dp3", "simple_dp3"] = "simple_dp3"
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (128, 256, 384)
    kernel_size: int = 5
    n_groups: int = 8

    # Scheduler
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "sample"

    @classmethod
    def simple_dp3_metaworld(cls, action_dim: int, state_dim: int,
                             num_points: int = 512) -> "DP3Config":
        return cls(
            variant="simple_dp3",
            down_dims=(128, 256, 384),
            diffusion_step_embed_dim=128,
            action_dim=action_dim,
            state_dim=state_dim,
            num_points=num_points,
        )

    @classmethod
    def dp3_full(cls, action_dim: int, state_dim: int,
                 num_points: int = 512) -> "DP3Config":
        return cls(
            variant="dp3",
            down_dims=(512, 1024, 2048),
            diffusion_step_embed_dim=128,
            action_dim=action_dim,
            state_dim=state_dim,
            num_points=num_points,
        )


# ---------------------------------------------------------------------------
# Sampling output
# ---------------------------------------------------------------------------

@dataclass
class DP3SamplingParams:
    num_inference_steps: int | None = None
    seed: int | None = None


@dataclass
class DP3Output:
    actions: torch.Tensor | None = None
    action_pred: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Linear normalizer (matches reference ``LinearNormalizer`` arithmetic)
# ---------------------------------------------------------------------------

class LinearNormalizer(nn.Module):
    """Per-key linear normalizer: ``y = x * scale + offset``.

    Stores parameters under ``params_dict.<key>.scale`` /
    ``params_dict.<key>.offset`` to match the reference checkpoint key
    layout. Stats (``input_stats.min/max/mean/std``) are also stored as
    parameters in the reference; we register them as buffers so
    ``state_dict`` round-trips include them but they don't appear in
    ``parameters()`` / get optimized.
    """

    def __init__(self):
        super().__init__()
        self.params_dict = nn.ParameterDict()
        # Buffers for input_stats (min/max/mean/std) — registered lazily on
        # first checkpoint load via ``register_stats_buffers``.
        self._stat_keys: dict[str, list[str]] = {}

    def register_field(self, key: str, scale: torch.Tensor,
                       offset: torch.Tensor) -> None:
        params = nn.ParameterDict({
            "scale": nn.Parameter(scale.detach().clone(), requires_grad=False),
            "offset": nn.Parameter(offset.detach().clone(), requires_grad=False),
        })
        self.params_dict[key] = params

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.params_dict:
            return x
        p = self.params_dict[key]
        return x * p["scale"] + p["offset"]

    def unnormalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.params_dict:
            return x
        p = self.params_dict[key]
        return (x - p["offset"]) / p["scale"]


# ---------------------------------------------------------------------------
# Observation encoder (PointNet + state_mlp)
# ---------------------------------------------------------------------------

class DP3Encoder(nn.Module):
    """Mirrors ``diffusion_policy_3d.model.vision.pointnet_extractor.DP3Encoder``.

    Submodule names (``extractor`` and ``state_mlp``) match the reference so
    a checkpoint state_dict loads with no remapping for this submodule.
    """

    def __init__(self, config: DP3Config):
        super().__init__()
        self.use_pc_color = config.use_pc_color
        if config.use_pc_color:
            self.extractor = PointNetEncoderXYZRGB(
                in_channels=6,
                out_channels=config.encoder_output_dim,
                use_layernorm=config.pointnet_use_layernorm,
                final_norm=config.pointnet_final_norm,
            )
        else:
            self.extractor = PointNetEncoderXYZ(
                in_channels=3,
                out_channels=config.encoder_output_dim,
                use_layernorm=config.pointnet_use_layernorm,
                final_norm=config.pointnet_final_norm,
            )
        self.state_mlp = build_state_mlp(
            config.state_dim, config.state_mlp_hidden_sizes,
        )
        self.out_dim = config.encoder_output_dim + config.state_mlp_hidden_sizes[-1]

    def forward(self, point_cloud: torch.Tensor,
                agent_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_cloud: (B*To, N, 3) or (B*To, N, 6) if use_pc_color.
            agent_pos:   (B*To, state_dim).
        Returns:
            (B*To, out_dim) where out_dim = encoder_output_dim + state_mlp[-1].
        """
        pn = self.extractor(point_cloud)
        st = self.state_mlp(agent_pos)
        return torch.cat([pn, st], dim=-1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DP3Pipeline(nn.Module):
    """Full DP3 inference pipeline.

    Composes ``obs_encoder`` (DP3Encoder), ``model`` (ConditionalUnet1D), and
    a per-key ``normalizer`` (LinearNormalizer) plus a
    ``diffusers.DDIMScheduler`` for the denoising loop.

    ``predict_action()`` mirrors ``diffusion_policy_3d.policy.dp3.DP3.predict_action``:
    normalize obs → encode → 10-step DDIM sample → unnormalize → slice
    ``[To-1 : To-1 + n_action_steps]``.
    """

    def __init__(self, config: DP3Config):
        super().__init__()
        self.config = config

        global_cond_dim = (
            config.encoder_output_dim
            + config.state_mlp_hidden_sizes[-1]
        ) * config.n_obs_steps

        self.obs_encoder = DP3Encoder(config)
        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
        )
        self.normalizer = LinearNormalizer()
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            set_alpha_to_one=config.set_alpha_to_one,
            steps_offset=config.steps_offset,
            prediction_type=config.prediction_type,
        )

    @torch.inference_mode()
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        noise: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Run the DDIM denoising loop.

        Returns:
            (batch_size, horizon, action_dim) predicted clean trajectory.
        """
        T = self.config.horizon
        Da = self.config.action_dim

        if noise is None:
            trajectory = torch.randn(
                (batch_size, T, Da), device=device, dtype=dtype,
            )
        else:
            trajectory = noise.to(device=device, dtype=dtype)

        steps = num_inference_steps or self.config.num_inference_steps
        # Leave timesteps on CPU. Iterating a GPU tensor in Python triggers
        # a CPU↔GPU sync per step inside diffusers' ``prev_timestep >= 0``
        # comparison; CPU timesteps avoids that. ``alphas_cumprod`` is moved
        # to the device once per pipeline (idempotent across calls).
        self.noise_scheduler.set_timesteps(steps)
        if (
            hasattr(self.noise_scheduler, "alphas_cumprod")
            and self.noise_scheduler.alphas_cumprod.device != device
        ):
            self.noise_scheduler.alphas_cumprod = (
                self.noise_scheduler.alphas_cumprod.to(device=device)
            )

        for t in self.noise_scheduler.timesteps:
            model_output = self.model(
                sample=trajectory,
                timestep=t,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory,
            ).prev_sample
        return trajectory

    @torch.inference_mode()
    def forward(
        self,
        point_cloud: torch.Tensor,
        agent_pos: torch.Tensor,
        noise: torch.Tensor | None = None,
        params: DP3SamplingParams | None = None,
    ) -> DP3Output:
        """Predict action chunk from observation.

        Args:
            point_cloud: (B, To, N, 3) or (B, To, N, 6) raw (un-normalized)
                point cloud, where To = ``config.n_obs_steps``.
            agent_pos: (B, To, state_dim) raw (un-normalized) robot state.
            noise: optional (B, horizon, action_dim) initial noise (for
                determinism). If None, drawn from default RNG.
            params: sampling parameters (override num_inference_steps).
        Returns:
            DP3Output with:
                actions: (B, n_action_steps, action_dim) — robot-space.
                action_pred: (B, horizon, action_dim) — full predicted chunk.
        """
        params = params or DP3SamplingParams()
        cfg = self.config
        device = point_cloud.device
        dtype = point_cloud.dtype
        B, To = point_cloud.shape[:2]
        assert To == cfg.n_obs_steps, (
            f"point_cloud has To={To}, expected {cfg.n_obs_steps}"
        )

        # 1. Normalize observations (per-key linear scale+offset).
        n_pc = self.normalizer.normalize(point_cloud, "point_cloud")
        if not cfg.use_pc_color:
            n_pc = n_pc[..., :3]
        n_state = self.normalizer.normalize(agent_pos, "agent_pos")

        # 2. Flatten (B, To, ...) -> (B*To, ...) and encode.
        pc_flat = n_pc.reshape(B * To, *n_pc.shape[2:])
        st_flat = n_state.reshape(B * To, *n_state.shape[2:])
        obs_feat = self.obs_encoder(pc_flat, st_flat)        # (B*To, Df)
        global_cond = obs_feat.reshape(B, -1)                # (B, To*Df)

        # 3. DDIM denoising loop.
        n_traj = self.conditional_sample(
            B, global_cond, device, dtype,
            noise=noise,
            num_inference_steps=params.num_inference_steps,
        )

        # 4. Unnormalize and slice executed action window.
        action_pred = self.normalizer.unnormalize(
            n_traj[..., : cfg.action_dim], "action",
        )
        start = cfg.n_obs_steps - 1
        end = start + cfg.n_action_steps
        actions = action_pred[:, start:end]

        return DP3Output(actions=actions, action_pred=action_pred)

    # ---- Weight loading ----

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a reference DP3 checkpoint state_dict.

        The reference's ``ema_model`` (or ``model``) state_dict from
        ``train.py`` saves keys directly compatible with our module
        hierarchy because we deliberately mirror submodule names
        (``obs_encoder.extractor.{mlp,final_projection}.*``,
        ``obs_encoder.state_mlp.*``,
        ``model.{down_modules,mid_modules,up_modules,final_conv,
        diffusion_step_encoder}.*``,
        ``normalizer.params_dict.{action,agent_pos,point_cloud}.{scale,offset}``).

        ``noise_scheduler_pc.*`` keys (from the reference's deepcopy of the
        scheduler) are silently dropped — they are re-derived at construction
        time from ``DP3Config``.

        Returns:
            Set of source key names that were consumed.
        """
        own = dict(self.named_parameters())
        own.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        skipped_unrecognized: list[str] = []

        for name, tensor in weights:
            # Drop the per-step noise scheduler clone the reference keeps.
            if name.startswith("noise_scheduler_pc."):
                loaded.add(name)
                continue
            # Drop the mask_generator buffers (training-only impainting mask).
            if name.startswith("mask_generator."):
                loaded.add(name)
                continue
            # ``input_stats`` are saved as Parameters in the reference; we
            # don't store them at all (they're not needed for inference).
            if "input_stats" in name:
                loaded.add(name)
                continue

            if name in own:
                target = own[name]
                if target.shape != tensor.shape:
                    logger.warning(
                        "Shape mismatch for %s: ckpt %s vs model %s; skipping",
                        name, tuple(tensor.shape), tuple(target.shape),
                    )
                    continue
                # ParameterDict entries store scale/offset as Parameters, so
                # ``own[name]`` is a Parameter — copy_ works for both Parameters
                # and buffers.
                target.data.copy_(tensor)
                loaded.add(name)
            else:
                skipped_unrecognized.append(name)

        if skipped_unrecognized:
            logger.debug(
                "DP3.load_weights: %d unrecognized keys (first 10): %s",
                len(skipped_unrecognized), skipped_unrecognized[:10],
            )

        # Lazily register normalizer fields if we found them via direct copy
        # but the corresponding ParameterDict slot didn't exist yet.
        # (In practice, register_normalizer_keys is called before load_weights
        # whenever the caller knows the keys; this branch is the safety net.)
        return loaded

    def register_normalizer_keys(self, keys: Iterable[str], dim_per_key: dict[str, int]) -> None:
        """Pre-allocate ``params_dict[key].scale/offset`` Parameters so that
        ``load_weights`` can copy into them without ``own.update`` missing.

        Call this *before* ``load_weights`` once you know the action / state /
        point_cloud feature dims (typically pulled from the reference
        checkpoint config).
        """
        for key in keys:
            d = dim_per_key[key]
            scale = torch.ones(d, dtype=torch.float32)
            offset = torch.zeros(d, dtype=torch.float32)
            self.normalizer.register_field(key, scale, offset)
