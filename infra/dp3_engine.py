"""DP3 inference engine for 3D diffusion policy models.

Loads a reference DP3 / Simple-DP3 checkpoint (a dill-pickled payload from
``train.py`` / ``TrainDP3Workspace``) into the kb-nano DP3Pipeline and
exposes a ``generate()`` API for action-chunk inference.

Mirrors the Pi0Engine pattern but specialized for DP3:
    - No HuggingFace download — checkpoints are local files / dirs produced
      by the reference's training loop.
    - The checkpoint contains both ``state_dicts['ema_model']`` (preferred at
      inference) and ``state_dicts['model']`` plus the OmegaConf ``cfg``
      that produced them; we read ``cfg`` to size the kb-nano DP3Config.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any

import torch

from ..tasks.baseline.L4.dp3 import (
    DP3Config,
    DP3Output,
    DP3Pipeline,
    DP3SamplingParams,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_dp3_checkpoint(path: str) -> dict[str, Any]:
    """Load a reference DP3 checkpoint payload.

    Reference saves via ``dill.dump(payload, f, ...)`` (see
    ``BasePolicy.save_checkpoint``); the payload is also readable with
    standard pickle.
    """
    if os.path.isdir(path):
        # Allow passing the run dir; default to checkpoints/latest.ckpt.
        ckpt_path = os.path.join(path, "checkpoints", "latest.ckpt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Pass the .ckpt path directly."
            )
        path = ckpt_path

    try:
        import dill  # noqa: F401 — reference uses dill, normal pickle works too
        with open(path, "rb") as f:
            return dill.load(f)
    except ImportError:
        with open(path, "rb") as f:
            return pickle.load(f)


def _config_from_payload(payload: dict[str, Any]) -> DP3Config:
    """Derive a kb-nano DP3Config from the reference's OmegaConf training cfg."""
    cfg = payload.get("cfg")
    if cfg is None:
        raise KeyError(
            "Checkpoint payload missing 'cfg'. Cannot infer DP3Config; "
            "construct DP3Pipeline manually."
        )
    # cfg may be OmegaConf DictConfig; index uniformly via attribute access.
    pol = cfg.policy
    shape_meta = cfg.shape_meta

    obs = shape_meta.obs
    pc_shape = list(obs.point_cloud.shape)
    state_dim = int(list(obs.agent_pos.shape)[0])
    action_dim = int(list(shape_meta.action.shape)[0])
    num_points = int(pc_shape[0])
    pc_in_channels = int(pc_shape[1]) if len(pc_shape) > 1 else 3

    use_pc_color = bool(getattr(pol, "use_pc_color", False))
    if use_pc_color and pc_in_channels < 6:
        raise ValueError(
            f"use_pc_color=True but point_cloud shape={pc_shape}"
        )

    pcfg = pol.pointcloud_encoder_cfg
    pn_use_layernorm = bool(getattr(pcfg, "use_layernorm", True))
    pn_final_norm = str(getattr(pcfg, "final_norm", "layernorm"))
    encoder_output_dim = int(getattr(pol, "encoder_output_dim", 64))

    sched = pol.noise_scheduler

    down_dims = tuple(int(d) for d in pol.down_dims)
    if down_dims == (128, 256, 384):
        variant = "simple_dp3"
    elif down_dims == (512, 1024, 2048):
        variant = "dp3"
    else:
        variant = "dp3"  # custom — treat as full DP3 layout

    return DP3Config(
        num_points=num_points,
        use_pc_color=use_pc_color,
        state_dim=state_dim,
        encoder_output_dim=encoder_output_dim,
        state_mlp_hidden_sizes=(64, 64),
        pointnet_use_layernorm=pn_use_layernorm,
        pointnet_final_norm=pn_final_norm,
        action_dim=action_dim,
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
        variant=variant,
        diffusion_step_embed_dim=int(pol.diffusion_step_embed_dim),
        down_dims=down_dims,
        kernel_size=int(pol.kernel_size),
        n_groups=int(pol.n_groups),
        num_train_timesteps=int(sched.num_train_timesteps),
        num_inference_steps=int(pol.num_inference_steps),
        beta_start=float(sched.beta_start),
        beta_end=float(sched.beta_end),
        beta_schedule=str(sched.beta_schedule),
        clip_sample=bool(sched.clip_sample),
        set_alpha_to_one=bool(sched.set_alpha_to_one),
        steps_offset=int(sched.steps_offset),
        prediction_type=str(sched.prediction_type),
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DP3Engine:
    """Engine for running DP3 / Simple-DP3 inference.

    Args:
        checkpoint_path: path to a ``latest.ckpt`` produced by the reference's
            ``train.py``. May also be a run directory containing
            ``checkpoints/latest.ckpt``.
        config: optional pre-built DP3Config. If omitted, the config is
            inferred from the checkpoint's stored ``cfg``.
        use_ema: if the checkpoint has both ``model`` and ``ema_model``
            state_dicts, choose ema (recommended for inference).
        seed: default RNG seed for deterministic noise generation.
        dtype: model dtype on GPU. Default fp32 for max parity with reference.
        device: target device.
        enforce_eager: skip ``torch.compile`` if True.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: DP3Config | None = None,
        use_ema: bool = True,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        enforce_eager: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.config_override = config
        self.use_ema = use_ema
        self.seed = seed
        self.dtype = dtype
        self.device_str = device
        self.device = torch.device(device)
        self.enforce_eager = enforce_eager
        self._pipeline: DP3Pipeline | None = None
        self._config: DP3Config | None = None

    @property
    def config(self) -> DP3Config:
        if self._config is None:
            self._get_pipeline()
        return self._config  # type: ignore[return-value]

    def _get_pipeline(self) -> DP3Pipeline:
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading DP3 checkpoint: %s", self.checkpoint_path)
        payload = _load_dp3_checkpoint(self.checkpoint_path)

        if self.config_override is not None:
            cfg = self.config_override
        else:
            cfg = _config_from_payload(payload)
        self._config = cfg

        pipeline = DP3Pipeline(cfg)

        # Pre-allocate normalizer slots so load_weights can copy in scale/offset.
        pipeline.register_normalizer_keys(
            ["action", "agent_pos", "point_cloud"],
            {
                "action": cfg.action_dim,
                "agent_pos": cfg.state_dim,
                "point_cloud": (6 if cfg.use_pc_color else 3),
            },
        )

        state_dicts = payload.get("state_dicts", {})
        sd = state_dicts.get("ema_model" if self.use_ema else "model")
        if sd is None:
            sd = state_dicts.get("model") or state_dicts.get("ema_model")
        if sd is None:
            raise KeyError(
                f"Checkpoint {self.checkpoint_path} missing 'model'/'ema_model' "
                "state_dict under 'state_dicts'."
            )

        # The reference saves state_dict items as a dict; convert to (name, tensor)
        # pairs and feed through DP3Pipeline.load_weights() (which silently drops
        # noise_scheduler_pc and mask_generator buffers).
        loaded = pipeline.load_weights(list(sd.items()))
        logger.info(
            "DP3 weights loaded: %d / %d source keys consumed",
            len(loaded), len(sd),
        )

        pipeline.to(device=self.device, dtype=self.dtype)
        pipeline.eval()

        if not self.enforce_eager:
            try:
                pipeline.model = torch.compile(
                    pipeline.model, mode="reduce-overhead",
                )
                logger.info("torch.compile applied to ConditionalUnet1D")
            except Exception as e:
                logger.warning("torch.compile failed, using eager: %s", e)

        self._pipeline = pipeline
        return pipeline

    def generate(
        self,
        point_cloud: torch.Tensor,
        agent_pos: torch.Tensor,
        params: DP3SamplingParams | None = None,
        noise: torch.Tensor | None = None,
    ) -> DP3Output:
        """Generate an action chunk from a single observation step.

        Args:
            point_cloud: (B, To, N, 3) or (B, To, N, 6) raw point cloud.
            agent_pos: (B, To, state_dim) raw robot state.
            params: sampling params (override num_inference_steps).
            noise: optional (B, horizon, action_dim) shared starting noise.
        Returns:
            DP3Output(actions, action_pred).
        """
        pipeline = self._get_pipeline()
        cfg = self._config
        assert cfg is not None
        params = params or DP3SamplingParams()

        if noise is None:
            seed = params.seed if params.seed is not None else self.seed
            g = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(
                point_cloud.shape[0], cfg.horizon, cfg.action_dim,
                generator=g, dtype=self.dtype, device=self.device,
            )
        else:
            noise = noise.to(device=self.device, dtype=self.dtype)

        point_cloud = point_cloud.to(device=self.device, dtype=self.dtype)
        agent_pos = agent_pos.to(device=self.device, dtype=self.dtype)

        return pipeline(
            point_cloud=point_cloud,
            agent_pos=agent_pos,
            noise=noise,
            params=params,
        )

    def warmup(self, num_steps: int = 2) -> None:
        """Run a small warmup inference to prime CUDA / compile."""
        pipeline = self._get_pipeline()
        cfg = self._config
        assert cfg is not None

        in_ch = 6 if cfg.use_pc_color else 3
        pc = torch.zeros(
            1, cfg.n_obs_steps, cfg.num_points, in_ch,
            device=self.device, dtype=self.dtype,
        )
        ap = torch.zeros(
            1, cfg.n_obs_steps, cfg.state_dim,
            device=self.device, dtype=self.dtype,
        )
        params = DP3SamplingParams(num_inference_steps=num_steps)
        self.generate(pc, ap, params=params)

    def _cleanup(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        torch.cuda.empty_cache()
