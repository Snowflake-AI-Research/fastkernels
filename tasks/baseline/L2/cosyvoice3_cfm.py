"""Conditional Flow Matching (CFM) for CosyVoice3 audio generation.

Adopted from vllm-omni CosyVoice3 code2wav_core/cfm.py.
Implements the Euler ODE solver with classifier-free guidance.
"""

from __future__ import annotations

from abc import ABC

import torch
import torch.nn as nn


class BASECFM(nn.Module, ABC):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=128):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params["solver"]
        self.sigma_min = cfm_params.get("sigma_min", 1e-4)
        self.estimator = None


class CausalConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64,
                 estimator: nn.Module | None = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params["t_scheduler"]
        self.training_cfg_rate = cfm_params["training_cfg_rate"]
        self.inference_cfg_rate = cfm_params["inference_cfg_rate"]
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None,
                cfm_seed: int | None = None):
        if cfm_seed is not None:
            gen = torch.Generator(device=mu.device)
            gen.manual_seed(cfm_seed)
            z = torch.randn(
                (mu.size(0), mu.size(1), mu.size(2)),
                device=mu.device, dtype=mu.dtype, generator=gen,
            ) * temperature
        else:
            z = torch.randn(
                (mu.size(0), mu.size(1), mu.size(2)),
                device=mu.device, dtype=mu.dtype,
            ) * temperature

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt
                       - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x.float()
