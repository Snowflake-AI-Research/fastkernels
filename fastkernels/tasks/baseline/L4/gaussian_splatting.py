"""Minimal 3D Gaussian Splatting renderer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.gaussian_splat_render import GaussianSplatRender


class GaussianSplatting(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        width: int,
        height: int,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        tile_size: int = 16,
    ):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("quats", quats)
        self.register_buffer("scales", scales)
        self.register_buffer("opacities", opacities)
        self.register_buffer("colors", colors)
        if viewmats is not None:
            self.register_buffer("viewmats", viewmats)
            self.register_buffer(
                "batched_colors",
                colors.unsqueeze(0).expand(viewmats.shape[0], -1, -1).contiguous(),
            )
            self.register_buffer(
                "batched_opacities",
                opacities.unsqueeze(0).expand(viewmats.shape[0], -1).contiguous(),
            )
        else:
            self.viewmats = None
            self.batched_colors = None
            self.batched_opacities = None
        if Ks is not None:
            self.register_buffer("Ks", Ks)
        else:
            self.Ks = None
        self.width = width
        self.height = height
        self.renderer = GaussianSplatRender(tile_size=tile_size)
        self._graph = None
        self._graph_rgb = None
        self._graph_alpha = None
        self._graph_ready = False

    def _render_eager(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        backgrounds: torch.Tensor | None,
        return_meta: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.renderer(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.colors,
            viewmats,
            Ks,
            self.width,
            self.height,
            backgrounds=backgrounds,
            batched_opacities=self.batched_opacities if viewmats is self.viewmats and self.batched_opacities is not None else None,
            batched_colors=self.batched_colors if viewmats is self.viewmats and self.batched_colors is not None else None,
            return_meta=return_meta,
        )

    def prepare_graph(self) -> bool:
        if self._graph_ready:
            return True
        if self.means.device.type != "cuda":
            return False
        if self.viewmats is None or self.Ks is None:
            return False

        stream = torch.cuda.Stream(device=self.means.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.means.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._render_eager(self.viewmats, self.Ks, None, False)
        torch.cuda.current_stream(device=self.means.device).wait_stream(stream)
        torch.cuda.synchronize(device=self.means.device)

        self._graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self._graph):
                self._graph_rgb, self._graph_alpha, _ = self._render_eager(
                    self.viewmats,
                    self.Ks,
                    None,
                    False,
                )
        except RuntimeError:
            self._graph = None
            self._graph_rgb = None
            self._graph_alpha = None
            self._graph_ready = False
            return False
        self._graph_ready = True
        return True

    def forward(
        self,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        backgrounds: torch.Tensor | None = None,
        return_meta: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if viewmats is None:
            if self.viewmats is None:
                raise ValueError("viewmats must be provided")
            viewmats = self.viewmats
        if Ks is None:
            if self.Ks is None:
                raise ValueError("Ks must be provided")
            Ks = self.Ks
        use_graph = (
            self._graph_ready
            and viewmats is self.viewmats
            and Ks is self.Ks
            and backgrounds is None
            and not return_meta
        )
        if use_graph:
            self._graph.replay()
            return self._graph_rgb, self._graph_alpha, {}
        return self._render_eager(viewmats, Ks, backgrounds, return_meta)
