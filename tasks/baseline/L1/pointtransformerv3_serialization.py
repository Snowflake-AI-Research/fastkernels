"""PointTransformerV3 serialization helpers."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


class _KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")
        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = key | ((x & mask) << (2 * i + 2)) | ((y & mask) << (2 * i + 1)) | ((z & mask) << (2 * i))
        return key

    def key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i))) >> (2 * i))
        return x, y, z


_KEY_LUT = _KeyLUT()


def z_order_encode_(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, b: Optional[Union[torch.Tensor, int]] = None, depth: int = 16):
    ex, ey, ez = _KEY_LUT.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()
    mask = 255 if depth > 8 else (1 << depth) - 1
    key = ex[x & mask] | ey[y & mask] | ez[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = ex[(x >> 8) & mask] | ey[(y >> 8) & mask] | ez[(z >> 8) & mask]
        key = key16 << 24 | key
    if b is not None:
        key = b.long() << 48 | key
    return key


def z_order_decode_(key: torch.Tensor, depth: int = 16):
    dx, dy, dz = _KEY_LUT.decode_lut(key.device)
    x = torch.zeros_like(key)
    y = torch.zeros_like(key)
    z = torch.zeros_like(key)
    b = key >> 48
    key = key & ((1 << 48) - 1)
    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 9) & 511
        x = x | (dx[k] << (i * 3))
        y = y | (dy[k] << (i * 3))
        z = z | (dz[k] << (i * 3))
    return x, y, z, b


def _right_shift(binary: torch.Tensor, k: int = 1, axis: int = -1) -> torch.Tensor:
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return torch.nn.functional.pad(binary[tuple(slicing)], (k, 0), mode="constant", value=0)


def _binary2gray(binary: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.logical_xor(binary, _right_shift(binary, axis=axis))


def _gray2binary(gray: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shift = 2 ** (torch.tensor([gray.shape[axis]], device=gray.device).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, int(shift.item())))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def hilbert_encode_(locs: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8, device=locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)
    if orig_shape[-1] != num_dims:
        raise ValueError("locs last dimension must match num_dims")
    if num_dims * num_bits > 63:
        raise ValueError("num_dims * num_bits must be <= 63")
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1).bitwise_and(bitpack_mask_rev).ne(0).byte().flatten(-2, -1)[..., -num_bits:]
    )
    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = _gray2binary(gray)
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)
    hh_uint8 = ((padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).squeeze().type(torch.uint8))
    return hh_uint8.view(torch.int64).squeeze()


def hilbert_decode_(hilberts: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    if num_dims * num_bits > 64:
        raise ValueError("num_dims * num_bits must be <= 64")
    hilberts = torch.atleast_1d(hilberts)
    orig_shape = hilberts.shape
    bitpack_mask = 2 ** torch.arange(0, 8, device=hilberts.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)
    hh_uint8 = hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    hh_bits = (
        hh_uint8.unsqueeze(-1).bitwise_and(bitpack_mask_rev).ne(0).byte().flatten(-2, -1)[:, -num_dims * num_bits :]
    )
    gray = _binary2gray(hh_bits)
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)
    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)
    extra_dims = 8 - (num_bits % 8)
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)
    locs_uint8 = ((padded.flip(-1).reshape((-1, num_dims, 8, 8)) * bitpack_mask).sum(3).squeeze().type(torch.uint8))
    flat_locs = locs_uint8.reshape((-1, num_dims * 8)).contiguous()
    decoded = flat_locs.view(torch.int64).reshape((-1, num_dims))
    return decoded.reshape((*orig_shape, num_dims))


@torch.inference_mode()
def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    return z_order_encode_(x, y, z, b=None, depth=depth)


@torch.inference_mode()
def z_order_decode(code: torch.Tensor, depth: int):
    x, y, z, _ = z_order_decode_(code, depth=depth)
    return torch.stack([x, y, z], dim=-1)


@torch.inference_mode()
def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


@torch.inference_mode()
def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)


@torch.inference_mode()
def encode(grid_coord: torch.Tensor, batch: torch.Tensor | None = None, depth: int = 16, order: str = "z"):
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise ValueError(f"Unsupported serialization order: {order}")
    if batch is not None:
        code = batch.long() << depth * 3 | code
    return code


class PointTransformerV3Serialization(nn.Module):
    def forward(
        self,
        grid_coord: torch.Tensor,
        batch: torch.Tensor | None = None,
        depth: int = 16,
        order: str = "z",
    ) -> torch.Tensor:
        return encode(grid_coord, batch=batch, depth=depth, order=order)
