"""Full-row FFT adaptation of CausalHiFTGenerator's internal STFT helper."""

import torch
from torch import nn


class SingleFrameSTFT(nn.Module):
    """Keep the existing torch.stft FFT mechanism, with one rectangular frame.

    Parent: L2 cosyvoice3_hifigan.CausalHiFTGenerator._stft. Changes are a
    rectangular window, no centered padding, and both frequency halves.
    The real-input FP32 conversion and FFT backend remain the same. This
    exposes an internal library capability, not an existing standalone task.
    """

    def __init__(self, length):
        super().__init__()
        self.length = length
        self.register_buffer("window", torch.ones(length, dtype=torch.float32), persistent=False)

    def forward(self, rows):
        if rows.ndim != 2 or rows.shape[-1] != self.length:
            raise ValueError("Single-frame STFT expects complete rows of its configured length")
        spectrum = torch.stft(
            rows.float(), n_fft=self.length, hop_length=self.length,
            win_length=self.length, window=self.window,
            center=False, normalized=False, onesided=False, return_complex=True,
        )
        return spectrum.squeeze(-1)


class RealFourier2D(nn.Module):
    """Compose real-input row transforms into the real part of a 2D FFT.

    If the first transform is A+iB, the real result after transforming the
    second axis is real(FFT(A)) - imag(FFT(B)). Both second-axis transforms
    share one batched call. Storage remains linear in the activation size;
    the additional real transform is an explicit constant-factor cost.
    """

    def __init__(self, sequence_length, hidden_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_transform = SingleFrameSTFT(hidden_size)
        self.sequence_transform = SingleFrameSTFT(sequence_length)

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        if (sequence, hidden) != (self.sequence_length, self.hidden_size):
            raise ValueError("The Fourier composition requires the configured sequence and hidden dimensions")
        if hidden_states.dtype != torch.float32:
            raise ValueError("This Fourier composition evaluates the native FP32 FNet workload")
        first = self.hidden_transform(hidden_states.reshape(batch * sequence, hidden))
        first = first.view(batch, sequence, hidden)
        rows = torch.cat((first.real.transpose(1, 2), first.imag.transpose(1, 2)), dim=0)
        second = self.sequence_transform(rows.reshape(2 * batch * hidden, sequence))
        second = second.view(2, batch, hidden, sequence)
        return (second[0].real - second[1].imag).transpose(1, 2)
