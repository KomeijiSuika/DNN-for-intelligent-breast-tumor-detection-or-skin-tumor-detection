from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PropagationParams:
    wavelength_m: float
    pixel_size_m: float
    z_m: float
    n0: float = 1.0


def _make_frequency_grid(n: int, pixel_size_m: float, device: torch.device, dtype: torch.dtype):
    # torch.fft.fftfreq returns cycles per meter
    fx = torch.fft.fftfreq(n, d=pixel_size_m, device=device)
    fy = torch.fft.fftfreq(n, d=pixel_size_m, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing="ij")
    return FX.to(dtype=dtype), FY.to(dtype=dtype)


def angular_spectrum_propagate(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    pixel_size_m: float,
    z_m: float,
    n0: float = 1.0,
    bandlimit: bool = True,
) -> torch.Tensor:
    """Angular spectrum propagation (scalar) using FFT.

    Args:
        field: Complex tensor with shape (..., H, W). H==W recommended.
        wavelength_m: Free-space wavelength.
        pixel_size_m: Sampling pitch (meters/pixel).
        z_m: Propagation distance.
        n0: Background refractive index.
        bandlimit: Mask evanescent components when True.

    Returns:
        Propagated complex field of same shape.
    """
    if field.dtype not in (torch.complex64, torch.complex128):
        field = field.to(torch.complex64)

    *batch, h, w = field.shape
    if h != w:
        raise ValueError(f"angular_spectrum_propagate expects square field, got {h}x{w}")

    device = field.device
    real_dtype = torch.float32 if field.dtype == torch.complex64 else torch.float64

    FX, FY = _make_frequency_grid(h, pixel_size_m, device=device, dtype=real_dtype)

    # Spatial frequency to normalized direction cosines
    # k = 2π n0 / λ
    k0 = 2.0 * math.pi / wavelength_m
    k = k0 * n0

    # sqrt term: 1 - (λ fx / n0)^2 - (λ fy / n0)^2
    # using cycles/m so multiply by λ/n0
    alpha = (wavelength_m / n0) * FX
    beta = (wavelength_m / n0) * FY
    root_arg = 1.0 - (alpha**2 + beta**2)

    if bandlimit:
        propagating = root_arg >= 0
        root = torch.zeros_like(root_arg)
        root[propagating] = torch.sqrt(root_arg[propagating])
        # Evanescent components are set to 0 (masked)
        H = torch.exp(1j * k * z_m * root) * propagating
    else:
        root = torch.sqrt(root_arg.to(torch.complex64 if real_dtype == torch.float32 else torch.complex128))
        H = torch.exp(1j * k * z_m * root)

    U1 = torch.fft.fft2(field, dim=(-2, -1))
    U2 = U1 * H
    u2 = torch.fft.ifft2(U2, dim=(-2, -1))
    return u2
