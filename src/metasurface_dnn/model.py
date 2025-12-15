from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .physics import angular_spectrum_propagate


@dataclass(frozen=True)
class DetectorRegion:
    x0: int
    x1: int
    y0: int
    y1: int


class PhaseMask(nn.Module):
    def __init__(self, height: int, width: int, init: str = "uniform"):
        super().__init__()
        self.height = height
        self.width = width

        # Unconstrained parameters -> mapped to [0, 2π] via sigmoid.
        self._phase_u = nn.Parameter(torch.zeros((height, width), dtype=torch.float32))
        if init == "uniform":
            nn.init.uniform_(self._phase_u, a=-2.0, b=2.0)
        elif init == "zeros":
            nn.init.zeros_(self._phase_u)
        else:
            raise ValueError(f"Unknown init: {init}")

    def phase_rad(self) -> torch.Tensor:
        return 2.0 * math.pi * torch.sigmoid(self._phase_u)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.dtype not in (torch.complex64, torch.complex128):
            field = field.to(torch.complex64)
        phase = self.phase_rad().to(field.device)
        modulation = torch.exp(1j * phase)
        return field * modulation


class IntensityDetector(nn.Module):
    def __init__(self, regions: Iterable[DetectorRegion]):
        super().__init__()
        self.regions = list(regions)
        if len(self.regions) < 2:
            raise ValueError("Need at least 2 regions for classification")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = (field.real**2 + field.imag**2)
        batch = intensity.shape[0]
        logits = torch.zeros((batch, len(self.regions)), device=intensity.device, dtype=intensity.dtype)
        for i, r in enumerate(self.regions):
            patch = intensity[:, r.x0 : r.x1, r.y0 : r.y1]
            logits[:, i] = patch.sum(dim=(-2, -1))
        return logits


class ThreeLayerMetasurfaceDNN(nn.Module):
    def __init__(
        self,
        *,
        size: int,
        wavelength_m: float,
        pixel_size_m: float,
        z_list_m: list[float],
        n0: float = 1.0,
        phase_init: str = "uniform",
        detector_regions: list[DetectorRegion],
    ):
        super().__init__()
        if len(z_list_m) != 4:
            raise ValueError("z_list_m must be [input->L1, L1->L2, L2->L3, L3->detector]")

        self.size = size
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.n0 = float(n0)
        self.z_list_m = [float(z) for z in z_list_m]

        self.l1 = PhaseMask(size, size, init=phase_init)
        self.l2 = PhaseMask(size, size, init=phase_init)
        self.l3 = PhaseMask(size, size, init=phase_init)
        self.detector = IntensityDetector(detector_regions)

    def forward(self, amplitude: torch.Tensor) -> torch.Tensor:
        # amplitude: (B,H,W) real -> complex field with zero phase
        if amplitude.ndim != 3:
            raise ValueError("Expected amplitude with shape (B,H,W)")
        if amplitude.shape[-1] != self.size or amplitude.shape[-2] != self.size:
            raise ValueError(f"Expected size {self.size}x{self.size}, got {amplitude.shape[-2:]}")

        field = amplitude.to(torch.float32).to(amplitude.device).to(torch.complex64)

        field = angular_spectrum_propagate(
            field,
            wavelength_m=self.wavelength_m,
            pixel_size_m=self.pixel_size_m,
            z_m=self.z_list_m[0],
            n0=self.n0,
        )
        field = self.l1(field)

        field = angular_spectrum_propagate(
            field,
            wavelength_m=self.wavelength_m,
            pixel_size_m=self.pixel_size_m,
            z_m=self.z_list_m[1],
            n0=self.n0,
        )
        field = self.l2(field)

        field = angular_spectrum_propagate(
            field,
            wavelength_m=self.wavelength_m,
            pixel_size_m=self.pixel_size_m,
            z_m=self.z_list_m[2],
            n0=self.n0,
        )
        field = self.l3(field)

        field = angular_spectrum_propagate(
            field,
            wavelength_m=self.wavelength_m,
            pixel_size_m=self.pixel_size_m,
            z_m=self.z_list_m[3],
            n0=self.n0,
        )
        logits = self.detector(field)
        return logits

    def export_phase_masks(self) -> dict[str, torch.Tensor]:
        return {
            "layer1_phase_rad": self.l1.phase_rad().detach().cpu(),
            "layer2_phase_rad": self.l2.phase_rad().detach().cpu(),
            "layer3_phase_rad": self.l3.phase_rad().detach().cpu(),
        }
