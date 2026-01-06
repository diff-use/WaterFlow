
import math

import torch

from torch import nn


class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))
    


class NormalizedBasis(torch.nn.Module):
    """Normalized version of a given radial basis.

    Args:
        basis (constructor): callable to build the underlying basis
        basis_kwargs (dict): parameters for the underlying basis
        n (int, optional): the number of samples to use for the estimated statistics
        r_min (float): the lower bound of the uniform square bump distribution for inputs
        r_max (float): the upper bound of the same
    """

    num_basis: int

    def __init__(
        self,
        r_max: float,
        r_min: float = 0.0,
        original_basis=BesselBasis,
        original_basis_kwargs: dict = {},
        n: int = 4000,
        norm_basis_mean_shift: bool = True,
    ):
        super().__init__()
        self.basis = original_basis(**original_basis_kwargs)
        self.r_min = r_min
        self.r_max = r_max
        assert self.r_min >= 0.0
        assert self.r_max > r_min
        self.n = n

        self.num_basis = self.basis.num_basis

        # Uniform distribution on [r_min, r_max)
        with torch.no_grad():
            # don't take 0 in case of weirdness like bessel at 0
            rs = torch.linspace(r_min, r_max, n + 1)[1:]
            bs = self.basis(rs)
            assert bs.ndim == 2 and len(bs) == n
            if norm_basis_mean_shift:
                basis_std, basis_mean = torch.std_mean(bs, dim=0)
            else:
                basis_std = bs.square().mean().sqrt()
                basis_mean = torch.as_tensor(
                    0.0, device=basis_std.device, dtype=basis_std.dtype
                )

        self.register_buffer("_mean", basis_mean)
        self.register_buffer("_inv_std", torch.reciprocal(basis_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.basis(x) - self._mean) * self._inv_std
