import torch
import math 

def cosine_cutoff(x: torch.Tensor, r_max: torch.Tensor, r_start_cos_ratio: float = 0.8):
    """A piecewise cosine cutoff starting the cosine decay at r_decay_factor*r_max.

    Broadcasts over r_max.
    """
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    r_decay: torch.Tensor = r_start_cos_ratio * r_max
    # for x < r_decay, clamps to 1, for x > r_max, clamps to 0
    x = x.clamp(r_decay, r_max)
    return 0.5 * (torch.cos((math.pi / (r_max - r_decay)) * (x - r_decay)) + 1.0)


@torch.jit.script
def polynomial_cutoff_broadcasted(
    x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0
) -> torch.Tensor:
    """Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


    Parameters
    ----------
    r_max : tensor
        Broadcasts over r_max.

    p : int
        Power used in envelope function
    """
    assert p >= 2.0
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    x = x / r_max

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)


def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)


class PolynomialCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return _poly_cutoff(x, self._factor, p=self.p)
