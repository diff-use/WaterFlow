from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def random_quaternion(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random quaternions.
    
    Args:
        batch_size (int): The number of quaternions to generate.
        device (torch.device): The device to generate the quaternions on.
    """
    q = torch.randn(batch_size, 4, device=device)
    return q




class RigidTransformer(nn.Module):
    """Rigid-body rotation and translation (batch form).

    This layer applies a rigid body rotation and translation,
    and can be composed with other generative geometry layers to modify poses.

    Internally, the coordinates are centered before rotation and translation.
    The rotation itself is parameterized as a quaternion to prevent
    Gimbal lock (https://en.wikipedia.org/wiki/Gimbal_lock).

    Args:
        center_intput (Boolean): Center the input coordinates (default: True)
            default.

    Inputs:
        X (torch.Tensor): Input coordinates with shape `(batch_size, ..., 3)`.
        dX (torch.Tensor): Translation vector with shape `(batch_size, 3)`.
        q (torch.Tensor): Rotation vector (quaternion) with shape `(batch_size, 4)`.
            It can be any 4-element real vector, but will internally be
            normalized to a unit quaternion.
        mask (tensor,optional): Mask tensor with shape `(batch_size, ..., 3)`.

    Outputs:
        X_t (torch.Tensor): Transformed coordinates with shape `(batch_size, ..., 3)`.
    """

    def __init__(self, center_rotation: bool = True, keep_centered: bool = False):
        super(RigidTransformer, self).__init__()
        self.center_rotation = center_rotation
        self.keep_centered = keep_centered
        self.dist_eps = 1e-5

    def _rotation_matrix(self, q_unc: torch.Tensor) -> torch.Tensor:
        """Build rotation matrix from quaternion parameters.

        See en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for further
        details on converting between quaternions and rotation matrices.

        Args:
            q_unc (torch.Tensor): Unnormalized quaternion representing rotation with
                shape `(batch_size, 3)`.

        Returns:
            R (torch.Tensor): Rotation matrix with shape `(batch_size, 3)`.
        """
        num_batch = q_unc.shape[0]
        q = F.normalize(q_unc, dim=-1)

        # fmt: off
        a,b,c,d = q.unbind(-1)
        a2,b2,c2,d2 = a**2, b**2, c**2, d**2
        R = torch.stack([
            a2 + b2 - c2 - d2,      2*b*c - 2*a*d,      2*b*d + 2*a*c,
                2*b*c + 2*a*d,  a2 - b2 + c2 - d2,      2*c*d - 2*a*b,
                2*b*d - 2*a*c,      2*c*d + 2*a*b,  a2 - b2 - c2 + d2
        ], dim=-1)
        # fmt: on

        R = R.view([num_batch, 3, 3])
        return R

    def forward(
        self,
        X: torch.Tensor,
        q: torch.Tensor,
        dX: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_batch = X.shape[0]
        X_flat = X.reshape([num_batch, -1, 3])

        # Flatten mask
        if mask is not None:
            shape_mask = list(mask.size())
            shape_X = list(X.size())
            shape_mask_expand = shape_mask + [
                1 for i in range(len(shape_X) - 1 - len(shape_mask))
            ]
            shape_mask_flat = list(X_flat.size())[:-1] + [1]

            mask_flat = mask.reshape(shape_mask_expand).expand(shape_X[:-1])
            mask_flat = mask_flat.reshape(shape_mask_flat)

            # Compute center
            X_mean = torch.sum(mask_flat * X_flat, 1, keepdims=True) / (
                torch.sum(mask_flat, 1, keepdims=True) + self.dist_eps
            )
        else:
            X_mean = torch.mean(X_flat, 1, keepdims=True)

        # Rotate around center of mass
        if self.center_rotation:
            X_centered = X_flat - X_mean
        else:
            X_centered = X_flat
        R = self._rotation_matrix(q)
        X_rotate = torch.einsum("bxr,bir->bix", R, X_centered)

        # Optionally preserve original centering
        if self.center_rotation and not self.keep_centered:
            X_rotate = X_rotate + X_mean

        # Translate
        if dX is not None:
            X_transform = X_rotate + dX.unsqueeze(1)
        else:
            X_transform = X_rotate

        if mask is not None:
            X_transform = mask_flat * X_transform + (1 - mask_flat) * X_flat

        X_transform = X_transform.view(X.shape)
        return X_transform
    


