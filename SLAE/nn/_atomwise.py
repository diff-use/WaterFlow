import logging
from typing import Optional, List, Union

import torch

from torch_geometric.data import Batch
import torch.nn as nn
import torch.nn.functional
from torch_scatter import scatter, scatter_softmax
#from torch_runstats.scatter import scatter


from SLAE.nn.graph_mixin import GraphModuleMixin


class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    constant: float

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        pooling_type="mean", # sum, normalized_sum, attention, gated
        avg_num_atoms=None,
        irreps_in={},
        feature_dim: int = 128,
        reduction_level: str = "residue", # residue, protein
    ):
        super().__init__()
        assert pooling_type in ("sum", "mean", "normalized_sum", "attention", "gated")

        self.constant = 1.0
        if pooling_type == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            pooling_type = "sum"
        self.pooling_type = pooling_type
        self.field = field
        self.out_field = f"{pooling_type}_{field}" if out_field is None else out_field
        self.reduction_level = reduction_level
        assert self.reduction_level in ["residue", "protein"]
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )

        # Define attention or gated pooling if selected
        if pooling_type == "attention":
            self.attention_query = nn.Parameter(torch.randn(feature_dim))  # Learnable query vector
            self.attention_proj = nn.Linear(feature_dim, feature_dim)      # Linear projection for attention
        elif pooling_type == "gated":
            self.gate_proj = nn.Linear(feature_dim, feature_dim)
            self.feature_proj = nn.Linear(feature_dim, feature_dim)


    def forward(self,
                batch: Batch,
                ) -> Batch:
        """
        :param batch: Batch containing atom-level (node) features.
        :param reduction_level: Level at which to perform reduction. 
                                Options are 'residue' and 'protein'.
        :return: Reduced representation as per the specified reduction level.
        """

        field = getattr(batch, self.field)

        if self.reduction_level == "residue":
            reduction_index = batch.residue_index
            
        elif self.reduction_level == "protein":
            reduction_index = batch.batch


            
        result = self._apply_reduction(field, reduction_index)

        # Apply normalization if needed
        if self.constant != 1.0:
            result = result * self.constant

        #batch[self.out_field] = result
        #if self.reduction_level == "residue":
        #    assert result.size(0) == batch.coords.size(0), f"Size mismatch in residue-level reduction, range of indices is from {torch.min(batch.residue_index)} to {torch.max(batch.residue_index)} in {batch}"
        setattr(batch, self.out_field, result)
        return batch

    def _apply_reduction(self, features, batch_index):
        """Apply the selected pooling method using torch_scatter functions."""
        if self.pooling_type == "sum":
            return self._sum_pooling(features, batch_index)
        elif self.pooling_type == "mean":
            return self._mean_pooling(features, batch_index)
        elif self.pooling_type == "attention":
            return self._attention_pooling(features, batch_index)
        elif self.pooling_type == "gated":
            return self._gated_pooling(features, batch_index)

    def _sum_pooling(self, features, batch_index):
        """Sum pooling using torch_scatter.scatter."""
        result = scatter(features, batch_index, dim=0, reduce='sum')
        return result

    def _mean_pooling(self, features, batch_index):
        """Mean pooling using torch_scatter.scatter."""
        result = scatter(features, batch_index, dim=0, reduce='mean')
        return result

    def _attention_pooling(self, features, batch_index):
        """Attention-based pooling using torch_scatter functions."""
        # Compute attention scores
        projected_features = self.attention_proj(features)
        scores = torch.matmul(projected_features, self.attention_query)

        # Compute attention weights using scatter_softmax
        attn_weights = scatter_softmax(scores, batch_index, dim=0)

        # Weighted sum of features with attention weights
        weighted_features = features * attn_weights.unsqueeze(-1)
        result = scatter(weighted_features, batch_index, dim=0, reduce='sum')
        return result

    def _gated_pooling(self, features, batch_index):
        """Gated pooling using torch_scatter functions."""
        # Apply gating to each feature
        gated_features = torch.sigmoid(self.gate_proj(features)) * self.feature_proj(features)

        # Sum the gated features
        result = scatter(gated_features, batch_index, dim=0, reduce='sum')
        return result