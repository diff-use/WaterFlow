"""Unit tests for scripts/train_confidence.py -- backbone freezing and the training step."""

import argparse

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData

from scripts.train_confidence import (
    BACKBONE_MODULE_NAMES,
    freeze_backbone,
    train_one_epoch,
)
from src.confidence import ConfidenceGVP
from src.constants import EDGE_PP, NUM_RBF


def _confidence_graph(n_prot=8, n_cand=4):
    """One graph shaped like a ConfidenceDataset sample."""
    data = HeteroData()
    data["protein"].x = F.one_hot(
        torch.randint(0, 16, (n_prot,)), num_classes=16
    ).float()
    data["protein"].pos = torch.randn(n_prot, 3)
    data["protein"].num_nodes = n_prot
    data["water"].x = F.one_hot(torch.full((n_cand,), 2), num_classes=16).float()
    data["water"].pos = torch.randn(n_cand, 3)
    data["water"].num_nodes = n_cand
    data["water"].target_confidence = torch.rand(n_cand)
    data["water"].within_accept_radius = (torch.arange(n_cand) % 2).float()
    data[EDGE_PP].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    data[EDGE_PP].edge_unit_vectors = torch.randn(3, 3)
    data[EDGE_PP].edge_rbf = torch.randn(3, NUM_RBF)
    return data


def _train_args(**overrides):
    args = argparse.Namespace(
        grad_clip=1.0,
        warmup_steps=0,
        freeze_backbone=True,
        grad_accum_steps=1,
    )
    return argparse.Namespace(**{**vars(args), **overrides})


@pytest.mark.unit
class TestTrainConfidenceFreezing:
    def test_freeze_backbone_leaves_only_score_head_trainable(
        self, device, gvp_encoder
    ):
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )

        freeze_backbone(model)

        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        assert trainable
        assert all(name.startswith("score_head.") for name in trainable)

    def test_train_one_epoch_frozen_backbone_trains_only_score_head(
        self, device, gvp_encoder
    ):
        batch = Batch.from_data_list([_confidence_graph()]).to(device)
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

        # Grads are zeroed after the step, so compare weights, not .grad.
        head_before = [p.detach().clone() for p in model.score_head.parameters()]
        backbone_before = [p.detach().clone() for p in model.updater.parameters()]

        train_one_epoch(
            model=model,
            loader=[batch],
            optimizer=optimizer,
            warmup_scheduler=None,
            device=device,
            args=_train_args(),
            step_counter=0,
        )

        assert model.score_head.training
        for name in BACKBONE_MODULE_NAMES:
            assert not getattr(model, name).training

        assert any(
            not torch.equal(before, param)
            for before, param in zip(head_before, model.score_head.parameters())
        )
        assert all(
            torch.equal(before, param)
            for before, param in zip(backbone_before, model.updater.parameters())
        )

    def test_unfrozen_backbone_stays_in_train_mode(self, device, gvp_encoder):
        batch = Batch.from_data_list([_confidence_graph()]).to(device)
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        backbone_before = [p.detach().clone() for p in model.updater.parameters()]

        train_one_epoch(
            model=model,
            loader=[batch],
            optimizer=optimizer,
            warmup_scheduler=None,
            device=device,
            args=_train_args(freeze_backbone=False),
            step_counter=0,
        )

        assert model.updater.training
        assert any(
            not torch.equal(before, param)
            for before, param in zip(backbone_before, model.updater.parameters())
        )
