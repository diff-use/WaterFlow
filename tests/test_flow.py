"""Unit tests for flow.py

All test cases created with assistance from Claude Code and refined.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, HeteroData

from src.flow import (
    FlowMatcher,
    FlowWaterGVP,
    ProteinWaterUpdate,
    build_knn_edges,
)
from src.gvp_encoder import GVPEncoder, ProteinGVPEncoder, make_encoder_data


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_hetero_data(device):
    """Minimal HeteroData with protein and water nodes."""
    data = HeteroData()
    
    # Protein: 10 atoms
    data['protein'].pos = torch.randn(10, 3, device=device)
    data['protein'].x = torch.randn(10, 16, device=device)
    data['protein'].batch = torch.zeros(10, dtype=torch.long, device=device)
    
    # Water: 5 molecules
    data['water'].pos = torch.randn(5, 3, device=device)
    data['water'].x = torch.randn(5, 16, device=device)
    data['water'].batch = torch.zeros(5, dtype=torch.long, device=device)
    
    # Protein-protein edges
    data['protein', 'pp', 'protein'].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long, device=device
    )
    
    return data


@pytest.fixture
def batched_hetero_data(device):
    """HeteroData with 2 graphs batched."""
    data = HeteroData()
    
    # Protein: 20 atoms (10 per graph)
    data['protein'].pos = torch.randn(20, 3, device=device)
    data['protein'].x = torch.randn(20, 16, device=device)
    data['protein'].batch = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(10, dtype=torch.long)
    ]).to(device)
    
    # Water: 8 molecules (4 per graph)
    data['water'].pos = torch.randn(8, 3, device=device)
    data['water'].x = torch.randn(8, 16, device=device)
    data['water'].batch = torch.cat([
        torch.zeros(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long)
    ]).to(device)
    
    data['protein', 'pp', 'protein'].edge_index = torch.tensor(
        [[0, 1, 10, 11], [1, 2, 11, 12]], dtype=torch.long, device=device
    )
    
    return data


@pytest.fixture
def mock_encoder(device):
    """Mock BaseProteinEncoder."""
    encoder = Mock()
    encoder.output_dims = (256, 32)  # Required by FlowWaterGVP
    encoder.encoder_type = 'mock'
    encoder.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))
    encoder.eval = Mock()

    def mock_forward(data):
        n = data['protein'].pos.size(0)
        s = torch.randn(n, 256, device=device)
        v = torch.randn(n, 32, 3, device=device)
        # Return 3 values: (s, V, pp_edge_attr)
        # Mock encoder returns None for edge features (like SLAE/ESM)
        return s, v, None

    encoder.side_effect = mock_forward
    encoder.__call__ = mock_forward
    return encoder


@pytest.mark.unit
class TestBuildKnnEdges:
    
    def test_basic_knn(self, device):
        src = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], device=device)
        dst = torch.tensor([[0.5, 0., 0.], [1.5, 0., 0.]], device=device)
        
        edges = build_knn_edges(src, dst, k=2)
        
        assert edges.shape[0] == 2
        assert edges.shape[1] > 0
        assert edges.dtype == torch.long
    
    def test_empty_src(self, device):
        src = torch.empty(0, 3, device=device)
        dst = torch.randn(5, 3, device=device)
        
        edges = build_knn_edges(src, dst, k=3)
        
        assert edges.shape == (2, 0)
    
    def test_empty_dst(self, device):
        src = torch.randn(5, 3, device=device)
        dst = torch.empty(0, 3, device=device)
        
        edges = build_knn_edges(src, dst, k=3)
        
        assert edges.shape == (2, 0)
    
    def test_self_edges_removed(self, device):
        pos = torch.randn(10, 3, device=device)
        
        edges = build_knn_edges(pos, pos, k=5)
        
        # No self-loops
        assert (edges[0] != edges[1]).all()
    
    def test_with_batch(self, device):
        src = torch.randn(10, 3, device=device)
        dst = torch.randn(8, 3, device=device)
        batch_src = torch.cat([torch.zeros(5), torch.ones(5)]).long().to(device)
        batch_dst = torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device)
        
        edges = build_knn_edges(src, dst, k=3, batch_src=batch_src, batch_dst=batch_dst)
        
        assert edges.shape[0] == 2
        assert edges.shape[1] > 0


@pytest.mark.unit
class TestMakeEncoderData:
    
    def test_basic_output(self, simple_hetero_data):
        enc_data = make_encoder_data(simple_hetero_data)

        assert isinstance(enc_data, Data)
        assert hasattr(enc_data, 'x')
        assert hasattr(enc_data, 'pos')
        assert hasattr(enc_data, 'edge_index')
    
    def test_shapes(self, simple_hetero_data):
        enc_data = make_encoder_data(simple_hetero_data)

        n_nodes = simple_hetero_data['protein'].pos.size(0)
        n_edges = simple_hetero_data['protein', 'pp', 'protein'].edge_index.size(1)

        assert enc_data.x.shape[0] == n_nodes
        assert enc_data.pos.shape == (n_nodes, 3)
        assert enc_data.edge_index.shape == (2, n_edges)
    
    def test_batch_preserved(self, batched_hetero_data):
        enc_data = make_encoder_data(batched_hetero_data)

        assert hasattr(enc_data, 'batch')
        assert enc_data.batch.shape[0] == batched_hetero_data['protein'].pos.size(0)
    
    def test_no_edges(self, device):
        data = HeteroData()
        data['protein'].pos = torch.randn(10, 3, device=device)
        data['protein'].x = torch.randn(10, 16, device=device)
        # No edges defined

        enc_data = make_encoder_data(data)

        assert enc_data.edge_index.shape == (2, 0)


@pytest.mark.unit
class TestProteinWaterUpdate:
    
    def test_init(self):
        updater = ProteinWaterUpdate(
            hidden_dims=(128, 16),
            rbf_dim=16,
            layers=2,
        )
        
        assert len(updater.blocks) == 2
        assert ('protein', 'pw', 'water') in updater.etypes
        assert ('water', 'ww', 'water') in updater.etypes
    
    def test_init_always_includes_all_edge_types(self):
        updater = ProteinWaterUpdate(
            hidden_dims=(128, 16),
            rbf_dim=16,
            layers=2,
        )

        assert ('protein', 'pp', 'protein') in updater.etypes
        assert ('water', 'wp', 'protein') in updater.etypes
    
    def test_build_edges(self, simple_hetero_data):
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)
        
        edge_dict = updater.build_edges(simple_hetero_data, k_pw=4, k_ww=3)
        
        assert ('protein', 'pw', 'water') in edge_dict
        assert ('water', 'ww', 'water') in edge_dict
        assert edge_dict[('protein', 'pw', 'water')].shape[0] == 2
    
    def test_build_edges_empty_water(self, device):
        data = HeteroData()
        data['protein'].pos = torch.randn(10, 3, device=device)
        data['protein'].x = torch.randn(10, 16, device=device)
        data['water'].pos = torch.empty(0, 3, device=device)
        data['water'].x = torch.empty(0, 16, device=device)
        
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)
        edge_dict = updater.build_edges(data)
        
        assert edge_dict[('protein', 'pw', 'water')].shape == (2, 0)
        assert edge_dict[('water', 'ww', 'water')].shape == (2, 0)
    
    def test_forward_shapes(self, simple_hetero_data, device):
        s_h, v_h = 128, 16
        updater = ProteinWaterUpdate(hidden_dims=(s_h, v_h), layers=1).to(device)
        
        n_p = simple_hetero_data['protein'].pos.size(0)
        n_w = simple_hetero_data['water'].pos.size(0)
        
        x_dict = {
            'protein': (torch.randn(n_p, s_h, device=device), 
                       torch.randn(n_p, v_h, 3, device=device)),
            'water': (torch.randn(n_w, s_h, device=device), 
                     torch.randn(n_w, v_h, 3, device=device)),
        }
        
        out = updater(x_dict, simple_hetero_data)
        
        assert out['water'][0].shape == (n_w, s_h)
        assert out['water'][1].shape == (n_w, v_h, 3)


# ============== Tests for FlowWaterGVP ==============

@pytest.mark.unit
class TestFlowWaterGVP:
    
    def test_init(self, mock_encoder, device):
        model = FlowWaterGVP(
            encoder=mock_encoder,
            hidden_dims=(128, 16),
            layers=2,
        ).to(device)
        
        assert model.hidden_dims == (128, 16)
        assert model.layers == 2
    
    def test_forward_output_shape(self, simple_hetero_data, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)
        
        t = torch.tensor([0.5], device=device)
        v_pred = model(simple_hetero_data, t)
        
        n_water = simple_hetero_data['water'].num_nodes
        assert v_pred.shape == (n_water, 3)
    
    def test_forward_no_water(self, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)
        
        data = HeteroData()
        data['protein'].pos = torch.randn(10, 3, device=device)
        data['protein'].x = torch.randn(10, 16, device=device)
        data['protein'].batch = torch.zeros(10, dtype=torch.long, device=device)
        data['protein', 'pp', 'protein'].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long, device=device
        )
        # No water nodes
        
        t = torch.tensor([0.5], device=device)
        v_pred = model(data, t)
        
        assert v_pred.shape == (0, 3)
    
    def test_self_conditioning(self, simple_hetero_data, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)
        
        n_water = simple_hetero_data['water'].num_nodes
        sc = {'x1_pred': torch.randn(n_water, 3, device=device)}
        t = torch.tensor([0.5], device=device)
        
        v_pred = model(simple_hetero_data, t, sc=sc)
        
        assert v_pred.shape == (n_water, 3)


# ============== Tests for FlowMatcher ==============

@pytest.mark.unit
class TestFlowMatcher:
    
    @pytest.fixture
    def flow_matcher(self, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        return FlowMatcher(model, p_self_cond=0.5)
    
    def test_compute_sigma(self, simple_hetero_data):
        sigma = FlowMatcher.compute_sigma(simple_hetero_data)
        
        assert isinstance(sigma, float)
        assert sigma > 0
    
    def test_training_step(self, flow_matcher, simple_hetero_data, device):
        optimizer = torch.optim.Adam(flow_matcher.model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        result = flow_matcher.training_step(
            simple_hetero_data, use_self_conditioning=False
        )
        optimizer.step()

        assert 'loss' in result
        assert 'rmsd' in result
        assert 'sigma' in result
        assert result['loss'] >= 0

    def test_training_step_with_self_cond(self, flow_matcher, simple_hetero_data, device):
        optimizer = torch.optim.Adam(flow_matcher.model.parameters(), lr=1e-4)

        # Force self-conditioning
        flow_matcher.p_self_cond = 1.0
        optimizer.zero_grad()
        result = flow_matcher.training_step(
            simple_hetero_data, use_self_conditioning=True
        )
        optimizer.step()
        
        assert 'loss' in result
    
    def test_validation_step(self, flow_matcher, simple_hetero_data):
        result = flow_matcher.validation_step(simple_hetero_data)
        
        assert 'loss' in result
        assert 'rmsd' in result
        assert result['loss'] >= 0
    
    @pytest.mark.slow
    def test_euler_integrate(self, flow_matcher, simple_hetero_data, device):
        results = flow_matcher.euler_integrate(
            simple_hetero_data, num_steps=5, use_sc=False, device=str(device)
        )
        # euler_integrate returns List[np.ndarray], one per input graph
        water_pred = results[0]

        n_water = simple_hetero_data['water'].num_nodes
        assert water_pred.shape == (n_water, 3)
        assert isinstance(water_pred, np.ndarray)

    @pytest.mark.slow
    def test_rk4_integrate(self, flow_matcher, simple_hetero_data, device):
        results = flow_matcher.rk4_integrate(
            simple_hetero_data, num_steps=5, use_sc=False,
            device=str(device), return_trajectory=True
        )
        # rk4_integrate returns List[Dict], one per input graph
        result = results[0]

        assert 'water_pred' in result
        assert 'water_true' in result
        assert 'protein_pos' in result
        assert 'trajectory' in result
        assert len(result['trajectory']) == 5
    
    def test_sample_euler(self, flow_matcher, simple_hetero_data, device):
        water_pred = flow_matcher.sample(
            simple_hetero_data, num_steps=3, method="euler", device=str(device)
        )
        
        n_water = simple_hetero_data['water'].num_nodes
        assert water_pred.shape == (n_water, 3)
    
    def test_sample_rk4(self, flow_matcher, simple_hetero_data, device):
        water_pred = flow_matcher.sample(
            simple_hetero_data, num_steps=3, method="rk4", device=str(device)
        )
        
        n_water = simple_hetero_data['water'].num_nodes
        assert water_pred.shape == (n_water, 3)


# ============== Tests for distortion ==============

@pytest.mark.unit
class TestDistortion:
    
    def test_distortion_enabled(self, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        fm = FlowMatcher(
            model,
            use_distortion=True,
            p_distort=1.0,  # Always apply
            t_distort=0.0,  # Apply at all times
            sigma_distort=0.5
        )

        assert fm.use_distortion is True
        assert fm.p_distort == 1.0


# ============== Edge case tests ==============

@pytest.mark.unit
class TestEdgeCases:
    
    def test_single_water_molecule(self, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)
        
        data = HeteroData()
        data['protein'].pos = torch.randn(10, 3, device=device)
        data['protein'].x = torch.randn(10, 16, device=device)
        data['protein'].batch = torch.zeros(10, dtype=torch.long, device=device)
        data['water'].pos = torch.randn(1, 3, device=device)  # Single water
        data['water'].x = torch.randn(1, 16, device=device)
        data['water'].batch = torch.zeros(1, dtype=torch.long, device=device)
        data['protein', 'pp', 'protein'].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long, device=device
        )
        
        t = torch.tensor([0.5], device=device)
        v_pred = model(data, t)
        
        assert v_pred.shape == (1, 3)
    
    def test_frozen_gvp_encoder(self, device):
        """Freezing is handled by the encoder itself, not FlowWaterGVP."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=True)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        # Verify encoder params are frozen
        for p in model.encoder.encoder.parameters():
            assert p.requires_grad is False


# ============== Tests for edge connectivity ==============

@pytest.mark.unit
class TestWaterEdgeConnectivity:
    """Tests to ensure all waters have edges (both protein-water and water-water)."""

    def test_all_waters_have_protein_edges(self, simple_hetero_data):
        """Ensure every water has at least one protein-water edge."""
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(simple_hetero_data, k_pw=4, k_ww=3)
        pw_edges = edge_dict[('protein', 'pw', 'water')]

        n_water = simple_hetero_data['water'].num_nodes

        # Check that all water nodes appear in the protein-water edges
        water_nodes_with_edges = torch.unique(pw_edges[1])
        assert len(water_nodes_with_edges) == n_water, \
            f"Only {len(water_nodes_with_edges)}/{n_water} waters have protein edges"

    def test_all_waters_have_water_edges(self, simple_hetero_data):
        """Ensure every water has at least one water-water edge (if multiple waters exist)."""
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(simple_hetero_data, k_pw=4, k_ww=3)
        ww_edges = edge_dict[('water', 'ww', 'water')]

        n_water = simple_hetero_data['water'].num_nodes

        if n_water > 1:
            # Check that all water nodes appear in the water-water edges
            water_nodes_with_edges = torch.unique(ww_edges[0])
            assert len(water_nodes_with_edges) == n_water, \
                f"Only {len(water_nodes_with_edges)}/{n_water} waters have water-water edges"

    def test_batched_waters_have_edges(self, batched_hetero_data):
        """Ensure all waters in a batched graph have edges."""
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(batched_hetero_data, k_pw=4, k_ww=3)
        pw_edges = edge_dict[('protein', 'pw', 'water')]
        ww_edges = edge_dict[('water', 'ww', 'water')]

        n_water = batched_hetero_data['water'].num_nodes

        # Check protein-water edges
        water_nodes_with_pw_edges = torch.unique(pw_edges[1])
        assert len(water_nodes_with_pw_edges) == n_water, \
            f"Only {len(water_nodes_with_pw_edges)}/{n_water} waters have protein edges in batched data"

        # Check water-water edges
        if n_water > 1:
            water_nodes_with_ww_edges = torch.unique(ww_edges[0])
            assert len(water_nodes_with_ww_edges) == n_water, \
                f"Only {len(water_nodes_with_ww_edges)}/{n_water} waters have water-water edges in batched data"

    def test_single_water_has_protein_edges_no_water_edges(self, device):
        """A single water should have protein edges but no water-water edges."""
        data = HeteroData()
        data['protein'].pos = torch.randn(10, 3, device=device)
        data['protein'].x = torch.randn(10, 16, device=device)
        data['protein'].batch = torch.zeros(10, dtype=torch.long, device=device)
        data['water'].pos = torch.randn(1, 3, device=device)  # Single water
        data['water'].x = torch.randn(1, 16, device=device)
        data['water'].batch = torch.zeros(1, dtype=torch.long, device=device)
        data['protein', 'pp', 'protein'].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long, device=device
        )

        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)
        edge_dict = updater.build_edges(data, k_pw=4, k_ww=3)

        pw_edges = edge_dict[('protein', 'pw', 'water')]
        ww_edges = edge_dict[('water', 'ww', 'water')]

        # Single water should have protein edges
        assert pw_edges.shape[1] > 0, "Single water should have at least one protein edge"
        water_nodes_with_edges = torch.unique(pw_edges[1])
        assert len(water_nodes_with_edges) == 1, "Single water must have protein edges"

        # Single water should have no water-water edges (since k_ww excludes self-loops)
        assert ww_edges.shape[1] == 0, "Single water should have no water-water edges"