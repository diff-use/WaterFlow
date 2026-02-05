import pytest
import torch
import torch.nn.functional as F

from src.gvp import GVP, tuple_sum, tuple_cat, _merge, _split, LayerNorm, Dropout


class TestGVPHelpers:
    """Tests for GVP helper functions."""
    
    def test_tuple_sum(self):
        """Test tuple summation."""
        t1 = (torch.ones(5, 3), torch.ones(5, 2, 3))
        t2 = (torch.ones(5, 3) * 2, torch.ones(5, 2, 3) * 2)
        result = tuple_sum(t1, t2)
        
        assert torch.allclose(result[0], torch.ones(5, 3) * 3)
        assert torch.allclose(result[1], torch.ones(5, 2, 3) * 3)
    
    def test_tuple_cat(self):
        """Test tuple concatenation."""
        t1 = (torch.ones(5, 3), torch.ones(5, 2, 3))
        t2 = (torch.ones(5, 4), torch.ones(5, 3, 3))
        result = tuple_cat(t1, t2, dim=-1)
        
        assert result[0].shape == (5, 7)  # 3 + 4
        assert result[1].shape == (5, 5, 3)  # 2 + 3
    
    def test_merge_split(self):
        """Test merge and split are inverses."""
        s = torch.randn(10, 5)
        v = torch.randn(10, 3, 3)
        
        merged = _merge(s, v)
        s_out, v_out = _split(merged, nv=3)
        
        assert torch.allclose(s, s_out)
        assert torch.allclose(v, v_out)


class TestGVP:
    """Tests for GVP layer."""
    
    def test_scalar_only_forward(self):
        """Test GVP with only scalar inputs/outputs."""
        gvp = GVP(in_dims=(10, 0), out_dims=(5, 0))
        x = torch.randn(8, 10)
        
        out = gvp(x)
        
        assert isinstance(out, torch.Tensor)
        assert out.shape == (8, 5)
    
    def test_vector_forward(self):
        """Test GVP with vector inputs/outputs."""
        gvp = GVP(
            in_dims=(10, 5), 
            out_dims=(8, 3),
            activations=(F.relu, torch.sigmoid),
            vector_gate=True
        )
        s = torch.randn(16, 10)
        v = torch.randn(16, 5, 3)
        
        s_out, v_out = gvp((s, v))
        
        assert s_out.shape == (16, 8)
        assert v_out.shape == (16, 3, 3)
    
    def test_no_vector_input(self):
        """Test GVP with scalar input, vector output."""
        gvp = GVP(in_dims=(10, 0), out_dims=(8, 3))
        x = torch.randn(16, 10)
        
        s_out, v_out = gvp(x)
        
        assert s_out.shape == (16, 8)
        assert v_out.shape == (16, 3, 3)
    
    def test_deterministic_forward(self):
        """Test forward pass is deterministic."""
        gvp = GVP(in_dims=(5, 2), out_dims=(3, 2))
        s = torch.randn(4, 5)
        v = torch.randn(4, 2, 3)
        
        out1 = gvp((s, v))
        out2 = gvp((s, v))
        
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])


class TestLayerNorm:
    """Tests for GVP LayerNorm."""
    
    def test_scalar_only(self):
        """Test with scalar features only."""
        ln = LayerNorm((10, 0))
        x = torch.randn(8, 10)
        
        out = ln(x)
        
        assert out.shape == (8, 10)
        # Check normalization
        assert torch.allclose(out.mean(dim=-1), torch.zeros(8), atol=1e-6)
    
    def test_scalar_vector(self):
        """Test with both scalar and vector features."""
        ln = LayerNorm((10, 5))
        s = torch.randn(8, 10)
        v = torch.randn(8, 5, 3)
        
        s_out, v_out = ln((s, v))
        
        assert s_out.shape == (8, 10)
        assert v_out.shape == (8, 5, 3)


class TestDropout:
    """Tests for GVP Dropout."""
    
    def test_dropout_training(self):
        """Test dropout in training mode."""
        drop = Dropout(drop_rate=0.5)
        drop.train()
        
        s = torch.ones(100, 10)
        v = torch.ones(100, 5, 3)
        
        s_out, v_out = drop((s, v))
        
        # Some values should be zeroed
        assert not torch.allclose(s_out, s)
    
    def test_dropout_eval(self):
        """Test dropout in eval mode (should be identity)."""
        drop = Dropout(drop_rate=0.5)
        drop.eval()
        
        s = torch.ones(100, 10)
        v = torch.ones(100, 5, 3)
        
        s_out, v_out = drop((s, v))
        
        # Should be unchanged in eval mode
        assert torch.allclose(s_out, s)
        assert torch.allclose(v_out, v)