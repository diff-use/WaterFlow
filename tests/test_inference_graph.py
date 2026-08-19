"""Tests for src/inference_graph.py -- building the flow-model graph from a raw file.

Runs in gvp mode so no precomputed embeddings are needed. The equivalence tests
pin the inference graph to what ProteinWaterDataset produces, guarding against
drift between the two graph-construction paths.
"""

from pathlib import Path

import pytest
import torch

from src.dataset import ProteinWaterDataset
from src.inference_graph import build_inference_graph


EDGE_PP = ("protein", "pp", "protein")


def _base_dir(fixture_path):
    # tests/test_files/<id>/<id>_final.pdb -> tests/test_files
    return str(Path(fixture_path).parent.parent)


def _dataset_graph(fixture_path, tmp_path, *, include_mates, include_ligands=True):
    """A ProteinWaterDataset graph for the same structure, filters off."""
    cache_key = Path(fixture_path).stem
    list_file = tmp_path / "list.txt"
    list_file.write_text(f"{cache_key}\n")
    ds = ProteinWaterDataset(
        pdb_list_file=str(list_file),
        processed_dir=str(tmp_path / "processed"),
        base_pdb_dir=_base_dir(fixture_path),
        encoder_type="gvp",
        include_mates=include_mates,
        include_ligands=include_ligands,
        preprocess=True,
        filter_by_distance=False,
        filter_by_edia=False,
        filter_by_bfactor=False,
    )
    return ds[0]


@pytest.mark.unit
class TestInferenceGraphContract:
    def test_drops_waters_and_centers(self, pdb_6eey):
        # 6eey carries crystallographic waters; they must not become nodes.
        data = build_inference_graph(pdb_6eey, encoder_type="gvp")
        assert data["water"].num_nodes == 0
        assert data["water"].pos.shape == (0, 3)
        assert data["protein"].pos.mean(dim=0).abs().max().item() < 1e-4  # centered

    def test_protein_node_contract(self, pdb_6eey):
        data = build_inference_graph(pdb_6eey, encoder_type="gvp")
        p = data["protein"]
        assert p.x.shape == (p.num_nodes, 16)
        assert p.pos.shape == (p.num_nodes, 3)
        assert p.residue_index.shape == (p.num_nodes,)
        assert p.is_mate.shape == (p.num_nodes,) and not p.is_mate.any()
        assert p.is_ligand.shape == (p.num_nodes,)
        assert data.num_asu_protein_atoms == p.num_nodes
        assert data.pdb_id == "6eey_final"
        # PP edges present with 16-dim RBF features
        assert data[EDGE_PP].edge_index.shape[0] == 2
        assert data[EDGE_PP].edge_rbf.shape[1] == 16

    def test_cif_input_matches_pdb(self, pdb_6eey, cif_6eey):
        from_pdb = build_inference_graph(pdb_6eey, encoder_type="gvp")
        from_cif = build_inference_graph(cif_6eey, encoder_type="gvp")
        assert from_pdb["protein"].num_nodes == from_cif["protein"].num_nodes
        assert torch.allclose(
            from_pdb["protein"].pos, from_cif["protein"].pos, atol=1e-4
        )

    def test_gvp_attaches_no_embedding(self, pdb_6eey):
        data = build_inference_graph(pdb_6eey, encoder_type="gvp")
        assert "embedding" not in data["protein"]

    def test_embedding_encoder_requires_processed_dir(self, pdb_6eey):
        with pytest.raises(ValueError, match="processed_dir"):
            build_inference_graph(pdb_6eey, encoder_type="esm", processed_dir=None)

    def test_ligands_appended_after_protein(self, pdb_4h0b):
        data = build_inference_graph(pdb_4h0b, encoder_type="gvp", include_ligands=True)
        p = data["protein"]
        assert p.is_ligand.any()
        # ligands are the trailing block: every ligand index is past the protein
        assert int(p.is_ligand.nonzero().min()) == int((~p.is_ligand).sum())

    def test_include_ligands_false_drops_them(self, pdb_4h0b):
        data = build_inference_graph(
            pdb_4h0b, encoder_type="gvp", include_ligands=False
        )
        assert not data["protein"].is_ligand.any()

    def test_mates_add_nodes_past_the_asu(self, pdb_6eey):
        asu = build_inference_graph(pdb_6eey, encoder_type="gvp", include_mates=False)
        mates = build_inference_graph(pdb_6eey, encoder_type="gvp", include_mates=True)
        assert mates["protein"].num_nodes > asu["protein"].num_nodes
        assert mates["protein"].is_mate.any()
        # ASU count is unchanged, and the ASU block is identical to the mates-off graph
        assert mates.num_asu_protein_atoms == asu["protein"].num_nodes
        n = asu["protein"].num_nodes
        assert torch.allclose(mates["protein"].pos[:n], asu["protein"].pos)
        assert not mates["protein"].is_mate[:n].any()
        assert mates["protein"].is_mate[n:].all()


@pytest.mark.integration
class TestEquivalenceWithDataset:
    """The inference protein graph must match ProteinWaterDataset's, so the model
    sees exactly what training preprocessing would have built."""

    @pytest.mark.parametrize("fixture", ["pdb_6eey", "pdb_4h0b"])
    def test_protein_graph_matches_dataset_no_mates(self, fixture, tmp_path, request):
        path = request.getfixturevalue(fixture)
        inf = build_inference_graph(path, encoder_type="gvp", include_mates=False)
        ds = _dataset_graph(path, tmp_path, include_mates=False)

        assert inf["protein"].num_nodes == ds["protein"].num_nodes
        assert torch.allclose(inf["protein"].pos, ds["protein"].pos, atol=1e-4)
        assert torch.equal(inf["protein"].x, ds["protein"].x)
        assert torch.equal(inf["protein"].residue_index, ds["protein"].residue_index)
        assert torch.equal(inf["protein"].is_ligand, ds["protein"].is_ligand)
        assert torch.equal(inf["protein"].is_mate, ds["protein"].is_mate)
        assert inf.num_asu_protein_atoms == ds.num_asu_protein_atoms
        assert inf["protein"].num_protein_residues == ds["protein"].num_protein_residues
        assert inf["protein"].num_residues == ds["protein"].num_residues
        # PP topology and edge features match (same deterministic radius graph)
        assert torch.equal(inf[EDGE_PP].edge_index, ds[EDGE_PP].edge_index)
        assert torch.allclose(
            inf[EDGE_PP].edge_unit_vectors, ds[EDGE_PP].edge_unit_vectors, atol=1e-4
        )
        assert torch.allclose(inf[EDGE_PP].edge_rbf, ds[EDGE_PP].edge_rbf, atol=1e-4)


@pytest.mark.unit
class TestBatching:
    def test_graphs_batch_via_pyg(self, pdb_6eey, pdb_4h0b):
        """Different-sized structures must batch through Batch.from_data_list --
        exactly what the flow integrators do internally to run many at once."""
        import copy

        from torch_geometric.data import Batch

        g1 = build_inference_graph(pdb_6eey, encoder_type="gvp")  # no ligands
        g2 = build_inference_graph(
            pdb_4h0b, encoder_type="gvp", include_ligands=True
        )  # ligands

        batch = Batch.from_data_list([copy.deepcopy(g1), copy.deepcopy(g2)])

        assert batch.num_graphs == 2
        # nodes concatenate and carry a per-graph index
        assert (
            batch["protein"].num_nodes
            == g1["protein"].num_nodes + g2["protein"].num_nodes
        )
        assert batch["protein"].batch.unique().tolist() == [0, 1]
        # PP edges concatenate (offset per graph)
        assert (
            batch[EDGE_PP].edge_index.shape[1]
            == g1[EDGE_PP].edge_index.shape[1] + g2[EDGE_PP].edge_index.shape[1]
        )
        # per-graph masks survive the merge
        assert int(batch["protein"].is_ligand.sum()) == int(
            g2["protein"].is_ligand.sum()
        )


@pytest.mark.unit
class TestPersistence:
    def test_out_dir_writes_graph(self, pdb_6eey, tmp_path):
        out = tmp_path / "processed_inf"
        data = build_inference_graph(pdb_6eey, encoder_type="gvp", out_dir=str(out))
        saved = out / "6eey_final.pt"
        assert saved.exists()
        reloaded = torch.load(saved, weights_only=False)
        assert reloaded["protein"].num_nodes == data["protein"].num_nodes
