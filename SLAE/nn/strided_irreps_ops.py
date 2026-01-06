

import torch
from typing import List, Optional, Tuple, NamedTuple
from math import ceil, sqrt

import torch
from torch import fx
import math

from e3nn.util.jit import compile_mode
from e3nn import o3
from e3nn.util.jit import compile
from e3nn.util import prod
from e3nn.o3 import Irreps
from e3nn.o3 import Instruction
from e3nn.o3._tensor_product._codegen import _sum_tensors

from opt_einsum_fx import jitable, optimize_einsums_full
from SLAE.nn.spmm import ExplicitGradSpmm







class StridedLayout:
    """Utility class to represent a strided layout of a tensor whose irreps all have the same multiplicity."""

    irreps: Irreps
    base_irreps: Irreps
    pad_to_multiple: int
    dim: int
    base_dim: int
    mul: int

    def __init__(self, irreps: Irreps, pad_to_multiple: int = 1):
        irreps = Irreps(irreps)
        if not self.can_be_strided(irreps):
            raise ValueError(f"Irreps `{irreps}` cannot be strided.")
        self.irreps = irreps
        self.base_irreps = Irreps([(1, ir) for _, ir in irreps])
        self.mul = self.irreps[0].mul if len(irreps) > 0 else 0
        assert self.irreps.dim == self.base_irreps.dim * self.mul
        self.pad_to_multiple = pad_to_multiple
        assert self.pad_to_multiple in (1, 2, 4, 8)

        self.base_dim = int(
            math.ceil(self.base_irreps.dim / self.pad_to_multiple)
            * self.pad_to_multiple
        )
        pad_by = self.base_dim - self.base_irreps.dim
        self.dim = self.base_dim * self.mul

        # indexes to convert
        self.indexes_to_strided = torch.zeros(self.dim, dtype=torch.long)
        self.indexes_to_catted = torch.zeros(self.irreps.dim, dtype=torch.long)
        i: int = 0
        for mul_i in range(self.mul):
            for irrep_i, (_, irrep) in enumerate(self.base_irreps):
                strided_indexes = torch.arange(start=i, end=i + irrep.dim)
                catted_indexes = (
                    torch.arange(irrep.dim)
                    + self.irreps[:irrep_i].dim
                    + irrep.dim * mul_i
                )
                self.indexes_to_strided[strided_indexes] = catted_indexes
                self.indexes_to_catted[catted_indexes] = strided_indexes
                i += irrep.dim
            # pad out this line of the [mul, k] shape
            i += pad_by

        # They should be inverses:
        assert torch.all(
            self.indexes_to_strided[self.indexes_to_catted]
            == torch.arange(self.irreps.dim)
        )

    @staticmethod
    def can_be_strided(irreps: Irreps) -> bool:
        """Check whether ``irreps`` is compatible with strided layout."""
        irreps = Irreps(irreps)
        if len(irreps) == 0:
            return True
        return all(irreps[0].mul == mul for mul, ir in irreps)

    def to_strided(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a tensor from default to strided layout."""
        return x[..., self.indexes_to_strided]

    def to_catted(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a tensor from strided to default layout."""
        return x[..., self.indexes_to_catted]



@compile_mode("script")
class MakeWeightedChannels(torch.nn.Module):
    weight_numel: int
    multiplicity_out: int
    _num_irreps: int

    def __init__(
        self,
        irreps_in,
        multiplicity_out: int,
        pad_to_alignment: int = 1,
    ):
        super().__init__()
        assert all(mul == 1 for mul, ir in irreps_in)
        assert multiplicity_out >= 1
        # Each edgewise output multiplicity is a per-irrep weighted sum over the input
        # So we need to apply the weight for the ith irrep to all DOF in that irrep
        w_index = sum(([i] * ir.dim for i, (mul, ir) in enumerate(irreps_in)), [])
        # pad to padded length
        n_pad = (
            int(ceil(irreps_in.dim / pad_to_alignment)) * pad_to_alignment
            - irreps_in.dim
        )
        # use the last weight, what we use doesn't matter much
        w_index += [w_index[-1]] * n_pad
        self._num_irreps = len(irreps_in)
        self.register_buffer("_w_index", torch.as_tensor(w_index, dtype=torch.long))
        # there is
        self.multiplicity_out = multiplicity_out
        self.weight_numel = len(irreps_in) * multiplicity_out

    def forward(self, edge_attr, weights):
        # weights are [z, u, i]
        # edge_attr are [z, i]
        # i runs over all irreps, which is why the weights need
        # to be indexed in order to go from [num_i] to [i]
        return torch.einsum(
            "zi,zui->zui",
            edge_attr,
            weights.view(
                -1,
                self.multiplicity_out,
                self._num_irreps,
            )[:, :, self._w_index],
        )


def codegen_strided_tensor_product_forward(
    irreps_in1: o3.Irreps,
    in1_var: List[float],
    irreps_in2: o3.Irreps,
    in2_var: List[float],
    irreps_out: o3.Irreps,
    out_var: List[float],
    instructions: List[Instruction],
    normalization: str = "component",
    shared_weights: bool = False,
    specialized_code: bool = True,
    sparse_mode: Optional[str] = None,
    pad_to_alignment: int = 1,
) -> Optional[fx.GraphModule]:
    """Returns None if strided doesn't make sense for this TP."""
    # TODO padding
    # Check if irreps can be strided
    try:
        layout_in1 = StridedLayout(irreps_in1, pad_to_multiple=pad_to_alignment)
        layout_in2 = StridedLayout(irreps_in2, pad_to_multiple=pad_to_alignment)
        layout_out = StridedLayout(irreps_out, pad_to_multiple=pad_to_alignment)
    except ValueError:
        # one cannot be strided
        return None

    # check the instructions
    assert specialized_code

    connection_mode = instructions[0].connection_mode
    if not all(ins.connection_mode == connection_mode for ins in instructions):
        return None

    has_weight = instructions[0].has_weight
    if not all(ins.has_weight == has_weight for ins in instructions):
        return None
    if not has_weight:
        assert connection_mode == "uuu"  # for now

    # TODO: sort insturctions?

    # Make the big w3j
    w3j_index = []
    w3j_values = []

    for ins_i, ins in enumerate(instructions):
        mul_ir_in1 = layout_in1.base_irreps[ins.i_in1]
        mul_ir_in2 = layout_in2.base_irreps[ins.i_in2]
        mul_ir_out = layout_out.base_irreps[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert (
            abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
            <= mul_ir_out.ir.l
            <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        )

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            raise ValueError

        this_w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        this_w3j_index = this_w3j.nonzero()
        w3j_values.append(
            this_w3j[this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]]
        )

        # Normalize the path through its w3j entries
        # TODO: path_weight
        # TODO: in and out var
        if normalization == "component":
            w3j_norm_term = 2 * mul_ir_out.ir.l + 1
        if normalization == "norm":
            w3j_norm_term = (2 * mul_ir_in1.ir.l + 1) * (2 * mul_ir_in2.ir.l + 1)
        alpha = sqrt(
            ins.path_weight  # per-path weight
            * out_var[ins.i_out]  # enforce output variance
            * w3j_norm_term
            / sum(
                in1_var[i.i_in1]
                * in2_var[i.i_in2]
                * {
                    "uvw": (layout_in1.mul * layout_in2.mul),
                    "uvu": layout_in2.mul,
                    "uvv": layout_in1.mul,
                    "uuw": layout_in1.mul,
                    "uuu": 1,
                    "uvuv": 1,
                }[connection_mode]
                for i in instructions
                if i.i_out == ins.i_out
            )
        )
        w3j_values[-1].mul_(alpha)

        this_w3j_index[:, 0] += layout_in1.base_irreps[: ins.i_in1].dim
        this_w3j_index[:, 1] += layout_in2.base_irreps[: ins.i_in2].dim
        this_w3j_index[:, 2] += layout_out.base_irreps[: ins.i_out].dim
        # Now need to flatten the index to be for [pk][ij]
        w3j_index.append(
            torch.cat(
                (
                    (ins_i if ins.has_weight else 0)  # unweighted all go in first path
                    * layout_out.base_dim
                    + this_w3j_index[:, 2].unsqueeze(-1),
                    this_w3j_index[:, 0].unsqueeze(-1) * layout_in2.base_dim
                    + this_w3j_index[:, 1].unsqueeze(-1),
                ),
                dim=1,
            )
        )

    num_paths: int = len(instructions) if has_weight else 1

    w3j = torch.sparse_coo_tensor(
        indices=torch.cat(w3j_index, dim=0).t(),
        values=torch.cat(w3j_values, dim=0),
        size=(
            num_paths * layout_out.base_dim,
            layout_in1.base_dim * layout_in2.base_dim,
        ),
    ).coalesce()

    # w3j is k,i,j, so this is whether, for nonzero entries,
    # the i index is always equal to the j index. If so, then
    # it is diagonal and we can eliminate the j dimension
    # in this case we are only taking diagonal (i == j)
    # entries from the outer product; but those values are just
    # the direct multiplication of the two tensors, eliminating
    # the need for the outer product.
    # obviously this only makes sense if they have the same size as well
    # this is more or less a test of whether this TP is an inner product
    w3j_i_indexes = torch.div(
        w3j.indices()[1], layout_in1.base_dim, rounding_mode="floor"
    )
    w3j_j_indexes = w3j.indices()[1] % layout_in1.base_dim
    w3j_is_ij_diagonal = (layout_in1.base_dim == layout_in2.base_dim) and torch.all(
        w3j_i_indexes == w3j_j_indexes
    )
    if w3j_is_ij_diagonal:
        # change the w3j to eliminate the dimension
        # now its just k,i
        w3j = torch.sparse_coo_tensor(
            indices=torch.stack((w3j.indices()[0], w3j_i_indexes)),
            values=w3j.values(),
            size=(
                num_paths * layout_out.base_dim,
                layout_in1.base_dim,
            ),
        )

    # TODO: support use of sparse w3j
    if sparse_mode is None:
        # in dense, must shape it for einsum:
        if w3j_is_ij_diagonal:
            kij_shape = (
                layout_out.base_dim,
                layout_in1.base_dim,
            )
        else:
            kij_shape = (
                layout_out.base_dim,
                layout_in1.base_dim,
                layout_in2.base_dim,
            )
        w3j = (
            w3j.to_dense()
            .reshape(((num_paths,) if num_paths > 1 else tuple()) + kij_shape)
            .contiguous()
        )
        del kij_shape
    elif sparse_mode == "coo":
        w3j = w3j.coalesce()
    elif sparse_mode == "csr":
        w3j = w3j.coalesce().to_sparse_csr()
    else:
        raise ValueError

    # Generate the mixer
    u, v, w = connection_mode
    uv = {"uv": "uv", "uu": "u"}[connection_mode[:2]]
    if has_weight:
        weight_label = {"uvw": "uvw", "uuu": "u", "uvv": "uv"}[connection_mode]

        z = "" if shared_weights else "z"

        weight_shape = {
            "uvw": (layout_in1.mul, layout_in2.mul, layout_out.mul),
            "uuu": (layout_in1.mul,),
            "uvv": (layout_in1.mul, layout_in2.mul),
        }[connection_mode]
        if num_paths > 1:
            # ^ if there's only one weighted path, the einsum simplifies without the p dimension
            weight_label = weight_label + "p"
            weight_shape = weight_shape + (num_paths,)
        if not shared_weights:
            weight_shape = (-1,) + weight_shape
    else:
        weight_shape = tuple()

    # generate actual code
    graph_out = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph_out)

    def Proxy(n):
        return fx.Proxy(n, tracer=tracer)

    # = Function definitions =
    x1s_out = Proxy(graph_out.placeholder("x1", torch.Tensor))
    x2s_out = Proxy(graph_out.placeholder("x2", torch.Tensor))
    if has_weight:
        ws_out = Proxy(graph_out.placeholder("w", torch.Tensor))
        ws_out = ws_out.reshape(weight_shape)

    if sparse_mode is None:
        w3j_proxy = Proxy(graph_out.get_attr("_big_w3j"))

    # convert to strided
    x1s_out = x1s_out.reshape(-1, layout_in1.mul, layout_in1.base_dim)
    x2s_out = x2s_out.reshape(-1, layout_in2.mul, layout_in2.base_dim)

    # do the einsum
    # has shape zwk
    j = "i" if w3j_is_ij_diagonal else "j"
    ij = "i" if w3j_is_ij_diagonal else "ij"
    if has_weight:
        if sparse_mode is None:
            # use einsum for the full contract
            einstr = f"{z}{weight_label},z{u}i,z{v}{j},{'p' if num_paths > 1 else ''}k{ij}->z{w}k"
            out = torch.einsum(einstr, ws_out, x1s_out, x2s_out, w3j_proxy)
        else:
            outer = torch.einsum(f"z{u}i,z{v}{j}->z{uv}{ij}", x1s_out, x2s_out)
            # \/ has shape [pk][ij] * [ij][zuv] = [pk][zuv]
            contracted = Proxy(
                graph_out.call_module(
                    "_w3j_mm",
                    (
                        outer.reshape(
                            -1,
                            (
                                layout_in1.base_dim
                                if w3j_is_ij_diagonal
                                else layout_in1.base_dim * layout_in2.base_dim
                            ),
                        ).T.node,
                    ),
                )
            ).T.reshape(
                (-1,)
                + {"uv": (layout_in1.mul, layout_in2.mul), "uu": (layout_in1.mul,)}[
                    connection_mode[:2]
                ]
                + (num_paths, layout_out.base_dim)
            )
            out = torch.einsum(f"z{uv}pk,{z}{weight_label}->z{w}k", contracted, ws_out)
    else:
        if sparse_mode is None:
            # use einsum for the full contract
            einstr = f"z{u}i,z{v}{j},{'p' if num_paths > 1 else ''}k{ij}->z{w}k"
            out = torch.einsum(einstr, x1s_out, x2s_out, w3j_proxy)
        else:
            outer = torch.einsum(f"z{u}i,z{v}{j}->z{uv}{ij}", x1s_out, x2s_out)
            # \/ has shape [k][ij] * [ij][zuv] = [pk][zuv]
            out = Proxy(
                graph_out.call_module(
                    "_w3j_mm",
                    (
                        outer.reshape(
                            -1,
                            (
                                layout_in1.base_dim
                                if w3j_is_ij_diagonal
                                else layout_in1.base_dim * layout_in2.base_dim
                            ),
                        ).T.node,
                    ),
                )
            ).T.reshape(
                (
                    -1,
                    layout_in1.mul,  # its only uuu for now
                    layout_out.base_dim,
                )
            )

    graph_out.output(out.node)

    # check graphs
    graph_out.lint()

    # Make GraphModules
    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    constants_root.register_buffer("_big_w3j", w3j)
    if sparse_mode is not None:
        constants_root._w3j_mm = ExplicitGradSpmm(w3j)
    graphmod_out = fx.GraphModule(constants_root, graph_out, class_name="tp_forward")

    if True:  # optimize_einsums
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, layout_in1.dim)),
            torch.zeros((batchdim, layout_in2.dim)),
            torch.zeros(
                1 if shared_weights else batchdim,
                sum(prod(ins.path_shape) for ins in instructions if ins.has_weight),
            ),
        )
        graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))

    graphmod_out.weight_shape = weight_shape
    graphmod_out._dim_in1 = layout_in1.base_dim
    graphmod_out._dim_in2 = layout_in2.base_dim
    graphmod_out._dim_out = layout_out.base_dim
    graphmod_out._mul_out = layout_out.mul
    graphmod_out.weight_numel = abs(prod(weight_shape))

    return graphmod_out


def Contracter(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions: List[Tuple[int, int, int]],
    has_weight: bool,
    connection_mode: str,
    pad_to_alignment: int = 1,
    shared_weights: bool = False,
    sparse_mode: Optional[str] = None,
):
    irreps_in1 = o3.Irreps(irreps_in1)
    assert all(mul == irreps_in1[0].mul for mul, ir in irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    assert all(mul == irreps_in2[0].mul for mul, ir in irreps_in2)
    irreps_out = o3.Irreps(irreps_out)
    assert all(mul == irreps_out[0].mul for mul, ir in irreps_out)

    mod = codegen_strided_tensor_product_forward(
        irreps_in1,
        [1.0 for _ in irreps_in1],
        irreps_in2,
        [1.0 for _ in irreps_in2],
        irreps_out,
        [1.0 for _ in irreps_out],
        instructions=[
            Instruction(
                i_in1,
                i_in2,
                i_out,
                connection_mode,
                has_weight,
                1.0,
                {
                    "uvw": (
                        irreps_in1[i_in1].mul,
                        irreps_in2[i_in2].mul,
                        irreps_out[i_out].mul,
                    ),
                    "uvu": (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                    "uvv": (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                    "uuw": (irreps_in1[i_in1].mul, irreps_out[i_out].mul),
                    "uuu": (irreps_in1[i_in1].mul,),
                    "uvuv": (
                        irreps_in1[i_in1].mul,
                        irreps_in2[i_in2].mul,
                    ),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out in instructions
        ],
        shared_weights=shared_weights,
        sparse_mode=sparse_mode,
        pad_to_alignment=pad_to_alignment,
    )
    if mod is None:
        raise ValueError("Couldn't use strided for given layout")
    if sparse_mode is None:
        mod = compile(mod)
    return mod



class Irrep_Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple


def codegen_strided_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Irrep_Instruction],
    normalization: str = "component",
    internal_weights: bool = False,
    shared_weights: bool = False,
    pad_to_alignment: int = 1,
) -> Optional[fx.GraphModule]:
    """Returns None if strided doesn't make sense for this TP."""
    # Check if irreps can be strided
    try:
        layout_in = StridedLayout(irreps_in, pad_to_multiple=pad_to_alignment)
        layout_out = StridedLayout(irreps_out, pad_to_multiple=pad_to_alignment)
    except ValueError:
        # one cannot be strided
        return None

    assert normalization == "component"

    if internal_weights:
        assert shared_weights

    # group instructions by output
    ins_per_output: List[List[Irrep_Instruction]] = [
        [ins for ins in instructions if ins.i_out == i]
        for i in range(len(layout_out.base_irreps))
    ]
    ins_group_irrep_slice: List[Tuple[int, int]] = []
    # check that each output is a mix of sequential irreps
    for ins_group in ins_per_output:
        if len(ins_group) == 0:
            ins_group_irrep_slice.append(None)
            continue
        i_ins = set(ins.i_in for ins in ins_group)
        ins_group_irrep_slice.append((min(i_ins), max(i_ins)))
        min_i_in, max_i_in = ins_group_irrep_slice[-1]
        assert i_ins == set(range(min_i_in, 1 + max_i_in))
        assert all(
            layout_in.base_irreps[min_i_in] == layout_in.base_irreps[i]
            for i in range(min_i_in, max_i_in + 1)
        ), "All mixed irreps must be the same"
        assert all(ins.i_out == ins_group[0].i_out for ins in ins_group)

    # TODO: split bad groups into multiple groups

    # generate actual code
    graph_out = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph_out)

    def Proxy(n):
        return fx.Proxy(n, tracer=tracer)

    # = Function definitions =
    x = Proxy(graph_out.placeholder("x", torch.Tensor))
    x = x.reshape(-1, layout_in.mul, layout_in.base_dim)
    if internal_weights:
        ws = Proxy(graph_out.get_attr("w", torch.Tensor))
    else:
        ws = Proxy(graph_out.placeholder("w", torch.Tensor))

    outs = [[] for _ in range(len(layout_out.base_irreps))]

    z = "" if shared_weights else "z"

    w_index: int = 0
    for ins_grp_i, (ins_grp, ins_grp_ins) in enumerate(
        zip(ins_per_output, ins_group_irrep_slice)
    ):
        if len(ins_grp) == 0:
            continue
        # for a given group, which mixes a consecutive set of irreps of the same irrep,
        # we can reduce it to a rectangular operation:
        to_mix = x[
            :,
            :,
            layout_in.base_irreps[: ins_grp_ins[0]]
            .dim : layout_in.base_irreps[: ins_grp_ins[1] + 1]
            .dim,
        ]  # index the i dim in z u i
        # ^ has i index ranging over ins_grp_ins inputs *of same irrep*, so we can rectangularize with a new "n" dimension:
        n: int = 1 + ins_grp_ins[1] - ins_grp_ins[0]
        to_mix = to_mix.reshape(
            -1, layout_in.mul, n, layout_in.base_irreps[ins_grp_ins[0]].ir.dim
        )  # z u n i
        n_weight = layout_in.mul * n * layout_out.mul
        if shared_weights:
            this_w = ws[w_index : w_index + n_weight].reshape(
                layout_out.mul, layout_in.mul, n
            )
        else:
            this_w = ws[:, w_index : w_index + n_weight].reshape(
                -1, layout_out.mul, layout_in.mul, n
            )
        outs[ins_grp[0].i_out].append(torch.einsum(f"{z}vun,zuni->zvi", this_w, to_mix))
        w_index += n_weight

    outs = [
        _sum_tensors(
            o,
            shape=(
                x.shape[0],
                layout_out.mul,
                layout_out.base_irreps[i].dim,
            ),
            like=x,
        )
        for i, (ins_grp, o) in enumerate(zip(ins_per_output, outs))
    ]
    outs = [
        (
            (1.0 / sqrt(sum(layout_in.mul for ins in instructions if ins.i_out == i)))
            * out
        )
        if len(ins_per_output[i]) > 0
        else out
        for i, out in enumerate(outs)
    ]
    if len(outs) > 1:
        out = torch.cat(outs, dim=-1)
    else:
        out = outs[0]

    # pad output
    padding: int = layout_out.base_dim - layout_out.base_irreps.dim
    if padding > 0:
        out = torch.nn.functional.pad(
            out,
            (0, padding),
        )

    graph_out.output(out.node)

    # check graphs
    graph_out.lint()

    # Make GraphModules
    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    if internal_weights:
        constants_root.w = torch.nn.Parameter(torch.randn(w_index))
    graphmod_out = fx.GraphModule(
        constants_root, graph_out, class_name="linear_forward"
    )

    if True:  # optimize_einsums
        # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
        # and still get the right path for _all_ batch dimensions.
        # This is because our einsums are essentially of the form:
        #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
        # In the first case, all but one operands have the batch dimension
        #    => The first contraction gains the batch dimension
        #    => All following contractions have batch dimension
        #    => All possible contraction paths have cost that scales linearly in batch size
        #    => The optimal path is the same for all batch sizes
        # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
        # https://github.com/dgasmith/opt_einsum/issues/158
        # for more details.
        #
        # TODO: consider the impact maximum intermediate result size on this logic
        #         \- this is the `memory_limit` option in opt_einsum
        # TODO: allow user to choose opt_einsum parameters?
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, layout_in.dim)),
            torch.zeros((tuple() if shared_weights else (batchdim,)) + (w_index,)),
        )
        graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))

    graphmod_out.weight_numel = w_index
    graphmod_out.dim_in = layout_in.base_dim

    return graphmod_out


def Linear(
    irreps_in,
    irreps_out,
    shared_weights: Optional[bool] = None,
    internal_weights: bool = False,
    instructions: Optional[List[Tuple[int, int]]] = None,
    pad_to_alignment: int = 1,
):
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    # == Instructions ==
    if instructions is None:
        # By default, make all possible connections
        instructions = [
            (i_in, i_out)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]
        # note that "empty" instructions to/from empty irreps are dealt with in the codegen

    instructions = [
        Irrep_Instruction(i_in=e[0], i_out=e[1], path_shape=None) for e in instructions
    ]

    mod = codegen_strided_linear(
        irreps_in,
        irreps_out,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        pad_to_alignment=pad_to_alignment,
    )

    if mod is None:
        raise ValueError
    return compile(mod)
