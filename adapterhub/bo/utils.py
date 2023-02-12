from typing import Optional, List, Tuple

import botorch.models
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Any

from typing import List, Optional, Tuple

import torch
from statsmodels.distributions.empirical_distribution import ECDF
from torch import Tensor
from torch.distributions import Normal


def apply_normal_copula_transform(
    Y: Tensor, ecdfs: Optional[List[ECDF]] = None
) -> Tuple[Tensor, List[ECDF]]:
    r"""Apply a copula transform independently to each output.
    Values are first mapped to quantiles through the empirical cdf, then
    through an inverse standard normal cdf.
    Note: this is not currently differentiable and it does not support
    batched `Y`.
    TODO: Remove dependency on ECDF or at least write an abstract specification
    of what we expect ECDF to do.
    Args:
        Y: A `n x m`-dim tensor of values
        ecdfs: A list of ecdfs to use in the transformation
    Returns:
        2-element tuple containing
        - A `n x m`-dim tensor of transformed values
        - A list of `m` ECDF objects.
    """
    if Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")
    normal = Normal(0, 1)
    Y_i_tfs = []
    ecdfs = ecdfs or []
    for i in range(Y.shape[-1]):
        Y_i = Y[:, i].cpu().numpy()
        if len(ecdfs) <= i:
            # compute new ecdf if None were provided
            ecdf = ECDF(Y_i)
            ecdfs.append(ecdf)
        else:
            # Otherwise use existing ecdf
            ecdf = ecdfs[i]
        # clamp quantiles here to avoid (-)infs at the extremes
        Y_i_tf = normal.icdf(torch.from_numpy(
            ecdf(Y_i)).to(Y).clamp(0.0001, 0.9999))
        Y_i_tfs.append(Y_i_tf)
    return torch.stack(Y_i_tfs, dim=-1), ecdfs


def collate_graphs_adj(
    Xs: torch.Tensor,
    adj_mats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function collate a batch of graphs into a large graph with disjointed components.
    Args:
        Xs: A `n x V x d`-dim tensor where `n` is the number of graphs, `v` is the number of vertices in the largest graph,
         and `d` is dimensionality of node features.
        adj_mats: shape: 'n x V x V'

        Note: this assume that the all graphs in the batch (including both the feature and adjacency matrices) are
        all zero-padded to ensure that the adjacency matrices are of the the same shape!
    Outputs:
        Xs_collated: A `(n * V) x d'-dim tensor where X of individual graphs have been collated together
        adj_mat_collated: A '(n * V) x (n * V)'-dim tensor, where the adj_mat of individual graphs are stacked to a block-diagonal tensor.
        batch: a '(n x V)'- dim tensor. Consider an input of 3 graphs with 4 nodes each, the batch will be
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    """
    assert (
        adj_mats.shape[-1] == adj_mats.shape[-2]
    ), "each element in adj_mat must be a square tensor!"
    assert (
        adj_mats.shape[:2] == Xs.shape[:2]
    ), "mismatch in shape between Xs and adj_mats"
    n, v, d = Xs.shape
    batch = torch.arange(n).repeat_interleave(v)
    Xs_collated = Xs.reshape(n * v, d)
    adj_mats_collated = torch.block_diag(*list(adj_mats))
    return Xs_collated, adj_mats_collated, batch


def parse_graphs(
    adj_mats: Optional[List[torch.Tensor]] = None,
    edge_list: Optional[List[torch.Tensor]] = None,
    Xs: Optional[List[torch.Tensor]] = None,
    node_degree_as_feature: bool = False,
    pad_size: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse a list of graphs of potentially different sizes into tensors.
    adj_mats: list of adjacency matrices. shape = n tensors of shape `|V|_i x |V|_i`, where `|V|_i` is the number of vertices
        of the i-th graph
    edge_list: list of edge pairs. shape = n tensors of shape `|E|_i x 2`, where `|E|_i` is the number of edges of the i-th graph.
        Note that either adj_mats or edge_list must be supplied. An error is raised if neither or both are provided.
    Xs: list of Tensors. shape = `n x |V|_i`, d where `|V|_i` is the number of vertices of the i-th graph and d is the
        dimensionality of feature matrix.
        This must be provided unless node_degree_as_feature is True.
    node_degree_as_feature: bool. When Xs is not provided and this flag is True, we will use the node degree as the
        node feature.
    pad_size: int. The adjacency and feature matrix of each input graph are zero_padded to this size. If not supplied,
        we will use the max(|V|_i)  across all inputs.
    """
    assert (adj_mats is not None) ^ (
        edge_list is not None
    ), "either adj_mats or edge_lists must be supplied!"
    if not node_degree_as_feature:
        assert (
            Xs is not None
        ), "feature matrices must be provided unless node_degree_as_feature is True"
    if Xs is not None:
        # will be inferred later if not supplied
        nnodes = [x.shape[-2] for x in Xs]
    elif adj_mats is not None:
        nnodes = [adj_mat.shape[-1] for adj_mat in adj_mats]
    elif edge_list is not None:
        nnodes = [edge.max() + 1 for edge in edge_list]

    pad_size = max(pad_size, max(nnodes)
                   ) if pad_size is not None else max(nnodes)

    adj_mats = adj_mats or edgelist2adj(edge_list, nnodes=nnodes)
    adj_mats = pad_adj_t(adj_mats, size=pad_size, as_tensor=True)

    if node_degree_as_feature and Xs is None:  # use the node degrees as the features
        Xs = [adj_mat.sum(dim=1).reshape(-1, 1) for adj_mat in adj_mats]
    Xs = pad_feature_matrices(Xs, size=pad_size, as_tensor=True)
    return Xs, adj_mats


def edgelist2adj(
    edge_list: List[torch.Tensor],
    nnodes: Optional[List[int]] = None,
    pad_zeros: bool = False,
) -> List[torch.Tensor]:
    """
    Convert edge_list representation of graphs to adjacency matrices. Note that this method generally supports edge_list of tensors with different lengths
    Args:
        edge_list: `n` tensors of shape `2 x |E|_i`  where |E|_i is the the number of edges of the i-th graph
        nnodes: list of int of shape `n`. If None, nnodes will be inferred from the `edge_list` where the number of nodes of
            each graph is taken by the node index in the edge list. This may not be accurate especially if there exist
            isolates in the graphs.
        pad_zeros: bool. If True, pad zeros in the adjacency matrix to the size of the largest graph in the batch.

        Note that this method does NOT assume we have an undirected graph with symmetrical adjacency matrix. To enforce this,
            you need to add edges in both directions in terms of node indices (e.g. [1, 2] and [2, 1])
    Output:
        output: list of `n` Tensors. Each tensor is of shape `|V|_i x |V|_i` where |V|_i is the number of nodes of the i-th graph if pad_zero is False.
            Otherwise each tensor is of shape `|V|_max` where |V|_max is the maximum number of nodes in the list of graph supplied.

    """
    if nnodes is not None:
        assert len(nnodes) == len(
            edge_list
        ), "the length of edge_list and nnodes must match!"
    output = []
    nnodes = nnodes or [edge.max() + 1 for edge in edge_list]
    max_nodes = max(nnodes)
    for i, edges in enumerate(edge_list):
        nnode = nnodes[i] if nnodes is not None else edges.max() + 1
        adj_t = (
            torch.sparse_coo_tensor(
                edges,
                values=torch.ones(edges.shape[1]),
                size=(max_nodes, max_nodes) if pad_zeros else (nnode, nnode),
            )
            .to_dense()
            .to(edges.device)
        )
        output.append(adj_t)
    return output


def pad_feature_matrices(
    Xs: List[torch.Tensor], size: Optional[int] = None, as_tensor: bool = False
):
    """
    Pad a list of feature matrices with trailing zeros where applicable
    Args:
        Xs: List of tensors. Each of shape `|V|_i x d`. `|V|_i`is the number of vertices of the i-th graph and is different
            in `Xs` in general; `d` is the dimensionality of the feature matrix and should be the same across all inputs.
        size: the size of the graph that each individual graph should be padded to. If not supplied, this will be the
            number of vertices of the largest graph in `Xs`.
        as_tensor: whether to return the output as a tensor of shape `(n * |V|_{max}) x d`. Otherwise return a list of length `d`.
    Output:
        if `as_tensor` is `True`:
            torch.Tensor of shape `(n * |V|_{max}) x d`. See `as_tensor` argument
        else:
            list of n tensors, each of shape `|V|_{max} x d`
    """
    if size is None:
        size = max([x.shape[-2] for x in Xs])
    output = []
    for x in Xs:
        x_padded = F.pad(x, pad=(0, 0, 0, size - x.shape[-2]))
        output.append(x_padded)
    if as_tensor:
        return torch.stack(output).to(Xs[0].device)
    return output


def pad_adj_t(
    adj_ts: List[torch.Tensor], size: Optional[int] = None, as_tensor: bool = False
):
    """
    Pad a list of adjacency tensors with trailing zeros where applicable.
    Args:
    adj_ts: List of tensors. Each of shape `|V|_i x |V|_i`. `|V|_i`is the number of vertices of the i-th graph and is different
        in `Xs` in general;
    size: the size of the graph that each individual graph should be padded to. If not supplied, this will be the
        number of vertices of the largest graph in adj_ts.
    as_tensor: whether to return the output as a tensor of shape `n x |V|_{max}`, d. Otherwise return a list.
    Output:
        if as_tensor is True:
            torch.Tensor of shape `(n * |V|_{max}) x |V|_{max}`. See as_tensor argument
        else:
            list of `n` tensors, each of shape `|V|_{max} x |V|_{max}`
    """
    if size is None:
        size = max([adj_t.shape[-1] for adj_t in adj_ts])
    output = []
    for adj_t in adj_ts:
        adj_t_padded = F.pad(
            adj_t, pad=(0, size - adj_t.shape[-1], 0, size - adj_t.shape[-1])
        )
        output.append(adj_t_padded)
    if as_tensor:
        return torch.stack(output).to(adj_ts[0].device)
    return output


def filter_invalid(X: torch.Tensor, X_avoid: torch.Tensor, return_index: bool = False):
    """Remove all occurences of `X_avoid` from `X`."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X_avoid.ndim == 1:
        X_avoid = X_avoid.reshape(-1, 1)
    idx = ~(X == X_avoid.unsqueeze(-2)).all(dim=-1).any(dim=-2)
    ret = X[idx]
    if X.ndim == 1:
        ret = ret.squeeze(1)
    if return_index:
        return ret, idx.nonzero().squeeze(-1).tolist()
    return ret
