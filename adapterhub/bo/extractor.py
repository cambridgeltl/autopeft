from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Dict, Any

import torch


def get_extractor(extractor_type: str = "wl", options: Optional[Dict[str, Any]] = None):
    if extractor_type == "wl":
        hps = {
            "h": 1,
        }
        hps.update(options or {})
        return WeisfeilerLehmanExtractor(**hps)
    else:
        raise NotImplementedError(f"Extractor type {extractor_type} is not implemented!")


class WeisfilerLehmanConv(torch.nn.Module):
    def __init__(self,
                 no_conv: bool = False,
                 store_feature_id: bool = False,
                 exclude_isolate: bool = True):
        super(WeisfilerLehmanConv, self).__init__()
        self.store_feature_id = store_feature_id
        self.no_conv = no_conv
        self.exclude_isolate = exclude_isolate
        self.reset_hashmap()

    def reset_hashmap(self):
        if self.exclude_isolate:
            self.hashmap = OrderedDict({"isolate": 0})
            self.hashmap_eval = OrderedDict({"isolate": 0})
        else:
            self.hashmap = OrderedDict({})
            self.hashmap_eval = OrderedDict({})
        self.feature_map = {}

    def convolve(self, x: torch.Tensor, adj_t: torch.Tensor, pass_message: bool = True):
        self.hashmap_eval = deepcopy(self.hashmap)
        out = []
        nonzero_idx = torch.nonzero(adj_t)
        col = nonzero_idx[:, 1]
        deg = adj_t.sum(dim=1).int().tolist()
        for i, (node, neighbors) in enumerate(zip(x.tolist(), x[col].split(deg))):
            if adj_t[i, :].sum() + adj_t[:, i].sum() == 0 and self.exclude_isolate:
                idx = "isolate"
            else:
                if pass_message:
                    idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
                else:
                    idx = hash(tuple([node]))
            if self.store_feature_id:
                self.feature_map[idx] = (
                    tuple([node] + neighbors.sort()[0].tolist())
                    if pass_message
                    else (node, )
                )
            if idx not in self.hashmap_eval:
                self.hashmap_eval[idx] = len(self.hashmap_eval)
                if self.training:
                    self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap_eval[idx])
        return torch.tensor(out, device=x.device)

    def histogram(self,
                  x: torch.Tensor,
                  batch: Optional[torch.Tensor] = None,
                  norm: bool = True,
                ):
        num_colors = len(self.hashmap_eval)
        batch_size = int(batch.max()) + 1
        index = batch * num_colors + x
        out = torch.zeros(num_colors * batch_size, dtype=index.dtype).scatter_add_(0, index, torch.ones_like(index))
        out = out.view(batch_size, num_colors)
        if not self.training:
            out = out[:, :len(self.hashmap)]
        if self.exclude_isolate:
            out = out[:, 1:]
        if norm:
            out = out.to(torch.double)
            out /= out.norm(dim=-1, keepdim=True)
        return out

    def forward(self,
                x: torch.Tensor,
                adj_t: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                norm: bool = True,
                return_embedding: bool = False,
                ):
        if not self.training:
            assert len(self.hashmap_eval)
        x = x.long().flatten()
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.convolve(x, adj_t=adj_t, pass_message= not self.no_conv)
        output = self.histogram(x, batch=batch, norm=norm,)
        if return_embedding:
            return output, x
        return x


class WeisfeilerLehmanExtractor(torch.nn.Module):
    def __init__(self, h: int = 1) -> None:
        super(WeisfeilerLehmanExtractor, self).__init__()
        assert  h >= 0
        self.h = h
        self.wl_convs = [WeisfilerLehmanConv(no_conv=True, exclude_isolate=False)] \
                        + [WeisfilerLehmanConv(no_conv=False) for _ in range(self.h)]

    def get_num_features(self) -> int:
        n = 0
        for wl in self.wl_convs:
            n += len(wl.hashmap)
        return n

    def reset_hashmap(self):
        for i in range(len(self.wl_convs)):
            self.wl_convs[i].reset_hashmap()

    def train(self, *args, **kwargs):
        for wl in self.wl_convs:
            wl.training = True

    def eval(self, *args, **kwargs):
        for wl in self.wl_convs:
            wl.training = False

    @torch.no_grad()
    def forward(self,
                x: torch.Tensor,
                adj_t: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                norm: bool = False) -> torch.Tensor:
        outputs = []
        for wl in self.wl_convs:
            output, x = wl(x, adj_t, norm=norm, batch=batch, return_embedding=True)
            outputs.append(output)
        if len(outputs) > 1:
            output = torch.cat(outputs, dim=-1)
        else:
            output = outputs[0]
        return output