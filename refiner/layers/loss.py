from typing import Dict, List, Tuple, Union, Optional, Callable
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import resize

from refiner.utils.config_types import LossCfg
from refiner.utils.named_tensor import NamedTensor
from refiner.layers.hog import HOGLayer

try:
    from refiner.layers.ma import get_ma_encoder
except:
    get_ma_encoder = None

def gom(out: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is not None:
        out, target = out * mask, target * mask
    scal = torch.abs((out * target)).sum()
    nout = torch.linalg.norm(out, dim=0, keepdim=True)  # 1 x H x W
    ntarget = torch.linalg.norm(target, dim=0, keepdim=True)
    norm = (nout * ntarget).sum()
    return -scal / norm


class ComparisonLoss(nn.Module, ABC):
    def __init__(
        self,
        comps: List[Tuple[Union[str, int]]],
        target: Union[NamedTensor, None] = None,
        name: str = "",
    ):
        super(ComparisonLoss, self).__init__()
        self.comps = [
            [comp["comp"][0], comp["comp"][1], comp["weight"]] for comp in comps
        ]
        self.channel_weights = sum([comp[2] for comp in self.comps])
        self.target_channels = list(set([c[1] for c in self.comps]))
        self.out_channels = list(set([c[0] for c in self.comps]))
        self.target = target.subtensor(self.target_channels)
        self.name = name

    @property
    @abstractmethod
    def names(self) -> List[str]:
        pass

    @abstractmethod
    def forward(
        self,
        out: NamedTensor,
        target: Optional[NamedTensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        pass


class GradLoss(ComparisonLoss):
    def __init__(
        self,
        comps: List[Tuple[Union[str, int]]],
        target: Union[NamedTensor, None] = None,
        name: str = "",
    ):
        super(GradLoss, self).__init__(comps, target, name)
        self.loss = gom
        mat = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            device=self.target.tensor.device,
            dtype=self.target.tensor.dtype,
        )
        self.padding = 1

        mat = torch.tensor(
            [[ 5,   8,  10,   8,  5], 
             [ 8,  20,  40,  20,  8],
             [ 0,   0,   0,   0,  0], 
             [-8, -20, -40, -20, -8], 
             [-5,  -8, -10,  -8, -5]], 
            device=self.target.tensor.device,
            dtype=self.target.tensor.dtype,
        ).T 
        self.padding = 2
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)  # 2 x 3 x 3
        self.weight = mat.unsqueeze(1)
        self.target = self._compute_gradients(self.target, self.target_channels)

    def _compute_gradients(self, out: NamedTensor, channels: List[str]) -> NamedTensor:
        out.channel_dim = 0  # C x H x W
        out = out[channels].unsqueeze(1)  # C x 1 x H x W
        out = F.conv2d(out, self.weight, padding=self.padding)  # C x 2 x H x W
        out[:, 0, :, 0], out[:, 0, :, -1] = 0, 0
        out[:, 1, 0, :], out[:, 1, -1, :] = 0, 0
        out = NamedTensor(out, channels, 0)
        return out

    @property
    def names(self) -> List[str]:
        return [f"{self.name}_{i}_{c[0]}_{c[1]}" for i, c in enumerate(self.comps)]

    def forward(
        self, out: NamedTensor, target: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        losses = []
        out = self._compute_gradients(out, self.out_channels)
        if target is None:
            target = self.target
        for c in self.comps:
            losses.append(
                c[2] / self.channel_weights * self.loss(out[c[0]], target[c[1]], mask)
            )
        return losses

def mseloss(out: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is not None:
        out, target = out * mask, target * mask
    l = torch.linalg.norm((out - mask) / out.shape[-1], dim = -1)

    if mask is not None:
        l = l.sum() / mask.sum()
    else: 
        l = l.mean()
    return l
class HOGLoss(ComparisonLoss):
    def __init__(
        self,
        hogs: List[HOGLayer],
        comps: List[Tuple[Union[str, int]]],
        target: Union[NamedTensor, None] = None,
        name: str = "",
    ):
        super(HOGLoss, self).__init__(comps, target, name)
        self.hogs = hogs
        self.targets = self._compute_features(self.target, self.target_channels)
        # self.loss = nn.MSELoss(reduction="mean")
        self.loss = mseloss
        self.channel_weights *= len(self.hogs)

    def _compute_features(
        self, out: NamedTensor, channels: List[str]
    ) -> List[NamedTensor]:
        features = []
        base_channel_dim = out.channel_dim
        out.channel_dim = 0
        for h in self.hogs:
            feat = h(out[channels])  # C x Hb x Wb x cpby x cpbx x B
            feat = NamedTensor(feat, out.names, channel_dim=0)
            features.append(feat)
        out.channel_dim = base_channel_dim
        return features

    @property
    def names(self):
        return [
            f"{self.name}_{i}_{c[0]}_{c[1]}"
            for i in range(len(self.hogs))
            for c in self.comps
        ]

    def forward(self, out: NamedTensor, target: Optional[NamedTensor] = None, mask: Optional[torch.Tensor] = None):
        losses = []
        out = out.subtensor(self.out_channels)
        out.channel_dim = 0
        targets = self.targets

        feats = self._compute_features(out, self.out_channels)
        masks = [h.forward_mask(mask) for h in self.hogs]
        for o, t, m in zip(feats, targets, masks):
            for c in self.comps:
                l = self.loss(o[c[0]], t[c[1]], m)
                losses.append(c[2] / self.channel_weights * l)

        return losses


class PhotoLoss(ComparisonLoss):
    def __init__(
        self,
        comps: List[Tuple[Union[str, int]]],
        target: Union[NamedTensor, None] = None,
        name: str = "",
    ):
        super(PhotoLoss, self).__init__(comps, target, name)
        self.loss = nn.MSELoss(reduction="mean")

    @property
    def names(self):
        return [f"{self.name}_{i}_{c[0]}_{c[1]}" for i, c in enumerate(self.comps)]

    def forward(
        self,
        out: NamedTensor,
        target: Union[NamedTensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> List[torch.Tensor]:
        losses = []
        if target is None:
            target = self.target
        for c in self.comps:
            losses.append(
                c[2] / self.channel_weights * self.loss(out[c[0]], target[c[1]])
            )
        return losses

class FeatureLoss(ComparisonLoss):
    def __init__(self, get_featurizer: Callable, comps: List[Tuple[Union[str, int]]], target: Union[NamedTensor, None] = None, sizes = [1, 2, 4, 8, 16], name: str = ''):
        super(FeatureLoss, self).__init__(comps, target, name)
        self.encoder = get_featurizer()
        self.targets = self._compute_features(self.target, self.target_channels, True)
        self.sizes = sizes

    def _compute_features(self, out: NamedTensor, channels: List[str], detach: bool = False) -> List[NamedTensor]:
        features = {}
        base_channel_dim = out.channel_dim
        out.channel_dim = 0
        out = out[channels].unsqueeze(1).repeat(1, 3, 1, 1)
        out = resize(out, (336, 336))

        features = self.encoder(out) # B x 1 x H x W 
        for s in features:
            if detach:
                features[s] = features[s].detach()
            features[s] = NamedTensor(features[s], channels, 0) # B x D x H x W
        out.channel_dim = base_channel_dim
        return features

    @property
    def names(self):
        return [f"{self.name}_{i}_{c[0]}_{c[1]}" for c in self.comps for i in self.sizes]

    def forward(self, out: NamedTensor, target: Union[NamedTensor, None] = None):
        losses = []
        out = out.subtensor(self.out_channels)
        out.channel_dim = 0
        targets = self._compute_features(target, self.target_channels) if target is not None else self.targets
        features = self._compute_features(out, self.out_channels)

        for s in self.sizes:
            fs, ts = features[s], targets[s]
            for c in self.comps:
                fc, tc = fs[c[0]], ts[c[1]]
                H, W = fc.shape[1], fc.shape[2]
                l = torch.linalg.norm((fc-tc), dim = 0) / math.sqrt(H*W)
                l = torch.mean(l)
                losses.append(c[2] / self.channel_weights * l)

        return losses
class MultiLoss:
    def __init__(self, img: NamedTensor, cfg: LossCfg):
        self.losses: List[ComparisonLoss] = []
        self.weights: List[float] = []
        device = img.tensor.device

        if cfg["hogs"] and cfg["hogs"]["weight"] > 0:
            hogs = [
                HOGLayer(**c, device=device)
                for c in cfg["hogs"]["layers"]
            ]
            loss = HOGLoss(hogs, cfg["hogs"]["channels"], img, "hog")
            self.losses.append(loss)
            self.weights.append(cfg["hogs"]["weight"])

        if cfg["photo"] and cfg["photo"]["weight"] > 0:
            loss = PhotoLoss(cfg["photo"]["channels"], img, "photo")
            self.losses.append(loss)
            self.weights.append(cfg["photo"]["weight"])

        if cfg["grad"] and cfg["grad"]["weight"] > 0:
            loss = GradLoss(cfg["grad"]["channels"], img, "grad")
            self.losses.append(loss)
            self.weights.append(cfg["grad"]["weight"])

        if cfg["ma"] and cfg["ma"]["weight"] > 0:
            loss = FeatureLoss(get_ma_encoder, cfg["ma"]["channels"], img, name = "ma")
            self.losses.append(loss)
            self.weights.append(cfg["ma"]["weight"])

        s = sum(self.weights)
        self.weights = [w / s for w in self.weights]

    def forward(self, img: NamedTensor, target: Optional[NamedTensor], mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        loss = 0.0
        losses = {}
        for l, w in zip(self.losses, self.weights):
            ls = l(img, mask = mask)
            ns = l.names
            for li, ni in zip(ls, ns):
                loss += w * li
                losses[ni] = li
        losses["loss"] = loss
        return losses

    @property
    def names(self) -> List[str]:
        names = []
        for l in self.losses:
            names += l.names
        return names

    @property
    def channels(self) -> List[str]:
        channels = []
        for l in self.losses:
            channels += l.out_channels
        channels = list(set(channels))
        channels.sort()
        return channels

    @property
    def target_channels(self) -> List[str]:
        target_channels = []
        for l in self.losses:
            target_channels += l.target_channels
        target_channels = list(set(target_channels))
        target_channels.sort()
        return target_channels
