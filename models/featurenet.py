from typing import Literal

import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import math
from torchvision import transforms as tvf


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)


    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.contiguous().view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

class GeM(nn.Module):
    def __init__(self, feat_dim, desc_dim, p=3):
        super().__init__()
        self.p = p
        self.whiten = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.LeakyReLU(),
            nn.Linear(desc_dim, desc_dim)
        )

    def forward(self, features):
        mean = (features ** self.p).mean(dim=1)
        return self.whiten(mean.sign() * mean.abs() ** (1 / self.p))


_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
    "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]


class DinoV2ExtractFeatures:


    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int,
                 facet: _DINO_FACETS = "token", use_cls=False,
                 norm_descs=True, device: str = "cpu") -> None:
        self.vit_type: str = dino_model
        self.dino_model: nn.Module = torch.hub.load(
            'facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                register_forward_hook(
                self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                attn.qkv.register_forward_hook(
                self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        return _forward_hook

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img = tvf.CenterCrop((h_new, w_new))(img)
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2 * d_len]
                else:
                    res = res[:, :, 2 * d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None  # Reset the hook
        return res

    def __del__(self):
        self.fh_handle.remove()

class FeatureNet(nn.Module):
    def __init__(self, gd_dim=1024, method="gem", num_cluster=64):
        super().__init__()
        vgg = models.vgg19(pretrained=True)
        del vgg.avgpool, vgg.classifier, vgg.features[-1]
        self.features = vgg.features
        self.method = method
        self.fea_dim = 512
        if method == "gem":
            self.global_desc = GeM(self.fea_dim, gd_dim)
        elif method == "vlad":
            self.global_desc = NetVLADLoupe(self.fea_dim, 15*20, num_cluster, gd_dim, False, True, False)

    def forward(self, img):
        fea = self.features(img)
        if self.method == 'gem':
            gd = self.global_desc(fea.reshape(fea.shape[0], fea.shape[1], -1).transpose(-1, -2))
        elif self.method == 'vlad':
            gd = self.global_desc(fea)
        return gd
