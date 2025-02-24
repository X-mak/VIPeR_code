#!/usr/bin/env python3

import numpy as np
import torch

from models.featurenet import DinoV2ExtractFeatures
from utils import Visualizer
from models.memory import TartanAirMemory, NordlandMemory, RobotCarMemory, CrossMemory
from utils import Projector, PairwiseCosine, gen_probe

from .lifelong import get_ll_loss


class MemReplayLoss():
    def __init__(self, writer=None, viz_start=float('inf'), viz_freq=200, counter=None, args=None):
        super().__init__()
        self.args = args
        self.writer, self.counter, self.viz_start, self.viz_freq = writer, counter, viz_start, viz_freq
        self.viz = Visualizer('tensorboard', writer=self.writer)
        if args.cross:
            self.memory = CrossMemory(dataset=args.all_data, current_data=args.dataset, capacity=args.mem_size, current_mem=args.current_mem_size, longterm_mem=args.longterm_mem_size)
        elif args.dataset == 'tartanair':
            self.memory = TartanAirMemory(capacity=args.mem_size, n_probe=1200, out_device=args.device, current_mem=args.current_mem_size, longterm_mem=args.longterm_mem_size)
        elif args.dataset == 'nordland':
            self.memory = NordlandMemory(capacity=args.mem_size, out_device=args.device, current_mem=args.current_mem_size, longterm_mem=args.longterm_mem_size)
        elif args.dataset == 'robotcar':
            self.memory = RobotCarMemory(capacity=args.mem_size, dist_tol=20, head_tol=15, out_device=args.device, current_mem=args.current_mem_size, longterm_mem=args.longterm_mem_size)

        self.n_triplet, self.n_recent, self.pos_pair, self.neg_pair, self.n_pair = 4, 0, 1, 5, 1
        self.min_sample_size = 32
        if not args.memory:
            self.memory.capacity, self.memory.current_cap, self.memory.eternal_cap = 1000, 0, 0
        if args.method == 'airloop':
            self.neg_pair = 1
        self.gd_match = GlobalDescMatchLoss(n_triplet=self.n_triplet, pos_pair=self.pos_pair, neg_pair=self.neg_pair, writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
        self.ll_loss = get_ll_loss(args, writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
        self.prev_gd = 0
        self.backbone = args.backbone
        if self.backbone == 'dinov2':
            self.extractor = DinoV2ExtractFeatures('dinov2_vitg14', 31, 'value', device=args.device)

    def __call__(self, net, img, aux, env, steps):
        device = img.device
        self.memory.steps = steps
        self.store_memory(img, aux, env)

        if len(self.memory) < self.min_sample_size:
            return torch.zeros(1).to(device)

        _, (ank_batch, pos_batch, neg_batch), (pos_rel, neg_rel) = \
            self.memory.sample_frames(self.n_triplet, self.n_recent, self.pos_pair, self.neg_pair)

        # no suitable triplet
        if ank_batch is None:
            return torch.zeros(1).to(device)

        img = recombine('img', ank_batch, pos_batch, neg_batch)
        top_thres = 0.05
        if self.args.dataset == 'nordland':top_thres = 0.08
        low_thres = 0.03
        loss = 0
        if self.backbone == 'vgg':
            gd = net(img=img)
            gd_match = self.gd_match(gd)
        elif self.backbone == 'dinov2':
            fea = self.extractor(img=img)
            gd = net(x=fea.reshape(fea.shape[0], 17, 22, -1).permute(0, 3, 1, 2))
            gd_match = self.gd_match(gd)
        if self.prev_gd != 0:
            if gd_match > self.prev_gd and gd_match - self.prev_gd > top_thres:
                self.gd_match.k = min(self.gd_match.k + 1, self.neg_pair-1)
            elif gd_match < self.prev_gd and self.prev_gd - gd_match > low_thres:
                self.gd_match.k = max(self.gd_match.k - 1, 0)

        self.prev_gd = gd_match
        loss += gd_match

        # forgetting prevention
        if self.ll_loss is not None:
            for ll_loss in self.ll_loss:
                loss_name = ll_loss.name.lower()
                if loss_name in ['mas', 'rmas', 'ewc', 'cewc', 'si']:
                    loss += ll_loss(model=net, gd=gd)
                elif loss_name in ['kd', 'rkd', 'pkd']:
                    if self.backbone == 'dinov2':
                        loss += ll_loss(model=net, gd=gd, img=fea.reshape(fea.shape[0], 17, 22, -1).permute(0, 3, 1, 2))
                    else: loss += ll_loss(model=net, gd=gd, img=img)
                else:
                    raise ValueError(f'Unrecognized lifelong loss: {ll_loss}')

        # logging and visualization
        if self.writer is not None:
            n_iter = self.counter.steps if self.counter is not None else 0
            self.writer.add_scalars('Loss', {'global': gd_match}, n_iter)
            self.writer.add_histogram('Misc/RelN', neg_rel, n_iter)
            self.writer.add_histogram('Misc/RelP', pos_rel, n_iter)
            self.writer.add_scalars('Misc/MemoryUsage', {'len': len(self.memory)}, n_iter)

        return loss

    def store_memory(self, imgs, aux, env):
        self.memory.swap(env[0])
        if isinstance(self.memory, TartanAirMemory) or (isinstance(self.memory, CrossMemory) and self.memory.current_data == 'tartanair'):
            depth_map, pose, K = aux
            points_w = Projector.pix2world(gen_probe(depth_map), depth_map, pose, K)
            self.memory.store_fifo(pos=points_w, img=imgs, depth_map=depth_map, pose=pose, K=K)
        elif isinstance(self.memory, NordlandMemory) or (isinstance(self.memory, CrossMemory) and self.memory.current_data == 'nordland'):
            offset = aux
            self.memory.store_fifo(img=imgs, offset=offset)
        elif isinstance(self.memory, RobotCarMemory) or (isinstance(self.memory, CrossMemory) and self.memory.current_data == 'robotcar'):
            location, heading = aux
            self.memory.store_fifo(img=imgs, location=location, heading=heading)

def recombine(key, *batches):
    tensors = [batch[key] for batch in batches]
    reversed_shapes = [list(reversed(tensor.shape[1:])) for tensor in tensors]
    common_shapes = []
    for shape in zip(*reversed_shapes):
        if all(s == shape[0] for s in shape):
            common_shapes.insert(0, shape[0])
        else:
            break
    return torch.cat([tensor.reshape(-1, *common_shapes) for tensor in tensors])


class GlobalDescMatchLoss():

    def __init__(self, n_triplet=8, pos_pair=2, neg_pair=6, writer=None,
                 viz=None, viz_start=float('inf'), viz_freq=200, counter=None, debug=False):
        super().__init__()
        self.counter, self.writer = counter, writer
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq
        self.cosine = PairwiseCosine()
        self.n_triplet, self.pos_pair, self.neg_pair = n_triplet, pos_pair, neg_pair
        self.k = 0

    def __call__(self, gd):
        gd_a, gd_p, gd_n = gd.split([self.n_triplet, self.n_triplet * self.pos_pair, self.n_triplet * self.neg_pair])
        gd_a = gd_a[:, None]
        gd_p = gd_p.reshape(self.n_triplet, self.pos_pair, *gd_p.shape[1:])
        gd_n = gd_n.reshape(self.n_triplet, self.neg_pair, *gd_n.shape[1:])

        sim_ap = self.cosine(gd_a, gd_p)
        sim_an = self.cosine(gd_a, gd_n)

        sim_an = sim_an.squeeze(1)
        sorted_indices_tensor = torch.argsort(sim_an, dim=1, descending=True)

        batch_indices = torch.arange(sim_an.size(0))
        k_indices = sorted_indices_tensor[batch_indices, self.k]
        an = sim_an[batch_indices, k_indices]
        triplet_loss = (an - sim_ap + 1).clamp(min=0)

        # logging
        n_iter = self.counter.steps
        if self.writer is not None:
            self.writer.add_histogram('Misc/RelevanceLoss', triplet_loss, n_iter)
            self.writer.add_histogram('Misc/SimAP', sim_ap, n_iter)
            self.writer.add_histogram('Misc/SimAN', sim_an, n_iter)
            self.writer.add_scalars('Misc/GD', {'2-Norm': torch.norm(gd, dim=-1).mean()}, n_iter)

        return triplet_loss.mean()