#!/usr/bin/env python3

import numpy as np
import torch
import random
from utils.misc import rectify_savepath
from utils import feature_pt_ncovis
import copy

class Memory():
    MAX_CAP = 2000
    STATE_DICT = ['swap_dir', 'out_device', 'capacity', 'current_cap', 'eternal_cap', 'current_num', 'eternal_num', 'n_frame',
                  'n_frame_hist', 'property_spec', '_states', '_store', '_rel', 'cutoff', 'steps']

    def __init__(self, property_spec, capacity=MAX_CAP, current_mem=MAX_CAP, longterm_mem=MAX_CAP, swap_dir='./memory', out_device='cuda'):
        super().__init__()
        self.swap_dir = swap_dir
        self.out_device = out_device
        self.capacity = capacity
        self.current_cap = current_mem
        self.eternal_cap = longterm_mem
        self.current_num = 0
        self.eternal_num = 0

        self.n_frame = 0
        self.n_frame_hist = None
        self.property_spec = property_spec
        self._states = {}
        self._store = None
        self._rel = None
        self.cutoff = None
        self.steps = 0
        self.mem = False

    def sample_frames(self, n_anchor, n_recent=0, pos_pair=2, neg_pair=6, n_try=10):
        n_frame = self.n_frame
        for _ in range(n_try):
            # combination of most recent frames and random frames prior to those
            a = torch.randint(n_frame + self.current_num + self.eternal_num, (n_anchor - n_recent,))
            cur_mask = (a >= n_frame)
            eter_mask = (a >= n_frame + self.current_num)
            a[cur_mask] += (self.capacity - n_frame)
            a[eter_mask] += (self.current_cap - self.current_num)

            ank_idx = torch.cat([
                a,
                torch.arange(n_frame - n_recent, n_frame)])
            relevance = self._rel[ank_idx, :self.capacity+self.current_cap+self.eternal_cap]
            # sample based on calculated or predefined cutoff
            if self.cutoff is None:
                cutoff = relevance.where(
                    relevance > 0, torch.tensor(np.nan).to(relevance)).nanquantile(
                    torch.tensor([0.1, 0.9]).to(relevance),
                    dim=1, keepdim=True)
                cutoff[1] = cutoff[1].clamp(min=0.4, max=0.7)
            else:
                cutoff = torch.tensor(self.cutoff)

            pos_prob = (relevance >= cutoff[1]).to(torch.float)
            neg_prob = (relevance <= cutoff[0]).to(torch.float)

            if cutoff.isfinite().all() and ((pos_prob > 0).sum(1) >= pos_pair).all() and ((neg_prob > 0).sum(1) >= neg_pair).all():
                break
        else:
            # no suitable triplets
            return [[None] * 3] * 2 + [[None] * 2]

        pos_idx = torch.multinomial(pos_prob, pos_pair, replacement=True)
        neg_idx = torch.multinomial(neg_prob, neg_pair, replacement=True)
        ank_batch = self._store[ank_idx]
        pos_batch = self._store[pos_idx]
        neg_batch = self._store[neg_idx]

        return (ank_idx, pos_idx, neg_idx), \
            (ank_batch, pos_batch, neg_batch), \
            (relevance.gather(1, pos_idx), relevance.gather(1, neg_idx))

    def store_fifo(self, **properties):
        frame_addr = torch.arange(len(list(properties.values())[0]))
        frame_addr = frame_addr + self.n_frame_hist
        filtered = (frame_addr[frame_addr >= self.capacity]) % self.capacity
        if filtered.shape[0] > 0:
            squ = self._store[filtered]
            self.store_current(**squ)
            self.n_frame = self.capacity
        else:
            self.n_frame += len(frame_addr)
        frame_addr = frame_addr % self.capacity
        self.n_frame_hist += len(frame_addr)
        self._store.store(frame_addr, **properties)
        self.update_rel(frame_addr)

    def store_current(self, **properties):
        length = len(list(properties.values())[0])
        prob = 1.0*self.current_cap/self.steps
        if self.current_num + length <= self.current_cap:
            frame_addr = torch.arange(length) + self.capacity + self.current_num

            self.current_num += length
            self._store.store(frame_addr, **properties)
            self.update_rel(frame_addr)
        elif random.random() < prob:
            idx = torch.randperm(self.current_num)[:length] + self.capacity
            self._store.store(idx, **properties)
            self.update_rel(idx)

    def store_eternal(self):
        if self.eternal_num == 0:
            cur_idx = torch.randperm(self.current_num)[:self.eternal_cap] + self.capacity
            eter_idx = torch.arange(self.eternal_cap) + self.capacity + self.current_cap

            self.eternal_num = cur_idx.shape[0]
            prop = self._store[cur_idx]
            self._store.store(eter_idx, **prop)
            self.update_rel(eter_idx)
        else:
            lamb = 0.5
            cur_idx = torch.randperm(self.current_num)[:int(self.eternal_cap*lamb)] + self.capacity
            eter_idx = torch.randperm(self.eternal_num)[:int(self.eternal_cap*lamb)] + self.capacity + self.current_cap
            prop = self._store[cur_idx]
            self._store.store(eter_idx, **prop)
            self.update_rel(eter_idx)

    def swap(self, name):
        if name in self._states:
            self._store, self._rel, self.n_frame_hist = self._states[name]
        else:
            if not self.mem:
                self.n_frame_hist = np.array(0)
                self._store = SparseStore(name=name, out_device=self.out_device,
                                          max_cap=self.capacity + self.current_cap + self.eternal_cap, **self.property_spec)
                self._rel = torch.zeros(self.capacity + self.current_cap + self.eternal_cap,
                                        self.capacity + self.current_cap + self.eternal_cap, device=self.out_device).fill_(
                    np.nan)
                self._states[name] = self._store, self._rel, self.n_frame_hist

            else:
                self.n_frame_hist = np.array(0)
                self.current_num, self.n_frame = 0, 0
                self._rel[:self.capacity+self.current_cap, :] = np.nan
                self._rel[:, :self.capacity+self.current_cap] = np.nan
                idx = torch.arange(self.eternal_num) + self.capacity + self.current_cap
                prop = copy.deepcopy(self._store[idx])
                self._store = SparseStore(name=name, out_device=self.out_device,
                                          max_cap=self.capacity + self.current_cap + self.eternal_cap,
                                          **self.property_spec)
                self._store.store(idx, **prop)
                self._states[name] = self._store, self._rel, self.n_frame_hist
                for key in list(self._states.keys()):
                    if key != name:
                        del self._states[key]
        self.n_frame = len(self._store) - self.current_num - self.eternal_num


    def update_rel(self, frame_idx):
        frame_idx = frame_idx.to(self._rel.device)
        # (n_frame, B)
        relevance = self.get_rel(torch.arange(self.n_frame), frame_idx).to(self._rel)
        self._rel[:self.n_frame, frame_idx] = relevance
        self._rel[frame_idx, :self.n_frame] = relevance.T
        if self.current_num > 0:
            relevance = self.get_rel(torch.arange(self.current_num) + self.capacity, frame_idx).to(self._rel)
            self._rel[self.capacity:self.capacity + self.current_num, frame_idx] = relevance
            self._rel[frame_idx, self.capacity:self.capacity + self.current_num] = relevance.T
        if self.eternal_num > 0:
            relevance = self.get_rel(torch.arange(self.eternal_num) + self.capacity + self.current_cap, frame_idx).to(
                self._rel)
            self._rel[self.capacity + self.current_cap:self.capacity + self.current_cap + self.eternal_num,
            frame_idx] = relevance
            self._rel[frame_idx,
            self.capacity + self.current_cap:self.capacity + self.current_cap + self.eternal_num] = relevance.T


    def get_rel(self, src_idx, dst_idx):
        raise NotImplementedError()

    def save(self, path, overwrite=True):
        save_path = rectify_savepath(path, overwrite=overwrite)
        torch.save(self, save_path)
        print('Saved memory: %s' % save_path)

    def load(self, path):
        self.mem = True
        loaded_mem = torch.load(path)
        for attr in self.STATE_DICT:
            setattr(self, attr, getattr(loaded_mem, attr))
        print('Loaded memory: %s' % path)

    def __len__(self):
        return self._store.__len__()

    def envs(self):
        return self._states.keys()


class TartanAirMemory(Memory):

    def __init__(self, capacity=Memory.MAX_CAP, current_mem=Memory.MAX_CAP, longterm_mem=Memory.MAX_CAP, n_probe=1200, img_size=(240, 320), swap_dir='./memory', out_device='cuda'):
        TARTANAIR_SPEC = {
            'pos': {'shape': (n_probe, 3), 'default': np.nan, 'device': out_device},
            'img': {'shape': (3,) + img_size, 'default': np.nan},
            'pose': {'shape': (3, 4), 'default': np.nan},
            'K': {'shape': (3, 3), 'default': np.nan},
            'depth_map': {'shape': (1,) + img_size, 'default': np.nan},
        }
        super().__init__(TARTANAIR_SPEC, capacity, current_mem, longterm_mem, swap_dir, out_device)
        self.STATE_DICT.append('n_probe')

        self.n_probe = n_probe

    def get_rel(self, src_idx, dst_idx):
        src_pos = self._store[src_idx, ['pos']]['pos']
        dst_info = self._store[dst_idx, ['pos', 'pose', 'depth_map', 'K']]
        dst_pos, dst_depth_map, dst_pose, dst_K = dst_info['pos'], dst_info['depth_map'], dst_info['pose'], dst_info['K']

        return feature_pt_ncovis(src_pos, dst_pos, dst_depth_map, dst_pose, dst_K)


class NordlandMemory(Memory):

    def __init__(self, window=5, capacity=Memory.MAX_CAP, current_mem=Memory.MAX_CAP, longterm_mem=Memory.MAX_CAP, img_size=(240, 320), swap_dir='./memory', out_device='cuda'):
        NORDLAND_SPEC = {
            'img': {'shape': (3,) + img_size, 'default': np.nan},
            'offset': {'shape': (), 'dtype': torch.int, 'default': -1},
        }
        super().__init__(NORDLAND_SPEC, capacity, current_mem, longterm_mem, swap_dir, out_device)
        self.STATE_DICT.append('cutoff')
        self.cutoff = [1 / (window + 0.5), 1 / (window + 0.5)] #0.18

    def get_rel(self, src_idx, dst_idx):
        src_off = self._store[src_idx, ['offset']]['offset']
        dst_off = self._store[dst_idx, ['offset']]['offset']

        return 1 / ((src_off[:, None] - dst_off[None, :]).abs() + 1)


class RobotCarMemory(Memory):

    def __init__(self, dist_tol=20, head_tol=15, capacity=Memory.MAX_CAP, current_mem=Memory.MAX_CAP, longterm_mem=Memory.MAX_CAP, img_size=(240, 320), swap_dir='./memory', out_device='cuda'):
        ROBOTCAR_SPEC = {
            'img': {'shape': (3,) + img_size, 'default': np.nan},
            'location': {'shape': (2,), 'dtype': torch.float64, 'default': np.nan},
            'heading': {'shape': (), 'default': np.nan},
        }
        super().__init__(ROBOTCAR_SPEC, capacity, current_mem, longterm_mem, swap_dir, out_device)
        self.STATE_DICT.extend(['cutoff', 'head_tol'])
        self.head_tol = head_tol
        self.cutoff = [1 / (dist_tol * 2 + 1), 1 / (dist_tol + 1)]

    def get_rel(self, src_idx, dst_idx):
        src_info = self._store[src_idx, ['location', 'heading']]
        dst_info = self._store[dst_idx, ['location', 'heading']]
        dist = torch.cdist(src_info['location'], dst_info['location']).to(torch.float)
        view_diff = (src_info['heading'][:, None] - dst_info['heading'][None, :]).abs()

        return (view_diff < self.head_tol).to(torch.float) / (dist + 1)


class CrossMemory(Memory):
    def __init__(self, dataset, current_data, head_tol=15, n_probe=1200, capacity=Memory.MAX_CAP, current_mem=Memory.MAX_CAP, longterm_mem=Memory.MAX_CAP, img_size=(240, 320), swap_dir='./memory', out_device='cuda'):


        self.head_tol = head_tol
        self.datasets = dataset
        self.current_data = current_data
        # merged property space
        PROPERTY_SPEC = {
            'img': {'shape': (3,) + img_size, 'default': np.nan},
        }
        if 'nordland' in dataset:
            PROPERTY_SPEC.update({
                'offset': {'shape': (), 'dtype': torch.int, 'default': -1},
            })
            # self.STATE_DICT.append('cutoff')
        if 'robotcar' in dataset:
            PROPERTY_SPEC.update({
                'location': {'shape': (2,), 'dtype': torch.float64, 'default': np.nan},
                'heading': {'shape': (), 'default': np.nan},
            })
            # self.STATE_DICT.extend(['cutoff', 'head_tol'])
        if 'tartanair' in dataset:
            PROPERTY_SPEC.update({
                'pos': {'shape': (n_probe, 3), 'default': np.nan, 'device': out_device},
                'pose': {'shape': (3, 4), 'default': np.nan},
                'K': {'shape': (3, 3), 'default': np.nan},
                'depth_map': {'shape': (1,) + img_size, 'default': np.nan},
            })
            # self.STATE_DICT.append('n_probe')
        super().__init__(PROPERTY_SPEC, capacity, current_mem, longterm_mem, swap_dir, out_device)
        self.data_belong = torch.full((capacity + self.current_cap + self.eternal_cap,), -1)

    def sample_frames(self, n_anchor, n_recent=0, pos_pair=2, neg_pair=6, n_try=10):
        n_frame = self.n_frame
        for _ in range(n_try):
            # combination of most recent frames and random frames prior to those
            a = torch.randint(n_frame + self.current_num + self.eternal_num, (n_anchor - n_recent,))
            cur_mask = (a >= n_frame)
            eter_mask = (a >= n_frame + self.current_num)
            a[cur_mask] += (self.capacity - n_frame)
            a[eter_mask] += (self.current_cap - self.current_num)

            ank_idx = torch.cat([
                a,
                torch.arange(n_frame - n_recent, n_frame)])
            relevance = self._rel[ank_idx, :self.capacity+self.current_cap+self.eternal_cap]
            # sample based on calculated or predefined cutoff
            data_belong = self.data_belong[ank_idx]
            cut_offs = torch.zeros((len(data_belong), 2), dtype=torch.float)

            # nordland
            window = 5
            cut_offs[data_belong == 1, :] = torch.tensor([1 / (window + 0.5), 1 / (window + 0.5)])
            # robotcar
            dist_tol = 20
            cut_offs[data_belong == 2, :] = torch.tensor([1 / (dist_tol * 2 + 1), 1 / (dist_tol + 1)])
            # tartanair
            mask_3 = data_belong == 3
            if mask_3.any():
                relevance_3 = relevance[mask_3]
                if relevance_3.nelement() > 0 and not torch.isnan(relevance_3).all():
                    quantiles = relevance_3.nanquantile(torch.tensor([0.1, 0.9]), dim=1, keepdim=True)
                    cut_offs[mask_3, :] = torch.cat([quantiles[0], quantiles[1].clamp(min=0.4, max=0.7)], dim=1)
            cut_offs = cut_offs.to(relevance.device)
            pos_prob = (relevance >= cut_offs[:, 1].unsqueeze(1)).to(torch.float)
            neg_prob = (relevance <= cut_offs[:, 0].unsqueeze(1)).to(torch.float)

            if cut_offs.isfinite().all() and ((pos_prob > 0).sum(1) >= pos_pair).all() and ((neg_prob > 0).sum(1) >= neg_pair).all():
                break
        else:
            # no suitable triplets
            return [[None] * 3] * 2 + [[None] * 2]

        pos_idx = torch.multinomial(pos_prob, pos_pair, replacement=True)
        neg_idx = torch.multinomial(neg_prob, neg_pair, replacement=True)

        ank_batch = self._store[ank_idx]
        pos_batch = self._store[pos_idx]
        neg_batch = self._store[neg_idx]

        return (ank_idx, pos_idx, neg_idx), \
            (ank_batch, pos_batch, neg_batch), \
            (relevance.gather(1, pos_idx), relevance.gather(1, neg_idx))


    def get_rel(self, src_idx, dst_idx):
        # judge src dataset and dst dataset
        src_idx = src_idx.to(self.out_device)
        dst_idx = dst_idx.to(self.out_device)
        if 'nordland' in self.datasets:
            src_off = self._store[src_idx, ['offset']]['offset']
            dst_off = self._store[dst_idx, ['offset']]['offset']
            src_nl_mask = src_off != -1
            dst_nl_mask = dst_off != -1
            self.data_belong[src_idx[src_nl_mask]] = 1
        if 'robotcar' in self.datasets:
            src_info = self._store[src_idx, ['location', 'heading']]
            dst_info = self._store[dst_idx, ['location', 'heading']]
            src_rc_mask = ~torch.isnan(src_info['heading'])
            dst_rc_mask = ~torch.isnan(dst_info['heading'])
            self.data_belong[src_idx[src_rc_mask]] = 2
        if 'tartanair' in self.datasets:
            src_pos = self._store[src_idx, ['pos']]['pos']
            dst_inf = self._store[dst_idx, ['pos', 'pose', 'depth_map', 'K']]
            dst_pos, dst_depth_map, dst_pose, dst_K = dst_inf['pos'], dst_inf['depth_map'], dst_inf['pose'], \
                dst_inf['K']
            src_ta_mask = ~torch.isnan(src_pos).all(dim=-1).all(dim=-1)
            dst_ta_mask = ~torch.isnan(dst_inf['pose']).all(dim=-1).all(dim=-1)
            self.data_belong[src_idx[src_ta_mask]] = 3
        rel_scores = torch.zeros(len(src_idx), len(dst_idx)).to(self.out_device)
        # nordland
        if 'nordland' in self.datasets and src_nl_mask.any() and dst_nl_mask.any():
            src_off_masked = src_off[src_nl_mask]
            dst_off_masked = dst_off[dst_nl_mask]
            valid_scores = 1 / ((src_off_masked[:, None] - dst_off_masked[None, :]).abs() + 1)
            rel_scores.masked_scatter_(src_nl_mask[:, None] & dst_nl_mask, valid_scores)
        # robotcar
        if 'robotcar' in self.datasets and src_rc_mask.any() and dst_rc_mask.any():
            src_heading_masked = src_info['heading'][src_rc_mask]
            src_location_masked = src_info['location'][src_rc_mask]
            dst_heading_masked = dst_info['heading'][dst_rc_mask]
            dst_location_masked = dst_info['location'][dst_rc_mask]
            view_diff = (src_heading_masked[:, None] - dst_heading_masked[None, :]).abs()
            dist = torch.cdist(src_location_masked, dst_location_masked).to(torch.float)
            valid_scores = (view_diff < self.head_tol).to(torch.float) / (dist + 1)
            rel_scores.masked_scatter_(src_rc_mask[:, None] & dst_rc_mask, valid_scores)
        # tartanair
        if 'tartanair' in self.datasets and src_ta_mask.any() and dst_ta_mask.any():
            src_pos = src_pos[src_ta_mask]
            dst_pos, dst_depth_map, dst_pose, dst_K = dst_pos[dst_ta_mask], dst_depth_map[dst_ta_mask], dst_pose[dst_ta_mask], dst_K[dst_ta_mask]
            valid_scores = feature_pt_ncovis(src_pos, dst_pos, dst_depth_map, dst_pose, dst_K)
            # valid_scores = torch.zeros(len(src_ta_mask), len(dst_ta_mask))
            rel_scores.masked_scatter_(src_ta_mask[:, None] & dst_ta_mask, valid_scores)
        return rel_scores

class SparseStore():

    def __init__(self, name='store', max_cap=2000, device='cpu', out_device='cuda', **property_spec):
        super().__init__()
        self.name = name
        self.buf = {}
        for name, specs in property_spec.items():
            shape = specs['shape']
            cap = specs.get('max_cap', max_cap)
            dtype = specs.get('dtype', torch.float32)
            dev = specs.get('device', device)
            def_val = specs.get('default', 0)

            self.buf[name] = {
                'shape': shape, 'capacity': cap, 'dtype': dtype, 'device': dev, 'default': def_val, 'values': {}}

        self.size = 0
        self.out_device = out_device

    @torch.no_grad()
    def store(self, _idx=None, **values):
        # sanity check
        batch = []
        for name, val in values.items():
            prop_shape = self.buf[name]['shape']
            prop_shape_st = len(val.shape) - len(prop_shape)
            assert prop_shape == val.shape[prop_shape_st:]
            batch.append(val.shape[:prop_shape_st])
        # coherent and linear indexing
        assert all(b == batch[0] for b in batch) and len(batch[0]) <= 1

        if isinstance(_idx, torch.Tensor):
            # avoids copy construct warning
            _idx = _idx.to(torch.long)
        elif _idx is not None:
            # any scalar or iterable
            _idx = torch.tensor(_idx, dtype=torch.long)
        else:
            # default indices
            _idx = torch.tensor(self.size, dtype=torch.long) if len(batch[0]) == 0 else \
                torch.arange(int(batch[0][0])) + self.size
        assert (len(_idx.shape) == len(batch[0]) == 0) or (len(_idx.shape) == 1 and len(_idx) == int(batch[0][0]))

        for name, val in values.items():
            self._store(self.buf[name], _idx, val)

        self.size = max(len(buf['values']) for buf in self.buf.values())

    def _store(self, buf, idx, value):
        value = value.to(buf['device'])
        if len(idx.shape) == 0:
            buf['values'][int(idx)] = value
        else:
            for i, val in zip(idx.tolist(), value.to(buf['device'], non_blocking=True)):
                buf['values'][int(i)] = val

    def _get(self, buf, idx):
        if int(idx) in buf['values']:
            return buf['values'][int(idx)]
        else:
            return torch.zeros(buf['shape'], dtype=buf['dtype'], device=buf['device']).fill_(buf['default'])

    @torch.no_grad()
    def __getitem__(self, idx):
        idx, include = idx if isinstance(idx, tuple) else (idx, self.buf.keys())
        idx = idx.to(torch.long) if isinstance(idx, torch.Tensor) else torch.tensor(idx, dtype=torch.long)
        ret = {name: [] for name in self.buf if name in include}
        if len(idx.shape) == 0:
            for name, buf in self.buf.items():
                ret[name].append(self._get(buf, idx))
            return {name: tensors[0] for name, tensors in ret.items()}
        else:
            for name in ret:
                for i in idx.flatten():
                    ret[name].append(self._get(self.buf[name], i))
            # make returned tensor respect shape of index
            return {name: torch.stack(tensors).reshape(*idx.shape, *tensors[0].shape).to(self.out_device, non_blocking=True)
                    for name, tensors in ret.items()}

    def __len__(self):
        return self.size


def test_store():
    store = SparseStore(pos={'shape': (12, 3), 'device': 'cuda', 'default': np.nan}, idx={'shape': (1,), 'dtype': torch.long})
    pos = torch.arange(360).reshape(10, 12, 3).to(torch.float)
    idx = torch.arange(10).reshape(10, 1)
    store.store(0, pos=pos[0], idx=idx[0])
    store.store(pos=pos[1:3], idx=idx[1:3])
    store.store(torch.arange(1, 4).cuda(), pos=pos[4:7], idx=idx[4:7])
    print(len(store))
    print(store[1])
    print(store[[0, 2]])
    print(store[[[0], [4]]])


if __name__ == '__main__':
    # test_store()
    memory = CrossMemory(dataset='nordlandrobotcar', capacity=200, head_tol=15)
    memory.swap('spring', None)
    imgs = torch.randn(2, 3, 240, 320)
    offset = torch.IntTensor([10,5])
    memory.store_fifo(img=imgs, offset=offset)
    imgs = torch.randn(2, 3, 240, 320)
    location = torch.randn(2, 2).to(torch.double)
    heading = torch.randn(2).to(torch.double)
    memory.store_fifo(img=imgs, location=location, heading=heading)
    imgs = torch.randn(1, 3, 240, 320)
    offset = torch.IntTensor([9])
    memory.store_fifo(img=imgs, offset=offset)
    print(memory._rel[:5, :5])
    memory.sample_frames(n_anchor=2, pos_pair=1, neg_pair=1)
    a = memory._store[0]
    b = memory._store[1]
    print(1)