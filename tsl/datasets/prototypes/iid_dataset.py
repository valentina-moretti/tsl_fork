# from tsl.data import SynchMode, BatchMapItem
from tsl.data.synch_mode import HORIZON, WINDOW
# from tsl.typing import DataArray
# import numpy as np
# from tsl.data.batch_map import BatchMap
from tsl.data import Data
import torch
# from tsl.data.mixin import DataParsingMixin
from tsl.data.preprocessing import ScalerModule
# from torch import Tensor
# from typing import Union
from torch.utils.data import Sampler
from tsl.data import SpatioTemporalDataset


_WINDOWING_KEYS = ['data', 'window', 'delay', 'horizon', 'stride']

class IIDDataset(SpatioTemporalDataset):
    
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
    
    def set_axis(self, axis):
        self.axis = axis
        
    @property
    def is_random_iid(self):
        return hasattr(self, 'batch_size')


    def __len__(self):
        return len(self._indices)

    
    def __setattr__(self, key, value):
        super(IIDDataset, self).__setattr__(key, value)
        
        if key in _WINDOWING_KEYS and all([hasattr(self, attr)
                                        for attr in _WINDOWING_KEYS]):
            last = (self.n_steps - self.sample_span + 1) * self.n_nodes
            self._indices = torch.arange(0, last, self.stride)
    
    def __getitem__(self, item):
        if self.is_random_iid:
            assert self.batch_size is not None, "batch_size is None."
            item = self.sample(self.batch_size)
            if self.transform is not None:
                item = self.transform(item)
            return item

        return super(IIDDataset, self).__getitem__(item)
    
    def make_random_iid(self, batch_size):
        self.batch_size = batch_size

    def sample(self, N):
        step_index = torch.randint(0, self.n_steps - self.horizon - self.window, (N,))
        node_index = torch.randint(0, self.n_nodes, (N,))
        window_index_expanded = torch.stack([step_index + i for i in range(self.window)], 1)
        sample = Data()
        for key, value in self.input_map.by_synch_mode(WINDOW).items():
            assert len(value.keys) == 1
            k = value.keys[0]
            tens, trans, pattern = getattr(self, k), self.scalers.get(k), self.patterns[k]
            if 'n' in pattern:
                tens = tens[window_index_expanded, node_index[:, None], None] 
                
            elif 't' in pattern:
                tens = tens[(step_index, None)]
            if trans is not None:
                if self.axis == 0:
                    sample.transform[key] = ScalerModule(**{k: p[None]
                                                            for k, p in
                                                            trans.params().items()})
                else:
                    sample.transform[key] = ScalerModule(**{k: p[None][:, :, node_index, :].permute(2,0,1,3) for k, p in trans.params().items() }) # why permute has 4 arguments?
                
                if value.preprocess:
                    tens = sample.transform[key].transform(tens)
                    
            sample.input[key] = tens
            sample.pattern[key] = pattern
            


        hor_index = torch.stack([step_index + i for i in range(self.delay + self.window, self.horizon + self.window, self.horizon_lag)], 1)
        for key, value in self.target_map.by_synch_mode(HORIZON).items():
            assert len(value.keys) == 1
            k = value.keys[0]
            tens, trans, pattern = getattr(self, k), self.scalers.get(k), self.patterns[k]
            if 'n' in pattern:
                tens = tens[(hor_index, node_index[:, None], None)]  # [N h 1 f]
            elif 't' in pattern:
                tens = tens[(hor_index, None)]
            if trans is not None:
                if self.axis == 0:
                    sample.transform[key] = ScalerModule(**{k: p[None]
                                                            for k, p in
                                                            trans.params().items()})
                else: 
                    sample.transform[key] = ScalerModule(**{k: p[None][:, :, node_index, :].permute(2,0,1,3) for k, p in trans.params().items() })
                if value.preprocess:
                    tens = trans.transform(tens)
            sample.target[key] = tens
            sample.pattern[key] = pattern
        sample.input.node_index = node_index[:, None]

        return sample


