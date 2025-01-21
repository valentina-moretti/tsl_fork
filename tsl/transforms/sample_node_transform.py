from torch_geometric.transforms import BaseTransform
import torch
from tsl.data import Data


class SampleNodeTransform(BaseTransform):
    """Reduce number of time series in :attr:`sample` removing random nodes."""

    def __init__(self, n_nodes: int, sampled_nodes: int, seed: int = None):
        self.n_nodes = n_nodes 
        self.sampled_nodes = sampled_nodes 
        # Save a torch generator with a specific seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
    def __call__(self, data: Data):
        node_mask = torch.randperm(self.n_nodes,
                                   generator=self.generator)[:self.sampled_nodes]
       
       
        for key, value in data.items():
            if key == 'transform':
                continue
            else:
                pattern = data.pattern[key].replace(' ', '')  # 't n f' -> 'tnf'
                if 'n' in pattern:
                    node_dim = pattern.index('n')  # 'tnf'.index('n') -> 1
                    data[key] = torch.index_select(value, node_dim, node_mask)
                if key in data.transform:
                    scaler = data.transform[key]
                    scaler.bias = torch.index_select(scaler.bias, node_dim, node_mask)
                    scaler.scale = torch.index_select(scaler.scale, node_dim, node_mask)


        return data