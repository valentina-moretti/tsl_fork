from typing import Optional

from einops import rearrange
from torch import Tensor
import torch
import torch.nn as nn

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog


class ARModel(BaseModel):
    r"""Simple univariate linear AR model for multistep forecasting.

    Args:
        input_size (int): Size of the input.
        temporal_order (int): Number of units in the recurrent cell.
        output_size (int): Number of output channels.
        exog_size (int): Size of the exogenous variables.
        horizon (int): Forecasting horizon.
    """

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 temporal_order: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 bias: bool = True):
        super(ARModel, self).__init__()

        input_size += exog_size
        self.linear = LinearReadout(input_size=input_size * temporal_order,
                                    output_size=output_size,
                                    horizon=horizon,
                                    bias=bias)
        self.temporal_order = temporal_order

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        x = maybe_cat_exog(x, u)
        x = x[:, -self.temporal_order:]
        x = rearrange(x, 'b s n f -> b n (s f)')
        return self.linear(x)


class VARModel(ARModel):
    r"""A simple VAR model for multistep forecasting.

    Args:
        input_size (int): Size of the input.
        n_nodes (int): Number of nodes.
        temporal_order (int): Number of units in the recurrent cell.
        output_size (int): Number of output channels.
        exog_size (int): Size of the exogenous variables.
        horizon (int): Forecasting horizon.
    """

    def __init__(self,
                 input_size: int,
                 temporal_order: int,
                 output_size: int,
                 horizon: int,
                 n_nodes: int,
                 exog_size: int = 0,
                 bias: bool = True):

        super(VARModel, self).__init__(input_size=input_size * n_nodes,
                                       temporal_order=temporal_order,
                                       output_size=output_size * n_nodes,
                                       horizon=horizon,
                                       exog_size=exog_size,
                                       bias=bias)

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches, steps, nodes, features]
        # u: [batches, steps, (nodes), features]
        *_, n, _ = x.size()
        x = rearrange(x, 'b t n f -> b t 1 (n f)')
        if u is not None and u.dim() == 4:
            u = rearrange(u, 'b t n f -> b t 1 (n f)')
        x = super(VARModel, self).forward(x, u)
        # [b, h, 1, (n f)]
        return rearrange(x, 'b h 1 (n f) -> b h n f', n=n)






class moving_avg(BaseModel):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(BaseModel):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(BaseModel):
    """
    Decomposition-Linear
    """
    def __init__(self,       
                 window_size: int,         
                 horizon: int,
                 kernel_size: int,
                 bool_individual: bool,
                 input_size: int):
        super(DLinear, self).__init__()
        
        self.pred_len = horizon
        self.seq_len = window_size
        self.input_size = input_size

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
    
        if bool_individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(input_size):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)                
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches, steps, nodes, features]
        # u: [batches, steps, (nodes), features]
            
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.input_size):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            print(seasonal_init.shape, trend_init.shape)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        
        return x.permute(0,2,1,3)
