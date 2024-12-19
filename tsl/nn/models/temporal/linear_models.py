from typing import Optional

from einops import rearrange
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog
from ...layers import RevIN


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
        
        batch, _ , nodes, features = x.size()
        
        # Padding on both ends of the time dimension
        front = x[:, :1, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        end = x[:, -1:, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        x = torch.cat([front, x, end], dim=1)  # Concatenate padding along the time dimension
        
        x = x.permute(0, 2, 3, 1)  # [b n f t]
        x = self.avg(x.reshape(-1, features, x.size(-1)))  # Apply AvgPool1d along timesteps
        x = x.reshape(batch, nodes, features, -1).permute(0, 3, 1, 2) 
        
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


class DLinearModel(BaseModel):
    """
    Decomposition-Linear
    This code was adapted from the original implementation https://github.com/cure-lab/LTSF-Linear
    """
    def __init__(self,       
                 window: int,         
                 horizon: int,
                 kernel_size: int,
                 bool_individual: bool,
                 input_size: int):
        super(DLinearModel, self).__init__()
        
        self.pred_len = horizon
        self.seq_len = window
        self.input_size = input_size
        self.bool_individual = bool_individual

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
    
        if self.bool_individual:
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
        seasonal_init = seasonal_init.permute(0, 2, 3, 1)  # [batches, nodes, features, steps]
        trend_init = trend_init.permute(0, 2, 3, 1) # [batches, nodes, features, steps]
        if self.bool_individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), seasonal_init.size(2), self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), seasonal_init.size(2), self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            
            for i in range(self.input_size):            
                seasonal_output[:, :, i, :] = self.Linear_Seasonal[i](seasonal_init[:, :, i, :])
                trend_output[:, :, i, :] = self.Linear_Trend[i](trend_init[:, :, i, :])
                
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        
        return x.permute(0,3,1,2)



class NLinearModel(BaseModel):
    """
    Normalization-Linear
    This code was adapted from the original implementation https://github.com/cure-lab/LTSF-Linear
    """
    def __init__(self, 
                 window: int,
                 horizon: int,
                 bool_individual: bool,
                 input_size: int):
        
        super(NLinearModel, self).__init__()
        self.seq_len = window
        self.pred_len = horizon
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = input_size
        self.individual = bool_individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [b, t, n, f]
        seq_last = x[:,-1:,:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2),x.size(3)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,:,i] = self.Linear[i](x[:,:,:,i].permute(0,2,1)).permute(0,2,1)
            x = output
        else:
            x = x.permute(0,2,3,1)
            x = self.Linear(x)
            x = x.permute(0,3,1,2)
        x = x + seq_last
        return x # [Batch, Output length, Channel]
    

    


'This code was adapted from the original implementation https://github.com/VEWOXIC/FITS/blob/main/models/FITS.py'
class RLinearModel(BaseModel):
    def __init__(self, 
                 window: int,
                 horizon: int,
                 input_size: int,
                 bool_individual: bool,
                 bool_rev: bool,
                 bool_drop: bool):
        super(RLinearModel, self).__init__()

        self.Linear = nn.ModuleList([
            nn.Linear(window, horizon) for _ in range(input_size)
        ]) if bool_individual else nn.Linear(window, horizon)
        
        self.dropout = nn.Dropout(bool_drop)
        self.rev = RevIN(input_size) if bool_rev else None
        self.individual = bool_individual
        self.pred_len = horizon


    def forward(self, x):
        # x: [b, t, n, f]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros([x.size(0),self.pred_len,x.size(2),x.size(3)],dtype=x.dtype).to(x.device)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, :, idx] = proj(x[:, :, :, idx].permute(0, 2, 1)).permute(0, 2, 1)
        else:
            pred = self.Linear(x.transpose(1, 3)).transpose(1, 3)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred


class FITSLinearModel(BaseModel):
    # TODO to adjust the dimensions: output and prediciton length
    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, 
                 window: int,
                 horizon: int,
                 input_size: int,
                 H_order: int,
                 base_T: int,
                 cut_freq: int,
                 bool_individual: bool):
        
        '''
        super(FITSLinearModel, self).__init__()
        self.seq_len = window
        self.pred_len = horizon
        self.individual = bool_individual
        self.channels = input_size

        if cut_freq == 0:
            self.dominance_freq = int(window// base_T + 1) * H_order + 10
        else:
            self.dominance_freq = cut_freq
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len
        '''


    def forward(self, x):
        # RIN
        '''
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        print(low_specx.shape, 'low_specx', x.shape, 'x', self.dominance_freq, 'dominance_freq')
        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2),low_specx.size(3)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                print(low_specx.shape)
                low_specxy_[:,:,:,i]=self.freq_upsampler[i](low_specx[:,:,:,i].permute(0,2,1)).permute(0,2,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,3,1)).permute(0,3,1,2)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2),low_specxy_.size(3)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:,:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        print(low_xy.shape, 'low_xy', low_specxy.shape, 'low_specxy')
        low_xy=low_xy * self.length_ratio # energy compemsation for the length change
        # dom_x=x-low_x
        
        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        return xy
        '''



class ExponentialSmoothingModel(BaseModel):
    'Version of the Exponential Smoothing where alpha is learnable'
    def __init__(self, alpha,
                 horizon: int):
        super(ExponentialSmoothingModel, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.horizon = horizon

    def forward(self, x):
        """
        x: [b, t, n, f]
        """
        b, t, n, f = x.shape
        smoothed = x[:, 0, :, :]  # Initial smoothed value (first time step)

        for i in range(1, t):
            smoothed = (self.alpha * x[:,i - 1] + (1 - self.alpha) * smoothed)
        
        # generate forecast over the horizon
        h = self.horizon
        forecast = torch.zeros(b, h, n, f).to('cuda')
        for i in range(1, h):
            forecast[:, i, :, :] = (self.alpha * x[:,i - 1] + (1 - self.alpha) * smoothed)
        
        return forecast
    


