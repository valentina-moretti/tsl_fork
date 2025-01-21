from typing import Optional

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch import cat, zeros, ones
from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import Transformer
from tsl.nn.layers import PositionalEncoding
from tsl.nn.layers.ops import Select
from tsl.nn.models.base_model import BaseModel
from tsl.nn.blocks.encoders import Informer


class TransformerModel(BaseModel):
    r"""A Transformer from the paper `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017) for
    multistep time series forecasting.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        output_size (int): Dimension of the output.
        ff_size (int): Units in the MLP after self attention.
        exog_size (int): Dimension of the exogenous variables.
        horizon (int): Number of forecasting steps.
        n_heads (int, optional): Number of parallel attention heads.
        n_layers (int, optional): Number of layers.
        dropout (float, optional): Dropout probability.
        axis (str, optional): Dimension on which to apply attention to update
            the representations. Can be either, 'time', 'nodes', or 'both'.
            (default: :obj:`'time'`)
        activation (str, optional): Activation function.
    """

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 32,
                 n_heads: int = 1,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 axis: str = 'time',
                 activation: str = 'elu'):
        super(TransformerModel, self).__init__()

        if exog_size:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(hidden_size, max_len=100)

        self.transformer_encoder = nn.Sequential(
            Transformer(input_size=hidden_size,
                        hidden_size=hidden_size,
                        ff_size=ff_size,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        activation=activation,
                        dropout=dropout,
                        axis=axis), Select(1, -1))

        self.readout = nn.Sequential(
            MLP(input_size=hidden_size,
                hidden_size=ff_size,
                output_size=output_size * horizon,
                dropout=dropout),
            Rearrange('b n (h f) -> b h n f', f=output_size, h=horizon))

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        b, *_ = x.size()
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b t f -> b t 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        x = self.pe(x)
        x = self.transformer_encoder(x)

        return self.readout(x)





class FCTransformerModel(TransformerModel):

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 n_nodes: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 32,
                 n_heads: int = 1,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 axis: str = 'time',
                 activation: str = 'elu'):
        super(FCTransformerModel, self).__init__(input_size=input_size * n_nodes,
                                               output_size=output_size * n_nodes,
                                               horizon=horizon,
                                               exog_size=exog_size,
                                               hidden_size=hidden_size,
                                               ff_size=ff_size,
                                               n_heads=n_heads,
                                               n_layers=n_layers,
                                               dropout=dropout,
                                               axis=axis,
                                               activation=activation)
        self.n_nodes = n_nodes

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        
        x = rearrange(x, 'b t n f -> b t 1 (n f)')
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b t f -> b t 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = self.readout(x)
        x = rearrange(x, 'b h 1 (n f) -> b h n f', n=self.n_nodes)
        return x


class InformerModel(BaseModel):
    """
    A multistep time series forecasting model based on the Informer architecture.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        horizon (int): Number of steps to forecast.
        seq_len (int): Length of the input sequence.
        label_len (int): Length of the label sequence (used for decoder input).
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the feed-forward layers.
        n_heads (int): Number of parallel attention heads.
        e_layers (int): Number of encoder layers.
        d_layers (int): Number of decoder layers.
        factor (int): Factor for ProbSparse attention mechanism.
        dropout (float): Dropout probability.
        attn (str): Type of attention ('prob' or 'full').
        embed (str): Type of positional embedding ('fixed' or 'learnable').
        activation (str): Activation function.
        distil (bool): Whether to use distilling in the encoder.
        device (torch.device): Device to run the model on.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 window: int,
                 label_len: int = 10,
                 hidden_size: int = 512,
                 d_ff: int = 512,
                 n_heads: int = 8,
                 e_layers: int = 3,
                 d_layers: int = 2,
                 factor: int = 5,
                 dropout: float = 0.1,
                 attn: str = 'prob',
                 embed: str = 'timeF',
                 activation: str = 'gelu',
                 distil: bool = True,
                 output_attention: bool = False,
                 padding: int = 0,
                 mix: bool = False):
        super(InformerModel, self).__init__()
        self.seq_len = window
        self.label_len = label_len
        self.pred_len = horizon
        self.padding = padding
        
        # Define Informer
        self.informer = Informer(
            enc_in=input_size,
            dec_in=input_size,
            c_out=output_size,
            out_len=horizon,
            factor=factor,
            d_model=hidden_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn=attn,
            embed=embed,
            activation=activation,
            distil=distil,
            mix=mix,
            output_attention=output_attention
        )

    def forward(self, x: Tensor, x_mark: Tensor, x_mark_horizon: Tensor) -> Tensor:
        """
        Forward pass for the Informer model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_size].
            x_mark_enc (Tensor): Encoded time features for the encoder input.
            x_mark_dec (Tensor): Encoded time features for the decoder input.

        Returns:
            Tensor: Output tensor of shape [batch_size, horizon, output_size].
        """
        n = x.size(2)
        x = rearrange(x, 'b s n f -> (b n) s f')
        # repeat x_mark and x_mark_horizon for each node
        x_mark_enc = x_mark.unsqueeze(2).repeat(1, 1, n, 1)
        x_mark_horizon = x_mark_horizon.unsqueeze(2).repeat(1, 1, n, 1)
        x_mark_enc = rearrange(x_mark_enc, 'b s n f_cov -> (b n) s f_cov')
        x_mark_horizon = rearrange(x_mark_horizon, 'b s n f_cov -> (b n) s f_cov')
         

        if self.padding==0:
            dec_pad = zeros(x.shape[0], self.pred_len, x.shape[-1]).cuda()
        elif self.padding==1:
            dec_pad = ones(x.shape[0], self.pred_len, x.shape[-1]).cuda()


    
        # x_enc = x[:, :self.seq_len, :]
        x_dec = cat([x[:, self.seq_len-self.label_len:, :], dec_pad], dim=1)
        
        x_mark_dec = cat([x_mark_enc[:, self.seq_len-self.label_len:, :], x_mark_horizon], dim=1)
        output = self.informer(
            x_enc=x,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec
        )
        output = rearrange(output, '(b n) h f -> b h n f', n=n)
        return output


class FCInformerModel(BaseModel):
    """Global Informer model for multistep forecasting.
        Instead of considering a single multivariate time series (as in the Informer model),
        the FCInformer model considers multiple time series (nodes).
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_nodes: int,
                 horizon: int,
                 window: int,
                 label_len: int = 10,
                 hidden_size: int = 512,
                 d_ff: int = 512,
                 n_heads: int = 8,
                 e_layers: int = 3,
                 d_layers: int = 2,
                 factor: int = 5,
                 dropout: float = 0.1,
                 attn: str = 'prob',
                 embed: str = 'timeF',
                 activation: str = 'gelu',
                 distil: bool = True,
                 padding: int = 0,
                 mix: bool = False,
                 output_attention: bool = False
                 ): 
        super(FCInformerModel, self).__init__()
        self.seq_len = window
        self.label_len = label_len
        self.pred_len = horizon
        self.padding = padding
        
        # Define Informer
        self.informer = Informer(
            enc_in=input_size*n_nodes,
            dec_in=input_size*n_nodes,
            c_out=output_size*n_nodes,
            out_len=horizon,
            factor=factor,
            d_model=hidden_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn=attn,
            embed=embed,
            activation=activation,
            distil=distil,
            mix=mix,
            output_attention=output_attention
        )


    def forward(self, x: Tensor, x_mark: Tensor, x_mark_horizon: Tensor) -> Tensor:
        """
        Forward pass for the FCInformer model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_size].
            x_mark (Tensor): Encoded time features corresponding to the input sequence.
            x_mark_horizon (Tensor): Encoded time features corresponding to the horizon.

        Returns:
            Tensor: Output tensor of shape [batch_size, horizon, output_size].
        """
        n = x.size(2)
        x = rearrange(x, 'b s n f -> b s (n f)')
        
        if self.padding==0:
            dec_pad = zeros(x.shape[0], self.pred_len, x.shape[-1]).cuda()
        elif self.padding==1:
            dec_pad = ones(x.shape[0], self.pred_len, x.shape[-1]).cuda()

        x_mark_enc = x_mark
        # x_enc = x[:, :self.seq_len, :]
        
        x_dec = cat([x[:, self.seq_len-self.label_len:, :], dec_pad], dim=1)
        
        x_mark_dec = cat([x_mark_enc[:, self.seq_len-self.label_len:, :], x_mark_horizon], dim=1)
        output = self.informer(
            x_enc=x,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec
        )
        output = rearrange(output, 'b h (n f) -> b h n f', n=n)
        return output
