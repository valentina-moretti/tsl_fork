from functools import partial
from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn

from tsl.nn import utils
from tsl.nn.layers.base import MultiHeadAttention
from tsl.nn.layers.norm import LayerNorm
from tsl.nn.blocks.encoders.attention import ProbAttention, FullAttention, AttentionLayer
from tsl.nn.blocks.encoders.embed import DataEmbedding
from tsl.nn.blocks.encoders.informer_encoder import InformerEncoder, InformerEncoderLayer, InformerConvLayer
from tsl.nn.blocks.decoders.informer_decoder import InformerDecoder, InformerDecoderLayer

class TransformerLayer(nn.Module):
    r"""A Transformer layer from the paper `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017).

    This layer can be instantiated to attend the temporal or spatial dimension.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        axis (str, optional): Dimension on which to apply attention to update
            the representations. Can be either, 'time' or 'nodes'.
            (default: :obj:`'time'`)
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention (has an effect only if :attr:`axis` is
            :obj:`'time'`). (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 axis='time',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(TransformerLayer, self).__init__()
        self.att = MultiHeadAttention(embed_dim=hidden_size,
                                      qdim=input_size,
                                      kdim=input_size,
                                      vdim=input_size,
                                      heads=n_heads,
                                      axis=axis,
                                      causal=causal)

        if input_size != hidden_size:
            self.skip_conn = nn.Linear(input_size, hidden_size)
        else:
            self.skip_conn = nn.Identity()

        self.norm1 = LayerNorm(input_size)

        self.mlp = nn.Sequential(LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, ff_size),
                                 utils.get_layer_activation(activation)(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ff_size, hidden_size),
                                 nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)

        self.activation = utils.get_functional_activation(activation)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """"""
        # x: [batch, steps, nodes, features]
        x = self.skip_conn(x) + self.dropout(
            self.att(self.norm1(x), attn_mask=mask)[0])
        x = x + self.mlp(x)
        return x


class SpatioTemporalTransformerLayer(nn.Module):
    r"""A :class:`~tsl.nn.blocks.encoders.TransformerLayer` which attend both
    the spatial and temporal dimensions by stacking two
    :class:`~tsl.nn.layers.base.MultiHeadAttention` layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention.
            (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(SpatioTemporalTransformerLayer, self).__init__()
        self.temporal_att = MultiHeadAttention(embed_dim=hidden_size,
                                               qdim=input_size,
                                               kdim=input_size,
                                               vdim=input_size,
                                               heads=n_heads,
                                               axis='time',
                                               causal=causal)

        self.spatial_att = MultiHeadAttention(embed_dim=hidden_size,
                                              qdim=hidden_size,
                                              kdim=hidden_size,
                                              vdim=hidden_size,
                                              heads=n_heads,
                                              axis='nodes',
                                              causal=False)

        self.skip_conn = nn.Linear(input_size, hidden_size)

        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(hidden_size)

        self.mlp = nn.Sequential(LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, ff_size),
                                 utils.get_layer_activation(activation)(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ff_size, hidden_size),
                                 nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """"""
        # x: [batch, steps, nodes, features]
        x = self.skip_conn(x) + self.dropout(
            self.temporal_att(self.norm1(x), attn_mask=mask)[0])
        x = x + self.dropout(
            self.spatial_att(self.norm2(x), attn_mask=mask)[0])
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    r"""A stack of Transformer layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        output_size (int, optional): Size of an optional linear readout.
        n_layers (int, optional): Number of Transformer layers.
        n_heads (int, optional): Number of parallel attention heads.
        axis (str, optional): Dimension on which to apply attention to update
            the representations. Can be either, 'time', 'nodes', or 'both'.
            (default: :obj:`'time'`)
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention (has an effect only if :attr:`axis` is
            :obj:`'time'` or :obj:`'both'`).
            (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 output_size=None,
                 n_layers=1,
                 n_heads=1,
                 axis='time',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(Transformer, self).__init__()
        self.f = getattr(F, activation)

        if ff_size is None:
            ff_size = hidden_size

        if axis in ['time', 'nodes']:
            transformer_layer = partial(TransformerLayer, axis=axis)
        elif axis == 'both':
            transformer_layer = SpatioTemporalTransformerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        layers = []
        for i in range(n_layers):
            layers.append(
                transformer_layer(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    ff_size=ff_size,
                    n_heads=n_heads,
                    causal=causal,
                    activation=activation,
                    dropout=dropout))

        self.net = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x: Tensor):
        """"""
        x = self.net(x)
        if self.readout is not None:
            return self.readout(x)
        return x







class Informer(nn.Module):
    r"""
    The Informer model for long-sequence time-series forecasting, refactored
    with separated Informer layers for modularity.

    Args:
        enc_in (int): Input dimension for the encoder.
        dec_in (int): Input dimension for the decoder.
        c_out (int): Output dimension.
        seq_len (int): Sequence length for the encoder.
        out_len (int): Output sequence length.
        factor (int): Factor for ProbSparse self-attention.
        d_model (int): Hidden dimensionality.
        n_heads (int): Number of attention heads.
        e_layers (int): Number of encoder layers.
        d_layers (int): Number of decoder layers.
        d_ff (int): Feed-forward network dimensionality.
        dropout (float): Dropout rate.
        attn (str): Type of attention ('prob' or 'full').
        embed (str): Type of embedding ('fixed' or 'learned').
        activation (str): Activation function ('gelu', 'relu', etc.).
        output_attention (bool): Whether to return attention weights.
        distil (bool): Whether to apply convolutional distillation.
        mix (bool): Whether to mix attention in the decoder.
        freq (str): Frequency of the time series.
    """

    def __init__(self, 
                 enc_in, 
                 dec_in, 
                 c_out, 
                 out_len,
                 factor=5, 
                 d_model=16, 
                 n_heads=8, 
                 e_layers=3, 
                 d_layers=2,
                 d_ff=512, 
                 dropout=0.0, 
                 attn='prob', 
                 embed='fixed', 
                 activation='gelu', 
                 output_attention=False, 
                 distil=True, 
                 mix=True,
                 freq='h'):
        super(Informer, self).__init__()

        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = InformerEncoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                InformerConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = InformerDecoder(
            [
                InformerDecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]