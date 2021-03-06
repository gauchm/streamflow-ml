import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class HistoricConv2d(nn.Conv2d):
    """Conv2d Module for temporal convolution that only convolves over past and current time steps.
    
    Kernel weights for future time steps (dimension T_{in}, starting with kernel_size[0] // 2 + 1) are masked to 0.
    For kernel_size[0] == 1, this is equivalent to a normal Conv2d-operation.
    
    Input: (N, C_{in}, T_{in}, V_{in})
    Output: (N, C_{out}, T_{out}, V_{out})
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.mask = nn.Parameter(torch.zeros((out_channels, in_channels // groups, kernel_size[0], kernel_size[1])),
                                 requires_grad=False)
        self.mask[:,:,:kernel_size[0]//2 + 1, :] = 1
        
        
    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.
    
    Adapted from https://github.com/yysijie/st-gcn
    
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, 
                 t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = HistoricConv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
    

class Model(nn.Module):
    """Spatial temporal graph convolutional networks.

    Adapted from https://github.com/yysijie/st-gcn

    Args:
        in_channels (int): Number of channels in the input data
        spatial_kernel_size (int): Size of the spatial kernel
        temporal_kernel_size (int): Size of the temporal kernel
        edge_importance_weighting (bool, optional): If True, adds a learnable importance weighting to the edges of the graph.
                                                    Need to specify adjacency is used. 
                                                    Note that with this option enabled, the network will only work on the specific graph 
                                                        (or graphs with the same adjacency shape)
        adjacency_shape (torch.Size, optional): shape of the adjacency list. Required if edge_importance_weighting is True.
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, V_{in})` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes.
    """

    def __init__(self, in_channels, spatial_kernel_size, temporal_kernel_size, 
                 edge_importance_weighting=False, adjacency_shape=None, **kwargs):
        super().__init__()

        # build networks
        self.kernel_size = (temporal_kernel_size, spatial_kernel_size)
        if adjacency_shape is not None:
            self.data_bn = nn.BatchNorm1d(in_channels * adjacency_shape[1])
        else:
            # No BatchNorm so we remain flexible in the input graph
            self.data_bn = nn.Identity()
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        hidden_channels = 16
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, self.kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, self.kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, self.kernel_size, 1, **kwargs),
        ))
        
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(adjacency_shape)) for i in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # reduce to one channel for prediction
        self.out_conv = HistoricConv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x, A, max_path_len):

        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)
        
        # Spatio-temporal convolution
        x, _ = self.st_gcn_networks[0](x, A * self.edge_importance[0])
        # make sure the data form the most upstream subbasin can influence the most downstream one.
        # max_path_len is the length of the longest source -> sink path.
        for i in range(max_path_len // self.kernel_size[1]):
            x, _ = self.st_gcn_networks[1](x, A * self.edge_importance[1])
        x, _ = self.st_gcn_networks[2](x, A * self.edge_importance[2])
        
        # prediction
        x = self.out_conv(x)
        return x[:,0,-1,:]  # return first (and only) channel and last time step

    
class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Adapted from https://github.com/yysijie/st-gcn

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1], t_kernel_size=kernel_size[0], t_padding=padding[0])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            HistoricConv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        
        return self.relu(x), A
