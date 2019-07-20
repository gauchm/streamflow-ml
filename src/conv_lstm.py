# Source: https://github.com/ndrplz/ConvLSTM_pytorch

import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, pooling=False,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        pooling  = self._extend_for_multilayer(pooling, num_layers)
        if not len(kernel_size) == len(hidden_dim) == len(pooling) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.pooling = pooling
        self.return_all_layers = return_all_layers

        cell_list = []
        height, width = self.height, self.width
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            
            cell_list.append(ConvLSTMCell(input_size=(height, width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
            if self.pooling[i]:
                height -= 2 * (kernel_size[i][0] // 2)
                width -= 2 * (kernel_size[i][1] // 2)
                cell_list.append(nn.MaxPool2d(self.kernel_size[i], stride=1, padding=0))
            else:
                cell_list.append(nn.Identity())

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: list or None
            If list of (h,c)-tuples, will use as LSTM states for the LSTM layers. 
            If None, will initialize states on each batch.
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        hidden_states = []
        if hidden_state is not None:
            hidden_states = hidden_state
        else:
            for layer_idx in range(self.num_layers):
                hidden_states.append(self.cell_list[2 * layer_idx].init_hidden(input_tensor.size(0), input_tensor.device))
        
        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[2 * layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                     cur_state=[h, c])
                output_inner.append(self.cell_list[2*layer_idx+1](h))

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            last_state_list.append((h.detach(), c.detach()))  # detach to stop BPTT between batches
            if self.return_all_layers | (layer_idx == 0):
                layer_output_list.append(layer_output)
            else:
                # Save memory if we will only need the last output anyways
                layer_output_list[0] = layer_output

        return layer_output_list, last_state_list

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    
class ConvLSTMRegression(nn.Module):
    def __init__(self, conv_input_size, fc_input_size, conv_input_dim, conv_hidden_dim, kernel_size, 
                 num_conv_layers, dropout, num_fc_layers, fc_hidden_dim, pooling, batch_first=False):
        super(ConvLSTMRegression, self).__init__()
        self.conv_lstm = ConvLSTM(conv_input_size, conv_input_dim, conv_hidden_dim, kernel_size, 
                                  num_conv_layers, pooling=pooling, batch_first=batch_first)
        self.dropout = nn.Dropout2d(p=dropout)
        self.pooling = pooling
        
        conv_out_height = conv_input_size[0]
        conv_out_width = conv_input_size[1]
        for i in range(num_conv_layers):
            if pooling[i]:
                conv_out_height -= 2 * (kernel_size[0] // 2)
                conv_out_width -= 2 * (kernel_size[1] // 2)
        conv_out_dim = conv_hidden_dim[-1] if isinstance(conv_hidden_dim, list) else conv_hidden_dim
        
        if num_fc_layers > 1:
            fc_layers = [nn.Linear(conv_out_dim * conv_out_height * conv_out_width + fc_input_size, fc_hidden_dim)] \
                        + [nn.Sequential(nn.ReLU(), nn.Linear(fc_hidden_dim, fc_hidden_dim)) for _ in range(num_fc_layers - 2)] \
                        + [nn.Sequential(nn.ReLU(), nn.Linear(fc_hidden_dim, 1))]
        elif num_fc_layers == 1:
            fc_layers = [nn.Linear(conv_out_dim * conv_out_height * conv_out_width + fc_input_size, 1)]
        else:
            raise ValueError('invalid num_fc_layers')
        self.fully_connected = nn.Sequential(*fc_layers)

    def forward(self, conv_input, fc_input, conv_hidden_states=None):
        batch_size = fc_input.shape[0]
        lstm_out, hidden = self.conv_lstm(conv_input, hidden_state=conv_hidden_states)
        fc_in_conv = self.dropout(lstm_out[-1][:,-1,:,:,:]).reshape((batch_size, -1))
        fc_in = torch.cat([fc_in_conv, fc_input], dim=1)
        
        return self.fully_connected(fc_in), hidden