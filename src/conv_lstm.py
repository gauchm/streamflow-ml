import torch.nn as nn
from torch.autograd import Variable
import torch
from src import utils


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell.
    
    Adapted from: https://github.com/ndrplz/ConvLSTM_pytorch
    
    Attributes:
        height, width: height and width of input
        input_dim: Number of input channels
        hidden_dim: Number of channels in the hidden state
        kernel_size: Size of the convolutional kernel
        bias (bool): Whether to include a bias
        conv: 2d-Convolution
    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """Initializes the ConvLSTM cell.
        
        Args:
            input_size (int, int): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int): Number of channels of hidden state.
            kernel_size (int, int): Size of the convolutional kernel.
            bias (bool): Whether to add the bias.
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
        """Performs a forward pass on the cell.
        
        Args:
            input_tensor: Cell input of shape (batch_size, input_dim, height, width)
            cur_state: Recurrent cell states of shapes (batch_size, hidden_dim, height, width)
        Returns:
            Tuple of new cell states.
        """
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
        """Creates initial hidden states."""
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device, requires_grad=True),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device, requires_grad=True))


class ConvLSTM(nn.Module):
    """Convolutional LSTM
    
    Adapted from: https://github.com/ndrplz/ConvLSTM_pytorch
    
    Attributes:
        height, width: height and width of input
        input_dim: Number of input channels
        hidden_dim: Number of channels in each hidden state
        kernel_size: Size of each convolutional kernel
        bias (bool): Whether to include a bias
        num_layers (int): Number of layers.
        batch_first (bool): Determines if batch is first or second dimension.
        return_all_layers: Whether all layer outputs will be returned, or just the last layer's.
        pooling: Whether to perform max-pooling after each layer
        cell_list: List of modules, containing the ConLSTMCells and potentially MaxPool2d modules.
    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, pooling=False,
                 batch_first=False, bias=True, return_all_layers=False):
        """Initializes the ConvLSTM.
        
        Args:
            input_size (int, int): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int or list(int)): Number of channels of hidden states.
            kernel_size ((int, int) or list((int, int))): Size of the convolutional kernels.
            num_layers (int): Number of ConvLSTM layers.
            pooling (bool or list(bool)): Whether to perform max-pooling after each layer.
            batch_first (bool): If True, expects input as (batch, timesteps, channels, height, width). Else, dim 0 and 1 are swapped.
            bias (bool): Whether to add the bias.
            return_all_layers (bool): Whether to return the output of all layers or just the last.
        """
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
        """Performs a forward pass on the input.
        
        Args:
            input_tensor: 5d-Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w), depending on self.batch_first
            hidden_state (list or None): If list of (h,c)-tuples, will use as LSTM states for the LSTM layers. If None, will initialize states on each batch.
        Returns:
            last_state_list: List of last state for all layers
            layer_output: list of last output for all layers (if return_all_layers) or list of length 1 with the output of the last layer (if not return_all_layers).
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

            last_state_list.append((h.clone().detach().requires_grad_(True), c.clone().detach().requires_grad_(True)))  # detach to stop BPTT between batches
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
    """ConvLSTM for regression, with fully-connected layers at the end.
    
    Allows to add non-time-series inputs, feeding them into the fc-layers together with the ConvLSTM output.
    
    Attributes:
        conv_lstm: Convolutional LSTM
        dropout: Dropout layer to be applied on the ConvLSTM output
        pooling: Whether to apply Max-Pooling in the ConvLSTM.
        fully_connected: The fc-layers.
    """
    def __init__(self, conv_input_size, fc_input_size, conv_input_dim, conv_hidden_dim, kernel_size, 
                 num_conv_layers, dropout, num_fc_layers, fc_hidden_dim, pooling, batch_first=False, fc_activation=nn.Sigmoid):
        """Initializes the network.
        
        Args:
            conv_input_size (int, int): Height and width of the timeseries input tensor as (height, width).
            fc_input_size (int): Dimension of additional inputs to the fully-connected layers.
            conv_input_dim (int): Number of channels of the timeseries input tensor.
            conv_hidden_dim (int or list(int)): Number of channels of the ConvLSTM hidden states.
            kernel_size ((int, int) or list((int, int))): Size of the ConLSTM convolutional kernels.
            num_conv_layers (int): Number of ConvLSTM layers.
            num_fc_layers (int): Number of fully-connected layers after the ConvLSTM.
            fc_hidden_dim (int): Size of the hidden fc layers
            pooling (list(bool)): Whether to perform max-pooling after each layer.
            batch_first (bool): If True, expects ConvLSTM input as (batch, timesteps, channels, height, width). Else, dim 0 and 1 are swapped. fc_input always has batch in dim 0.
            fc_activation: Activation function to use in the fully-connected layers.
        """
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
                        + [nn.Sequential(fc_activation(), nn.Linear(fc_hidden_dim, fc_hidden_dim)) for _ in range(num_fc_layers - 2)] \
                        + [nn.Sequential(fc_activation(), nn.Linear(fc_hidden_dim, 1))]
        elif num_fc_layers == 1:
            fc_layers = [nn.Linear(conv_out_dim * conv_out_height * conv_out_width + fc_input_size, 1)]
        else:
            raise ValueError('invalid num_fc_layers')
        self.fully_connected = nn.Sequential(*fc_layers)

    def forward(self, conv_input, fc_input, conv_hidden_states=None):
        """Performs a forward pass on the input.
        
        Args:
            conv_input: time-series input to the ConvLSTM
            fc_input: Additional non-time-series input to the fc-layers
            conv_hidden_states: If provided, will use these hidden states in the ConvLSTM. Else, will initialize on each batch.
        Returns:
            Prediction of the fully-connected layer (tensor of shape (batch_size)),
            New hidden ConvLSTM states
        """
        batch_size = fc_input.shape[0]
        lstm_out, hidden = self.conv_lstm(conv_input, hidden_state=conv_hidden_states)
        fc_in_conv = self.dropout(lstm_out[-1][:,-1,:,:,:]).reshape((batch_size, -1))
        fc_in = torch.cat([fc_in_conv, fc_input], dim=1)
        
        return self.fully_connected(fc_in), hidden
    
    
class ConvLSTMLSTMRegression(nn.Module):
    """ConvLSTM for regression, with LSTM layers (non-convolutional) and a linear layer at the end.
    
    Attributes:
        conv_lstm: Convolutional LSTM
        lstm: The LSTM layers
        linear: Final linear layer, generating the predictions.
        dropout: Dropout layer to be applied on the ConvLSTM output
        pooling: Whether to apply Max-Pooling in the ConvLSTM.
        num_lstm_layers, lstm_hidden_dim: Number and size of LSTM layers
    """
    def __init__(self, conv_input_size, lstm_input_size, conv_input_dim, conv_hidden_dim, kernel_size, 
                 num_conv_layers, dropout, num_lstm_layers, lstm_hidden_dim, pooling):
        """Initializes the network.
        
        Args:
            conv_input_size (int, int): Height and width of the timeseries input tensor as (height, width).
            lstm_input_size (int): Dimension of additional inputs to the fully-connected layers.
            conv_input_dim (int): Number of channels of the timeseries input tensor.
            conv_hidden_dim (int or list(int)): Number of channels of the ConvLSTM hidden states.
            kernel_size ((int, int) or list((int, int))): Size of the ConLSTM convolutional kernels.
            num_conv_layers (int): Number of ConvLSTM layers.
            dropout: Probability of dropout after the ConvLSTM.
            num_lstm_layers (int): Number of LSTM layers after the ConvLSTM.
            lstm_hidden_dim (int): Size of the hidden LSTM layers
            pooling (list(bool)): Whether to perform max-pooling after each layer.
        """
        super(ConvLSTMLSTMRegression, self).__init__()
        self.conv_lstm = ConvLSTM(conv_input_size, conv_input_dim, conv_hidden_dim, kernel_size, 
                                  num_conv_layers, pooling=pooling, batch_first=True)
        self.dropout = nn.Dropout2d(p=dropout)
        self.pooling = pooling
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        
        conv_out_height = conv_input_size[0]
        conv_out_width = conv_input_size[1]
        for i in range(num_conv_layers):
            if pooling[i]:
                conv_out_height -= 2 * (kernel_size[0] // 2)
                conv_out_width -= 2 * (kernel_size[1] // 2)
        conv_out_dim = conv_hidden_dim[-1] if isinstance(conv_hidden_dim, list) else conv_hidden_dim
        
        self.lstm = nn.LSTM(conv_out_dim * conv_out_height * conv_out_width + lstm_input_size, lstm_hidden_dim, num_lstm_layers, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, conv_input, lstm_input, conv_hidden_states=None, lstm_hidden_states=None):
        """Performs a forward pass on the input.
        
        Args:
            conv_input: time-series input to the ConvLSTM
            lstm_input: Additional input to the LSTM layers
            conv_hidden_states: If provided, will use these hidden states in the ConvLSTM. Else, will initialize on each batch.
            lstm_hidden_states: If provided, will use these hidden states in the LSTM. Else, will initialize on each batch.
        Returns:
            Prediction of the linear layer (tensor of shape (batch_size)),
            New hidden ConvLSTM states,
            New hidden LSTM states.
        """
        batch_size = lstm_input.shape[0]
        conv_out, conv_hidden = self.conv_lstm(conv_input, hidden_state=conv_hidden_states)
        
        lstm_in_conv = self.dropout(conv_out[-1][:,:,:,:,:]).reshape((batch_size, conv_out[-1].shape[1], -1))
        lstm_in = torch.cat([lstm_in_conv, torch.unsqueeze(lstm_input, 1).repeat(1,lstm_in_conv.shape[1],1)], dim=2)
        
        if lstm_hidden_states is None:
            lstm_hidden = (torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim).to(lstm_input.device),
                           torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim).to(lstm_input.device))
        else:
            lstm_hidden = lstm_hidden_states
        lstm_out, lstm_hidden = self.lstm(lstm_in, lstm_hidden)
        return self.linear(lstm_out[:,-1,:]), conv_hidden, lstm_hidden
    
    
class ConvLSTMGridWithGeophysicalInput(nn.Module):
    """ConvLSTM for regression, with convolutional (non-LSTM) layers at the end.
    
    Allows to add non-time-series spatial inputs, feeding them into the conv-layers together with the ConvLSTM output.
    
    Attributes:
        feed_timesteps (int): Number of ConvLSTM output steps to feed into the conv-layers. The last feed_timesteps layers will be fed into the first conv-layer as individual channels.
        conv_lstm: Convolutional LSTM
        dropout: Dropout layer to be applied after each Conv-layer.
        upsample: TransposedConvolution2d layer to upsample the ConvLSTM output to the geophysical data dimension. 
                  This way, the ConvLSTM can operate on the coarse RDRS grid, and the Conv-layers can operate on the fine geophysical grid.
        conv_out: List of convolutional (non-LSTM) layers, together with activation, BatchNorm2d and Dropout layers
    """
    def __init__(self, input_size, input_dim, geophysical_dim, convlstm_hidden_dim, conv_hidden_dim, convlstm_kernel_size, conv_kernel_size, num_convlstm_layers, num_conv_layers, conv_activation, dropout=0.0, geophysical_size=None, feed_timesteps=1):
        """Initializes the network.
        
        Args:
            input_size (int, int): Height and width of the timeseries input tensor as (height, width).
            input_dim (int): Number of channels of the timeseries input tensor.
            geophysical_dim (int): Number of channels of the non-timeseries input tensor.
            convlstm_hidden_dim (list(int)): Number of channels of the ConvLSTM hidden states.
            conv_hidden_dim (list(int)): Number of channels of the convolutional hidden states.
            convlstm_kernel_size (list((int, int))): Size of the ConLSTM convolutional kernels.
            conv_kernel_size (list((int, int))): Size of the convolutional kernels.
            num_convlstm_layers (int): Number of ConvLSTM layers.
            num_conv_layers (int): Number of convolutional layers.
            conv_activation: Activation function to use in the convolutional layers.
            dropout: Probability of dropout after each convolutional layer.
            geophysical_size (int, int): height and width of the non-timeseries input.
            feed_timesteps (int): Number of ConvLSTM output steps to feed into the conv-layers. The last feed_timesteps layers will be fed into the first conv-layer as individual channels.
        """
        super(ConvLSTMGridWithGeophysicalInput, self).__init__()
        self.feed_timesteps = feed_timesteps
        self.conv_lstm = ConvLSTM((input_size[0], input_size[1]), input_dim, convlstm_hidden_dim, convlstm_kernel_size, num_convlstm_layers, batch_first=True)
        self.dropout = nn.Dropout2d(p=dropout)
        self.upsample = nn.Identity() 
        if geophysical_size is not None and (input_size[0] != geophysical_size[0] or input_size[1] != geophysical_size[1]):
            stride = geophysical_size[0] // input_size[0], geophysical_size[1] // input_size[1]
            upsample_kernel = (geophysical_size[0] + stride[0] * (1 - input_size[0]), geophysical_size[1] + stride[1] * (1 - input_size[1]))
            self.upsample = nn.ConvTranspose2d(convlstm_hidden_dim[-1] * feed_timesteps, convlstm_hidden_dim[-1] * feed_timesteps, 
                                               upsample_kernel, stride=stride)
        if num_conv_layers == 1:
            pad = conv_kernel_size[0][0] // 2, conv_kernel_size[0][1] // 2
            self.conv_out = nn.Conv2d(convlstm_hidden_dim[-1] * feed_timesteps + geophysical_dim, 1, conv_kernel_size[0], padding=pad)
        else:
            pad = conv_kernel_size[0][0] // 2, conv_kernel_size[0][1] // 2
            conv_layers = [nn.BatchNorm2d(convlstm_hidden_dim[-1] * feed_timesteps + geophysical_dim), 
                           nn.Conv2d(convlstm_hidden_dim[-1] * feed_timesteps + geophysical_dim, conv_hidden_dim[0], 
                                     conv_kernel_size[0], padding=pad),
                           nn.Dropout(p=dropout), conv_activation()]
            for i in range(1, num_conv_layers - 1):
                pad = conv_kernel_size[i][0] // 2, conv_kernel_size[i][1] // 2
                conv_layers.append(nn.BatchNorm2d(conv_hidden_dim[i-1]))
                conv_layers.append(nn.Conv2d(conv_hidden_dim[i-1], conv_hidden_dim[i], conv_kernel_size[i], padding=pad))
                conv_layers.append(conv_activation())
                conv_layers.append(nn.Dropout2d(p=dropout))
            pad = conv_kernel_size[-1][0] // 2, conv_kernel_size[-1][1] // 2
            conv_layers.append(nn.BatchNorm2d(conv_hidden_dim[-1]))
            conv_layers.append(nn.Conv2d(conv_hidden_dim[-1], 1, conv_kernel_size[-1], padding=pad))
            self.conv_out = nn.Sequential(*conv_layers)
        
    def forward(self, input_tensor, geophysics_tensor, hidden_state=None):
        """Performs a forward pass on the input.
        
        Args:
            input_tensor: time-series input to the ConvLSTM
            geophyiscs_tensor: Additional non-timeseries input to the convolutional layers
            hidden_state: If provided, will use these hidden states in the ConvLSTM. Else, will initialize on each batch.
        Returns:
            Prediction of the last convolutional layer (tensor of shape (batch_size, height, width)),
            New hidden ConvLSTM states.
        """
        convlstm_out, hidden = self.conv_lstm(input_tensor, hidden_state=hidden_state)
        # last feed_timesteps outputs of last layer
        convlstm_out = convlstm_out[-1][:,-self.feed_timesteps:,:,:,:].reshape((convlstm_out[-1].shape[0],-1,*convlstm_out[-1].shape[3:]))
        convlstm_out = self.dropout(convlstm_out)
        upsampled_convlstm_out = self.upsample(convlstm_out)
        conv_in = torch.cat([upsampled_convlstm_out, geophysics_tensor], dim=1)
        return self.conv_out(conv_in)[:,0,:,:], hidden