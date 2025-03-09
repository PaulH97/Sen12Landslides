from torchvision import models
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
import torch.nn.functional as F
import torch.nn.init as init

import torch
import torch.nn as nn

class FCN_CRNN(nn.Module):
    def __init__(self, 
        num_classes, 
        num_channels, 
        timeseries_len, 
        img_res, 
        train_stage, 
        attn_channels,
        cscl_win_size,
        cscl_win_dilation,
        cscl_win_stride,
        attn_groups,
        crnn_model_name, 
        bidirectional, 
        avg_hidden_states,
        pretrained,
        early_feats, 
        br_layer
    ):
        super(FCN_CRNN, self).__init__()
        self.num_channels = num_channels
        self.fcn_input_size = (timeseries_len, self.num_channels, img_res, img_res)
        self.crnn_input_size = (timeseries_len, 128, img_res, img_res)
        self.hidden_dims = 128
        self.lstm_kernel_sizes = (3, 3)
        self.conv_kernel_size = 3
        self.lstm_num_layers = 1
        self.avg_hidden_states = avg_hidden_states
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.early_feats = early_feats
        self.use_planet = False
        self.resize_planet = False
        self.num_bands_dict = {'planet': 0, 's1': 0, 's2': self.num_channels, 'all': self.num_channels}
        self.main_attn_type = 'None'
        self.attn_dims = 32
        self.main_crnn = False
        self.enc_crnn = False
        self.enc_attn = False
        self.enc_attn_type = None
        self.crnn_model_name = crnn_model_name
        self.pretrained = pretrained
        self.processed_feats = {'main': None, 'enc4': None, 'enc3': None, 'enc2': None, 'enc1': None }
        self.stage = train_stage
        self.br_layer = br_layer

        # Create UNet encoder/decoder and force them to float32
        if not self.early_feats:
            self.fcn = make_UNet_model(n_class=self.crnn_input_size[1], 
                                       num_bands_dict=self.num_bands_dict, 
                                       late_feats_for_fcn=True,
                                       pretrained=self.pretrained, 
                                       use_planet=self.use_planet, 
                                       resize_planet=self.resize_planet).float()
        else:
            self.fcn_enc = make_UNetEncoder_model(self.num_bands_dict, 
                                                  use_planet=self.use_planet, 
                                                  resize_planet=self.resize_planet, 
                                                  pretrained=self.pretrained).float()
            self.fcn_dec = make_UNetDecoder_model(self.num_classes, 
                                                  late_feats_for_fcn=False,  
                                                  use_planet=self.use_planet, 
                                                  resize_planet=self.resize_planet).float()

        # Instantiate CRNN module(s)
        if self.crnn_model_name == "gru":
            if self.early_feats:
                self.crnn = CGRUSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes,
                                          self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1],
                                          self.bidirectional, self.avg_hidden_states)
            else:
                self.crnn = CGRUSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes,
                                          self.conv_kernel_size, self.lstm_num_layers, self.num_classes,
                                          self.bidirectional, self.avg_hidden_states)
            # Force CRNN parameters to float32
            self.crnn = self.crnn.float()
        elif self.crnn_model_name == "clstm":
            self.attns = self.get_attns()  # get_attns should return modules in float32
            self.crnns = self.get_crnns()  # get_crnns should return modules in float32
            self.final_convs = self.get_final_convs()  # these are already nn.Conv2d (will be float32 by default)

        # Setup attention mechanism for specific training stages and force to float32
        if self.stage in [0, 3, 4]:
            self.attn_channels = attn_channels
            self.kernel_size = cscl_win_size
            self.dilation = cscl_win_dilation
            self.stride = cscl_win_stride
            self.groups = attn_groups
            self.norm_emb = False
            self.attn_sim = ContextSelfSimilarity(
                in_channels=self.hidden_dims, 
                attn_channels=self.attn_channels, 
                kernel_size=self.kernel_size,
                stride=self.stride, 
                dilation=self.dilation, 
                groups=self.groups, 
                bias=False, 
                norm_emb=self.norm_emb
            ).float()
        elif self.stage in [2]:
            self.main_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.num_classes,
                                            kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size - 1) / 2))
            # Conv2d layers are float32 by default

    def forward(self, input_tensor, hres_inputs=None):
        # Convert the input tensor to the model's dtype
        target_dtype = next(self.parameters()).dtype
        input_tensor = input_tensor.to(target_dtype)
        batch, timestamps, bands, rows, cols = input_tensor.size()
        fcn_input = input_tensor.view(batch * timestamps, bands, rows, cols)
        fcn_input_hres = None

        # Encode and decode features
        fcn_output = self.fcn(fcn_input, fcn_input_hres)

        # Apply CRNN: reshape and get outputs (assume crnn outputs 5D: [B, T, C, H, W])
        crnn_input = fcn_output.view(batch, timestamps, -1, fcn_output.shape[-2], fcn_output.shape[-1])
        if hasattr(self, "crnn_main") and self.crnn_main(crnn_input) is not None:
            crnn_output_fwd, crnn_output_rev = self.crnn_main(crnn_input)
        else:
            crnn_output_fwd = crnn_input
            crnn_output_rev = None

        # Apply attention (or aggregation) based on model type:
        if self.crnn_model_name == "gru":
            # For GRU, simply use the CRNN output without attention.
            reweighted = crnn_output_fwd
        else:
            reweighted = attn_or_avg(self.attns['main'], self.avg_hidden_states,
                                    crnn_output_fwd, crnn_output_rev, self.bidirectional)

        # If reweighted is 5D ([B, T, C, H, W]), aggregate the time dimension.
        if reweighted.dim() == 5:
            # For example, average over the time dimension:
            reweighted = torch.mean(reweighted, dim=1)  # Resulting shape: [B, C, H, W]

        if self.stage in [0]:
            return self.attn_sim(reweighted)
        elif self.stage in [2]:
            logits = self.main_finalconv(reweighted)
            # Upsample logits from (H_out, W_out) to (img_res, img_res)
            logits = F.interpolate(logits, size=(self.fcn_input_size[2], self.fcn_input_size[3]),
                                mode='bilinear', align_corners=False)
            if self.br_layer:
                logits = self.br(logits)
            return logits

    def get_crnns(self):
        # Initialize CRNN layers and convert them to float32
        self.crnn_main = self.crnn_enc4 = self.crnn_enc3 = self.crnn_enc2 = self.crnn_enc1 = None
        if self.early_feats:
            if self.main_crnn:
                self.crnn_main = CLSTMSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes, 
                                                self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1], 
                                                self.bidirectional, self.avg_hidden_states).float()
            if self.enc_crnn:
                crnn_input0, crnn_input1, crnn_input2, crnn_input3 = self.crnn_input_size
                self.crnn_enc4 = CLSTMSegmenter([crnn_input0, crnn_input1//2, crnn_input2*2, crnn_input3*2], self.hidden_dims, self.lstm_kernel_sizes, 
                                                self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//2, self.bidirectional).float()
                self.crnn_enc3 = CLSTMSegmenter([crnn_input0, crnn_input1//4, crnn_input2*4, crnn_input3*4], self.hidden_dims, self.lstm_kernel_sizes, 
                                                self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//4, self.bidirectional).float()
                if self.use_planet and not self.resize_planet:
                    self.crnn_enc2 = CLSTMSegmenter([crnn_input0, crnn_input1//8, crnn_input2*8, crnn_input3*8], self.hidden_dims, self.lstm_kernel_sizes, 
                                                    self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//8, self.bidirectional).float()
                    self.crnn_enc1 = CLSTMSegmenter([crnn_input0, crnn_input1//16, crnn_input2*16, crnn_input3*16], self.hidden_dims, self.lstm_kernel_sizes, 
                                                    self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//16, self.bidirectional).float()
        else:
            self.crnn_main = CLSTMSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes, 
                                            self.conv_kernel_size, self.lstm_num_layers, self.num_classes, self.bidirectional).float()
        self.crnns = { 'main': self.crnn_main, 'enc4': self.crnn_enc4, 'enc3': self.crnn_enc3, 'enc2': self.crnn_enc2, 'enc1': self.crnn_enc1 }
        return self.crnns

    def get_attns(self):
        # Initialize attention layers and force them to float32
        self.attn_enc4 = self.attn_enc3 = self.attn_enc2 = self.attn_enc1 = None
        if self.early_feats:
            if self.enc_attn:
                self.attn_enc4 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims).float() 
                self.attn_enc3 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims).float()
                if self.use_planet and not self.resize_planet:
                    self.attn_enc2 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims).float()
                    self.attn_enc1 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims).float()
        self.attn_main = ApplyAtt(self.main_attn_type, self.hidden_dims, self.attn_dims).float()
        self.attns = { 'main': self.attn_main, 'enc4': self.attn_enc4, 'enc3': self.attn_enc3, 'enc2': self.attn_enc2, 'enc1': self.attn_enc1 } 
        return self.attns

    def get_final_convs(self):
        # Initialize final convolution layers (Conv2d layers are float32 by default)
        self.enc4_finalconv = self.enc3_finalconv = self.enc2_finalconv = self.enc1_finalconv = None
        if self.early_feats:
            self.main_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1], kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
            if self.enc_crnn:
                self.enc4_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//2, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
                self.enc3_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//4, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
                if self.use_planet and not self.resize_planet: 
                    self.enc2_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//8, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
                    self.enc1_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//16, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
        else:
            self.main_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.num_classes, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
        self.final_convs = { 'main': self.main_finalconv, 'enc4': self.enc4_finalconv, 'enc3': self.enc3_finalconv, 'enc2': self.enc2_finalconv, 'enc1': self.enc1_finalconv}
        return self.final_convs

def make_UNet_model(n_class, num_bands_dict, late_feats_for_fcn=False, pretrained=True, use_planet=False, resize_planet=False):
    model = UNet(n_class, num_bands_dict, late_feats_for_fcn, use_planet, resize_planet)
    
    if pretrained:
        pre_trained = models.vgg13(pretrained=True)
        pre_trained_features = list(pre_trained.features)
        model.unet_encode.enc3.encode[3] = pre_trained_features[2]
        model.unet_encode.enc4.encode[0] = pre_trained_features[5]
        model.unet_encode.enc4.encode[3] = pre_trained_features[7]
        model.unet_encode.center[0] = pre_trained_features[10]
    
    model = model
    return model

def make_UNetEncoder_model(num_bands_dict, use_planet=True, resize_planet=False, pretrained=True):
    model = UNet_Encode(num_bands_dict, use_planet, resize_planet)
    
    if pretrained:
       # TODO: Why are pretrained weights from vgg13? 
        pre_trained = models.vgg13(pretrained=True)
        pre_trained_features = list(pre_trained.features)

        model.enc3.encode[3] = pre_trained_features[2] #  64 in,  64 out
        model.enc4.encode[0] = pre_trained_features[5] #  64 in, 128 out
        model.enc4.encode[3] = pre_trained_features[7] # 128 in, 128 out
        model.center[0] = pre_trained_features[10]     # 128 in, 256 out

    model = model()
    return model

def make_UNetDecoder_model(n_class, late_feats_for_fcn, use_planet, resize_planet):
    model = UNet_Decode(n_class, late_feats_for_fcn, use_planet, resize_planet)
    model = model()
    return model

class CLSTMSegmenter(nn.Module):
    """ CLSTM followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, 
                 lstm_num_layers, num_outputs, bidirectional, with_pred=False, 
                 avg_hidden_states=None, attn_type=None, d=None, r=None, dk=None, dv=None): 

        super(CLSTMSegmenter, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.with_pred = with_pred        

        if self.with_pred:
            self.avg_hidden_states = avg_hidden_states
            self.attention = ApplyAtt(attn_type, hidden_dims, d=d, r=r, dk=dk, dv=dv) 
            self.final_conv = nn.Conv2d(in_channels=hidden_dims, 
                                        out_channels=num_outputs, 
                                        kernel_size=conv_kernel_size, 
                                        padding=int((conv_kernel_size-1)/2)) 
            self.logsoftmax = nn.LogSoftmax(dim=1)
        
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.clstm_rev = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers, bidirectional)
        
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        initialize_weights(self)
       
    def forward(self, inputs):

        layer_outputs, last_states = self.clstm(inputs)
    
        rev_layer_outputs = None
        if self.bidirectional:
            rev_inputs = torch.flip(inputs, dims=[1])
            rev_layer_outputs, rev_last_states = self.clstm_rev(rev_inputs)

        if self.with_pred:
            # Apply attention
            reweighted = attn_or_avg(self.attention, self.avg_hidden_states, layer_outputs, rev_layer_outputs, self.bidirectional)

            # Apply final conv
            scores = self.final_conv(reweighted)
            output = self.logsoftmax(scores)
            return output
        else:
            return layer_outputs, rev_layer_outputs

class CLSTM(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_dims, 
                 kernel_sizes, 
                 lstm_num_layers, 
                 batch_first=True, 
                 bias=True, 
                 return_all_layers=False,
                 var_length=False):
        """
           Args:
                input_size - (tuple) should be (time_steps, channels, height, width)
                hidden_dims - (list of ints) number of filters to use per layer
                kernel_sizes - lstm kernel sizes
                lstm_num_layers - (int) number of stacks of ConvLSTM units per step
        """

        super(CLSTM, self).__init__()
        (self.num_timesteps, self.start_num_channels, self.height, self.width) = input_size
        self.lstm_num_layers = lstm_num_layers
        self.bias = bias
        self.var_length = var_length
        
        if isinstance(kernel_sizes, list):
            if len(kernel_sizes) != lstm_num_layers and len(kernel_sizes) == 1:
                self.kernel_sizes = kernel_sizes * lstm_num_layers
            else:
                self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * lstm_num_layers      
        
        if isinstance(hidden_dims, list):
            if len(hidden_dims) != lstm_num_layers and len(hidden_dims) == 1:
                self.hidden_dims = hidden_dims * lstm_num_layers
            else:
                self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = [hidden_dims] * lstm_num_layers       
        
        self.init_hidden_state = self._init_hidden()
        self.init_cell_state = self._init_hidden()
        #print(self.init_cell_state)

        cell_list = []
        for i in range(self.lstm_num_layers):
            cur_input_dim = self.start_num_channels if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvLSTMCell(input_dim = cur_input_dim,
                                          hidden_dim = self.hidden_dims[i],
                                          num_timesteps = self.num_timesteps,
                                          kernel_size = self.kernel_sizes[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        initialize_weights(self)

    def forward(self, input_tensor, hidden_state=None):

        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.lstm_num_layers):
            # double check that this is right? i.e not resetting every time to 0?
            #print(self.init_hidden_state)
            h, c = self.init_hidden_state[layer_idx], self.init_cell_state[layer_idx]
            h = h.expand(input_tensor.size(0), h.shape[1], h.shape[2], h.shape[3])
            c = c.expand(input_tensor.size(0), c.shape[1], c.shape[2], c.shape[3])
            output_inner_layers = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c], timestep=t)

                output_inner_layers.append(h)

            layer_output = torch.stack(output_inner_layers, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # TODO: Rework this so that we concatenate all the internal outputs as features for classification
        # Just take last output for prediction
        layer_outputs = layer_output_list[-1]
        last_states = last_state_list[-1:]
        
        return layer_outputs, last_states

    def _init_hidden(self):
        init_states = []
        for i in range(self.lstm_num_layers):
            init_states.append(nn.Parameter(torch.zeros(1, self.hidden_dims[i], self.width, self.height)))
        return init_states  # nn.ParameterList(init_states)

class ConvLSTMCell(nn.Module):
    """
        ConvLSTM Cell based on Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
        arXiv: https://arxiv.org/abs/1506.04214

        Implementation based on stefanopini's at https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    """
    def __init__(self, input_dim, hidden_dim, num_timesteps, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
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

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
       
        self.h_conv = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
       
        self.input_conv = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        

        self.h_norm = RecurrentNorm2d(4 * self.hidden_dim, self.num_timesteps)
        self.input_norm = RecurrentNorm2d(4 * self.hidden_dim, self.num_timesteps)
        self.cell_norm = RecurrentNorm2d(self.hidden_dim, self.num_timesteps)
        
        initialize_weights(self)

    def forward(self, input_tensor, cur_state, timestep):
        
        h_cur, c_cur = cur_state
        # BN over the outputs of these convs
        combined_conv = self.h_norm(self.h_conv(h_cur), timestep) + self.input_norm(self.input_conv(input_tensor), timestep)
 
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        # BN over the tanh
        h_next = o * self.cell_norm(torch.tanh(c_next), timestep)
        
        
        return h_next, c_next

class CGRUSegmenter(nn.Module):
    """ cgru followed by conv for segmentation output
    """
    def __init__(self, input_size, hidden_dims, gru_kernel_sizes, 
                 conv_kernel_size, gru_num_layers, num_classes, bidirectional, early_feats):

        super().__init__()
        self.early_feats = early_feats

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.cgru = CGRU(input_size, hidden_dims, gru_kernel_sizes, gru_num_layers)
        self.bidirectional = bidirectional
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        self.logsoftmax = nn.LogSoftmax(dim=1)
        initialize_weights(self)

    def forward(self, inputs):
        layer_output_list, last_state_list = self.cgru(inputs)
        final_state = last_state_list[0]
        if self.bidirectional:
            rev_inputs = torch.tensor(inputs.cpu().detach().numpy()[::-1].copy(), dtype=torch.float32)()
            rev_layer_output_list, rev_last_state_list = self.cgru(rev_inputs)
            final_state = torch.cat([final_state, rev_last_state_list[0][0]], dim=1)
        scores = self.conv(final_state)

        output = scores if self.early_feats else self.logsoftmax(scores)
        return output
    
class CGRU(nn.Module):

    def __init__(self, input_size, hidden_dims, kernel_sizes, gru_num_layers, batch_first=True, bias=True, return_all_layers=False):
        """
           Args:
                input_size - (tuple) should be (time_steps, channels, height, width)
                hidden_dims - (list of ints) number of filters to use per layer
                kernel_sizes - lstm kernel sizes
                gru_num_layers - (int) number of stacks of ConvLSTM units per step
        """

        super(CGRU, self).__init__()
        (self.num_timesteps, self.start_num_channels, self.height, self.width) = input_size

        self.gru_num_layers = gru_num_layers
        self.bias = bias
        
        if isinstance(kernel_sizes, list):
            if len(kernel_sizes) != gru_num_layers and len(kernel_sizes) == 1:
                self.kernel_sizes = kernel_sizes * gru_num_layers
            else:
                self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * gru_num_layers      
        
        if isinstance(hidden_dims, list):
            if len(hidden_dims) != gru_num_layers and len(hidden_dims) == 1:
                self.hidden_dims = hidden_dims * gru_num_layers
            else:
                self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = [hidden_dims] * gru_num_layers       
        
        self.init_hidden_state = self._init_hidden()
        
        cell_list = []
        for i in range(self.gru_num_layers):
            cur_input_dim = self.start_num_channels if i == 0 else self.hidden_dims[i-1]

            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim = cur_input_dim,
                                         hidden_dim = self.hidden_dims[i],
                                         num_timesteps = self.num_timesteps,
                                         kernel_size = self.kernel_sizes[i],
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        initialize_weights(self)

    def forward(self, input_tensor, hidden_state=None):

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.gru_num_layers):
            # double check that this is right? i.e not resetting every time to 0?
            h = self.init_hidden_state[layer_idx]
            h = h.expand(input_tensor.size(0), h.shape[1], h.shape[2], h.shape[3])()
            output_inner_layers = []
            
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=h, timestep=t)

                output_inner_layers.append(h)

            layer_output = torch.stack(output_inner_layers, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append(h)

        # TODO: Rework this so that we concatenate all the internal outputs as features for classification
        # Just take last output for prediction
        layer_output_list = layer_output_list[-1:]
        last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self):
        init_states = []
        for i in range(self.gru_num_layers):
            init_states.append(nn.Parameter(torch.zeros(1, self.hidden_dims[i], self.width, self.height)))
        return nn.ParameterList(init_states)

class RecurrentNorm2d(nn.Module):
    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(RecurrentNorm2d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            # no bias term as described in the paper
            self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            # initialize to .1 as advocated in the paper
            self.weight.data = torch.ones(self.num_features) * .1
            
    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

class ConvGRUCell(nn.Module):
    """
        
    """
    def __init__(self, input_size, input_dim, hidden_dim, num_timesteps, kernel_size, bias):
        """
        Initialize BiConvRNN cell.
        
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

        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.h_conv = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.input_conv = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.W_h = nn.Conv2d(in_channels=self.hidden_dim,
                             out_channels=self.hidden_dim,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)
        
        self.U_h = nn.Conv2d(in_channels=self.input_dim,
                             out_channels=self.hidden_dim,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)
        
        self.h_norm = RecurrentNorm2d(2 * self.hidden_dim, self.num_timesteps)
        self.input_norm = RecurrentNorm2d(2 * self.hidden_dim, self.num_timesteps)
        
        initialize_weights(self)

    def forward(self, input_tensor, cur_state, timestep):
        # TODO: should not have to call cuda here, figure out where this belongs
        input_tensor()
        # BN over the outputs of these convs
        
        combined_conv = self.h_norm(self.h_conv(cur_state), timestep) + self.input_norm(self.input_conv(input_tensor), timestep)
        
        u_t, r_t = torch.split(combined_conv, self.hidden_dim, dim=1) 
        u_t = torch.sigmoid(u_t)
        r_t = torch.sigmoid(r_t)
        h_tilde = torch.tanh(self.W_h(r_t * cur_state) + self.U_h(input_tensor))
        h_next = (1 - u_t) * h_tilde + u_t * h_tilde
        
        return h_next

class VectorAtt(nn.Module):

    def __init__(self, hidden_dim_size):
        """
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width)
            Returns reweighted hidden states.
        """
        super(VectorAtt, self).__init__()
        self.linear = nn.Linear(hidden_dim_size, 1, bias=False)
        nn.init.constant_(self.linear.weight, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states, lengths=None):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous() # puts channels last
        weights = self.softmax(self.linear(hidden_states))
        b, t, c, h, w = weights.shape
        if lengths is not None: #TODO: gives backprop bug
            for i, length in enumerate(lengths):
                weights[i, t:] *= 0
        reweighted = weights * hidden_states
        return reweighted.permute(0, 1, 4, 2, 3).contiguous()

class TemporalAtt(nn.Module):

    def __init__(self, hidden_dim_size, d, r):
        """
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width)
            Returns reweighted timestamps. 

            Implementation based on the following blog post: 
            https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69
        """
        super(TemporalAtt, self).__init__()
        self.w_s1 = nn.Linear(in_features=hidden_dim_size, out_features=d, bias=False) 
        self.w_s2 = nn.Linear(in_features=d, out_features=r, bias=False) 
        nn.init.constant_(self.w_s1.weight, 1)
        nn.init.constant_(self.w_s2.weight, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous()
        z1 = self.tanh(self.w_s1(hidden_states))
        attn_weights = self.softmax(self.w_s2(z1))
        reweighted = attn_weights * hidden_states
        reweighted = reweighted.permute(0, 1, 4, 2, 3).contiguous()
        return reweighted

class SelfAtt(nn.Module):
    def __init__(self, hidden_dim_size, dk, dv):
        """
            Self attention.
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width) 

            Implementation based on self attention in the following paper: 
            https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
        """
        super(SelfAtt, self).__init__()
        self.dk = dk
        self.dv = dv

        self.w_q = nn.Linear(in_features=hidden_dim_size, out_features=dk, bias=False)
        self.w_k = nn.Linear(in_features=hidden_dim_size, out_features=dk, bias=False)
        self.w_v = nn.Linear(in_features=hidden_dim_size, out_features=dv, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous()
        nb, nt, nr, nc, nh = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        queries = self.w_q(hidden_states)
        keys = self.w_k(hidden_states)
        values = self.w_v(hidden_states)
        
        attn = torch.mm(self.softmax(torch.mm(queries, torch.transpose(keys, 0, 1)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float)())), values)      
        attn = attn.view(nb, nt, nr, nc, -1)
        attn = attn.permute(0, 1, 4, 2, 3).contiguous() 
        return attn

class ApplyAtt(nn.Module):
    def __init__(self, attn_type, hidden_dim_size, attn_dims):
        super(ApplyAtt, self).__init__()
        if attn_type == 'vector':
            self.attention = VectorAtt(hidden_dim_size)
        elif attn_type == 'temporal':
            self.attention = TemporalAtt(hidden_dim_size, attn_dims['d'], attn_dims['r'])
        elif attn_type == 'self':
            self.attention = SelfAtt(hidden_dim_size, attn_dims['dk'], attn_dims['dv'])
        elif attn_type == 'None':
            self.attention = None
        else:
            raise ValueError('Specified attention type is not compatible')

    def forward(self, hidden_states):
        attn_weighted = self.attention(hidden_states) if self.attention is not None else None
        return attn_weighted

class _EncoderBlock(nn.Module):
    """ U-Net encoder block
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DownSample(nn.Module):
    """ U-Net downsample block
    """
    def __init__(self):
        super(_DownSample, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.downsample(x)

class _DecoderBlock(nn.Module):
    """ U-Net decoder block
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
    
class UNet(nn.Module):
    """Bring the encoder and decoder together into full UNet model
    """
    def __init__(self, num_classes, num_bands_dict, late_feats_for_fcn=False, use_planet=False, resize_planet=False):
        super(UNet, self).__init__()
        self.unet_encode = UNet_Encode(num_bands_dict, use_planet, resize_planet)
        self.unet_decode = UNet_Decode(num_classes, late_feats_for_fcn, use_planet, resize_planet)

    def forward(self, x, hres):
        center1, enc4, enc3, enc2, enc1 = self.unet_encode(x, hres) 
        final = self.unet_decode(center1, enc4, enc3, enc2, enc1)
        return final

class UNet_Encode(nn.Module):
    """ U-Net architecture definition for encoding (first half of the "U")
    """
    def __init__(self, num_bands_dict, use_planet=False, resize_planet=False):
        super(UNet_Encode, self).__init__()

        self.downsample = _DownSample() 
        self.use_planet = use_planet
        self.resize_planet = resize_planet      
 
        self.planet_numbands = num_bands_dict['planet']
        self.s1_numbands = num_bands_dict['s1']
        self.s2_numbands = num_bands_dict['s2']

        feats = 16
        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            enc3_infeats = num_bands_dict['all']
        elif self.use_planet and not self.resize_planet:
            self.enc1_hres = _EncoderBlock(self.planet_numbands, feats)
            self.enc2_hres = _EncoderBlock(feats, feats*2)
            if (self.s1_numbands > 0) or (self.s2_numbands > 0):# and model_name not in ['mi_clstm']:
                self.enc1_lres = _EncoderBlock(self.s1_numbands + self.s2_numbands, feats)
                self.enc2_lres = _EncoderBlock(feats, feats*2)
                enc3_infeats = feats*2 + feats*2
            else: 
                enc3_infeats = feats*2
       
        self.enc3 = _EncoderBlock(enc3_infeats, feats*4)
        self.enc4 = _EncoderBlock(feats*4, feats*8)

        self.center = nn.Sequential(
            nn.Conv2d(feats*8, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True))
        
        initialize_weights(self)

    def forward(self, x, hres):

        if hres is not None: hres = hres()
        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            enc3 = self.enc3(x)
        else:
            if hres is None:
                enc1_hres = self.enc1_hres(x)
            else:
                enc1_lres = self.enc1_lres(x)
                enc2_lres = self.enc2_lres(enc1_lres)
                enc1_hres = self.enc1_hres(hres)
 
            down1_hres = self.downsample(enc1_hres)
            enc2_hres = self.enc2_hres(down1_hres)
            down2 = self.downsample(enc2_hres)

            if hres is not None: 
                down2 = torch.cat((enc2_lres, down2), 1)

            enc3 = self.enc3(down2)

        down3 = self.downsample(enc3)
        enc4 = self.enc4(down3)
        down4 = self.downsample(enc4)
        center1 = self.center(down4)

        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            enc2 = None 
            enc1 = None
        else:
            enc2 = self.downsample(enc2_hres)
            enc1 = self.downsample(self.downsample(enc1_hres))

        return center1, enc4, enc3, enc2, enc1

class UNet_Decode(nn.Module):
    """ U-Net architecture definition for decoding (second half of the "U")
    """
    def __init__(self, num_classes, late_feats_for_fcn, use_planet=False, resize_planet=False):
        super(UNet_Decode, self).__init__()

        self.late_feats_for_fcn = late_feats_for_fcn
        self.use_planet = use_planet
        self.resize_planet = resize_planet
        
        feats = 16
        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            extra_enc_feats = 0
        elif self.use_planet and not self.resize_planet: # else
            extra_enc_feats = feats + feats*2

        self.center_decode = nn.Sequential(
            nn.Conv2d(feats*16, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(feats*16, feats*8, kernel_size=2, stride=2))    
        self.dec4 = _DecoderBlock(feats*16, feats*8, feats*4)
        self.final = nn.Sequential(
            nn.Conv2d(feats*8 + extra_enc_feats, feats*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feats*4, feats*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feats*2, num_classes, kernel_size=3, padding=1),
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)
        initialize_weights(self)

    def forward(self, center1, enc4, enc3, enc2=None, enc1=None):

        # DECODE
        center2 = self.center_decode(center1)
        dec4 = self.dec4(torch.cat([center2, enc4], 1)) 

        if enc2 is not None: # concat earlier highres features
            dec4 = torch.cat([dec4, enc2, enc1], 1)

        final = self.final(torch.cat([dec4, enc3], 1)) 

        if not self.late_feats_for_fcn:
            final = self.logsoftmax(final)
        return final

class ContextSelfSimilarity(nn.Module):
    def __init__(self, in_channels, attn_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 norm_emb=False, sigmoid_sim=False):
        super(ContextSelfSimilarity, self).__init__()
        self.attn_channels = attn_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # assert self.final_stride >= 1.0, "CSSL stride should be geq to dilation rate for special case to work"
        if self.stride >= self.dilation:
            assert self.stride % self.dilation == 0, \
                "CSCL stride should be integer multiple of dilation rate for special case to work"
            self.first_stride = self.dilation
            self.final_stride = int(self.stride / self.dilation)
            self.final_dilation = 1
            self.final_kernel_size = self.kernel_size
        elif self.stride < self.dilation:
            assert self.dilation % self.stride == 0, \
                "CSCL dilation should be integer multiple of stride for special case to work"
            self.first_stride = self.stride
            self.final_stride = 1
            self.final_dilation = int(self.dilation / self.stride)
            self.final_kernel_size = (self.kernel_size - 1) * self.final_dilation + 1

        self.padding = self.final_kernel_size // 2

        self.groups = groups
        self.norm_emb = norm_emb
        self.sigmoid_sim = sigmoid_sim

        assert self.attn_channels % self.groups == 0, "attn_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(attn_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(attn_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, attn_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, attn_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x[:, :, ::self.first_stride, ::self.first_stride]

        batch, channels, height, width = x.shape

        q_out = self.query_conv(x[:, :, ::self.final_stride, ::self.final_stride])
        q_out = q_out.view(batch, self.groups, self.attn_channels // self.groups, height, width, 1)
        q_out = q_out.permute(0, 1, 3, 4, 5, 2)

        k_out = self.key_conv(x)
        k_out = F.pad(k_out, [self.padding, self.padding, self.padding, self.padding], value=0)
        k_out = self.unfold2D(k_out)
        k_out = k_out[:, :, :, :, ::self.final_dilation, ::self.final_dilation]

        k_out_h, k_out_w = k_out.split(self.attn_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        k_out = k_out.contiguous().view(batch, self.groups, self.attn_channels // self.groups, height, width, -1)
        k_out = k_out.permute(0, 1, 3, 4, 2, 5)

        if self.norm_emb:
            q_out = F.normalize(q_out, p=2, dim=5)
            k_out = F.normalize(k_out, p=2, dim=4)
        height1, width1 = q_out.shape[2:4]

        sim = torch.matmul(q_out, k_out)
        sim = sim.sum(dim=1).reshape(batch, height1, width1, self.kernel_size, self.kernel_size)
        if self.sigmoid_sim:
            sim = F.sigmoid(sim)

        return sim

    def unfold2D(self, x):
        return x.unfold(2, size=self.final_kernel_size, step=self.final_stride)\
                .unfold(3, size=self.final_kernel_size, step=self.final_stride)

    def local_agg(self, x):
        batch, channels, height, width = x.size()
        x_win = torch.nn.functional.pad(x, [self.padding, self.padding, self.padding, self.padding]). \
            unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        height1, width1 = x_win.shape[-4:-2]
        x_win = x_win.reshape(batch, channels, height1, width1, self.kernel_size ** 2).permute(0, 2, 3, 4, 1)
        sim = self.forward(x).reshape(batch, height1, width1, 1, self.kernel_size ** 2)

        out = torch.matmul(torch.softmax(sim, dim=-1), x_win)
        out = out.squeeze(3).permute(0, 3, 1, 2)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class AttentionAggregate(ContextSelfSimilarity):
    def __init__(self, in_channels, out_channels, attn_channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=False, norm_emb=False, out_op="sum"):
        super(AttentionAggregate, self).__init__(in_channels, attn_channels, kernel_size, stride, dilation, groups,
                                                 bias, norm_emb)
        self.out_channels = out_channels
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        torch.nn.init.zeros_(self.value_conv.weight)

        if out_op == "sum":
            self.out_op = lambda x, y: x + y
        elif out_op == "cat":
            self.out_op = lambda x, y: torch.cat((x, y), dim=1)

    def forward(self, x, s):
        b, h, w, hs, ws = s.shape
        s = F.softmax(s.reshape(b, h, w, -1), dim=-1).unsqueeze(-1)
        x = x[:, :, ::self.dilation, ::self.dilation]
        v_out = self.value_conv(x)
        v_out = F.pad(v_out, [self.padding, self.padding, self.padding, self.padding], value=0)
        v_out = self.unfold2D(v_out)
        v_out = v_out[:, :, ::self.final_stride, ::self.final_stride, :, :]
        v_out = v_out.reshape(b, self.out_channels, h, w, -1).permute(0, 2, 3, 1, 4)
        out = torch.matmul(v_out, s).squeeze(-1).permute(0, 3, 1, 2)
        return self.out_op(out, x)

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.dilated_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.dilated_kernel_size // 2
        self.center_idx = kernel_size // 2

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = self.unfold2D(k_out)
        v_out = self.unfold2D(v_out)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def unfold2D(self, x):
        return x.unfold(2, self.dilated_kernel_size, self.stride) \
                   .unfold(3, self.dilated_kernel_size, self.stride)[:, :, :, :, ::self.dilation, ::self.dilation]

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def attn_or_avg(attention, avg_hidden_states, layer_outputs, rev_layer_outputs, bidirectional, lengths=None):
    if (attention is None) or (attention(layer_outputs) is None):
        if not avg_hidden_states:
            # TODO: want to take the last non-zero padded output here instead!
            last_fwd_feat = layer_outputs[:, -1, :, :, :]
            last_rev_feat = rev_layer_outputs[:, -1, :, :, :] if bidirectional else None
            reweighted = torch.concat([last_fwd_feat, last_rev_feat], dim=1) if bidirectional else last_fwd_feat
            reweighted = torch.mean(reweighted, dim=1)
        else:
            if lengths is not None:
                layer_outputs = [torch.mean(layer_outputs[i, :length], dim=0) for i, length in enumerate(lengths)]
                # TODO: sequences are padded, so you need to reverse only the non-padded inputs
                if rev_layer_outputs is not None: rev_layer_outputs = [torch.mean(rev_layer_outputs[i, :length], dim=0) for i, length in enumerate(lengths)] 
                outputs = torch.stack(layer_outputs + rev_layer_outputs) if rev_layer_outputs is not None else torch.stack(layer_outputs)
                reweighted = outputs #?
            else:
                outputs = torch.cat([layer_outputs, rev_layer_outputs], dim=1) if rev_layer_outputs is not None else layer_outputs
                reweighted = torch.mean(outputs, dim=1)
    else:
        outputs = torch.cat([layer_outputs, rev_layer_outputs], dim=1) if rev_layer_outputs is not None else layer_outputs
        reweighted = attention(outputs, lengths)
        reweighted = torch.sum(reweighted, dim=1)
    return reweighted

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling for FCN"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()