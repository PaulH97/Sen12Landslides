"""
code modified from: https://github.com/roserustowicz/crop-type-mapping
File housing all models.

Each model can be created by invoking the appropriate function
given by:

    make_MODELNAME_model(MODEL_SETTINGS)

Changes to allow this are still in progess
"""
from torchvision import models
from .constants import *
from .modelling.cgru_segmenter import CGRUSegmenter
from .modelling.clstm_segmenter import CLSTMSegmenter
from .modelling.fcn8 import FCN8
from .modelling.unet import UNet, UNet_Encode, UNet_Decode
from .modelling.unet3d import UNet3D
from .modelling.multi_input_clstm import MI_CLSTM
from .modelling.only_clstm_mi import ONLY_CLSTM_MI
from .modelling.attention import ApplyAtt, attn_or_avg

from benchmarking.helperFunctions import get_params_values
from benchmarking.models.UNet2D_CLSTM.LocalSelfAttention.cscl import ContextSelfSimilarity

class FCN_CRNN(nn.Module):
    def __init__(self, config):
        # fcn_input_size, crnn_input_size, crnn_model_name,
        #          hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, avg_hidden_states,
        #          num_classes, bidirectional, pretrained, early_feats, use_planet, resize_planet,
        #          num_bands_dict, main_crnn, main_attn_type, attn_dims,
        #          enc_crnn, enc_attn, enc_attn_type, stage=2):
        super(FCN_CRNN, self).__init__()

        self.fcn_input_size = (config['timeseries_len'], config['num_channels'], config['img_res'], config['img_res'])
        self.crnn_input_size = (config['timeseries_len'], 128, config['img_res'], config['img_res'])
        self.hidden_dims = 128
        self.lstm_kernel_sizes = (3, 3)
        self.conv_kernel_size = 3
        self.lstm_num_layers = 1
        self.avg_hidden_states = True
        self.num_classes = config['num_classes']
        self.bidirectional = False
        self.early_feats = False
        self.use_planet = False
        self.resize_planet = False
        self.num_bands_dict = {'planet': 0, 's1': 0, 's2': config['num_channels'], 'all': config['num_channels']}
        self.main_attn_type = 'None'
        self.attn_dims = 32
        self.main_crnn = False
        self.enc_crnn = False
        self.enc_attn = False
        self.enc_attn_type = None
        self.crnn_model_name = 'clstm'
        self.pretrained = False
        self.processed_feats = {'main': None, 'enc4': None, 'enc3': None, 'enc2': None, 'enc1': None }
        self.stage = config['train_stage']
        self.br_layer = get_params_values(config, "br_layer", False)

        # get appropriate encoder / decoder
        if not self.early_feats:
            self.fcn = make_UNet_model(n_class=self.crnn_input_size[1], num_bands_dict=self.num_bands_dict, late_feats_for_fcn=True,
                                       pretrained=self.pretrained, use_planet=self.use_planet, resize_planet=self.resize_planet)
        else:
            self.fcn_enc = make_UNetEncoder_model(self.num_bands_dict, use_planet=self.use_planet, resize_planet=self.resize_planet, pretrained=self.pretrained)
            self.fcn_dec = make_UNetDecoder_model(self.num_classes, late_feats_for_fcn=False,  use_planet=self.use_planet, resize_planet=self.resize_planet)

        if self.crnn_model_name == "gru":
            if self.early_feats:
                self.crnn = CGRUSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes,
                                          self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1],
                                          self.bidirectional, self.avg_hidden_states)
            else:
                self.crnn = CGRUSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes,
                                          self.conv_kernel_size, self.lstm_num_layers, self.num_classes,
                                          self.bidirectional, self.avg_hidden_states)

        elif self.crnn_model_name == "clstm":
            self.attns = self.get_attns()
            self.crnns = self.get_crnns()
            self.final_convs = self.get_final_convs()

        if self.stage in [0, 3, 4]:
            self.attn_channels = config['attn_channels']
            self.kernel_size = config['cscl_win_size']
            self.dilation = config['cscl_win_dilation']
            self.stride = config['cscl_win_stride']
            # self.padding = config['attn_pad_size']
            self.groups = config['attn_groups']
            self.norm_emb = get_params_values(config, 'norm_emb', False)
            self.attn_sim = ContextSelfSimilarity(
                in_channels=self.hidden_dims, attn_channels=self.attn_channels, kernel_size=self.kernel_size,
                stride=self.stride, dilation=self.dilation, groups=self.groups, bias=False, norm_emb=self.norm_emb)

        elif self.stage in [2]:
            self.main_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.num_classes,
                                            kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size - 1) / 2))
            #self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hres_inputs=None):
        batch, timestamps, bands, rows, cols = input_tensor.size()
        fcn_input = input_tensor.reshape(batch * timestamps, bands, rows, cols)

        fcn_input_hres = None

        # Encode and decode features
        fcn_output = self.fcn(fcn_input, fcn_input_hres)

        # Apply CRNN
        crnn_input = fcn_output.view(batch, timestamps, -1, fcn_output.shape[-2], fcn_output.shape[-1])
        if self.crnn_main(crnn_input) is not None:
            crnn_output_fwd, crnn_output_rev = self.crnn_main(crnn_input)
        else:
            crnn_output_fwd = crnn_input
            crnn_output_rev = None

        # Apply attention
        reweighted = attn_or_avg(self.attns['main'], self.avg_hidden_states, crnn_output_fwd, crnn_output_rev, self.bidirectional)

        if self.stage in [0]:
            # print(reweighted.shape)
            return self.attn_sim(reweighted)

        elif self.stage in [2]:
            # Apply final conv
            logits = self.main_finalconv(reweighted)
            # print('logits before br: ', logits.shape)
            if self.br_layer:
                logits = self.br(logits)
                # print('logits after br : ', logits.shape)
            return logits

    def get_crnns(self):
        self.crnn_main = self.crnn_enc4 = self.crnn_enc3 = self.crnn_enc2 = self.crnn_enc1 = None
        if self.early_feats:
            if self.main_crnn:
                self.crnn_main = CLSTMSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1], self.bidirectional) 
            if self.enc_crnn:
                crnn_input0, crnn_input1, crnn_input2, crnn_input3 = self.crnn_input_size
                self.crnn_enc4 = CLSTMSegmenter([crnn_input0, crnn_input1//2, crnn_input2*2, crnn_input3*2], self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//2, self.bidirectional)
                self.crnn_enc3 = CLSTMSegmenter([crnn_input0, crnn_input1//4, crnn_input2*4, crnn_input3*4], self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//4, self.bidirectional)
                if self.use_planet and not self.resize_planet:
                    self.crnn_enc2 = CLSTMSegmenter([crnn_input0, crnn_input1//8, crnn_input2*8, crnn_input3*8], self.hidden_dims, self.lstm_kernel_sizes, 
                                       self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//8, self.bidirectional)
                    self.crnn_enc1 = CLSTMSegmenter([crnn_input0, crnn_input1//16, crnn_input2*16, crnn_input3*16], self.hidden_dims, self.lstm_kernel_sizes, 
                                       self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//16, self.bidirectional)
        else:
            self.crnn_main = CLSTMSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.num_classes, self.bidirectional)
        self.crnns = { 'main': self.crnn_main, 'enc4': self.crnn_enc4, 'enc3': self.crnn_enc3, 'enc2': self.crnn_enc2, 'enc1': self.crnn_enc1 }
        return self.crnns

    def get_attns(self):
        self.attn_enc4 = self.attn_enc3 = self.attn_enc2 = self.attn_enc1 = None
        if self.early_feats:
            if self.enc_attn:
                self.attn_enc4 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims) 
                self.attn_enc3 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims)
                if self.use_planet and not self.resize_planet:
                    self.attn_enc2 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims)
                    self.attn_enc1 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims)
        self.attn_main = ApplyAtt(self.main_attn_type, self.hidden_dims, self.attn_dims)
        self.attns = { 'main': self.attn_main, 'enc4': self.attn_enc4, 'enc3': self.attn_enc3, 'enc2': self.attn_enc2, 'enc1': self.attn_enc1 } 
        return self.attns

    def get_final_convs(self):
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

def make_MI_CLSTM_model(num_bands, 
                        unet_out_channels,
                        crnn_input_size,
                        hidden_dims, 
                        lstm_kernel_sizes, 
                        lstm_num_layers, 
                        conv_kernel_size, 
                        num_classes, 
                        avg_hidden_states, 
                        early_feats, 
                        bidirectional,
                        max_timesteps,
                        satellites,
                        resize_planet,
                        grid_size,
                        main_attn_type,
                        attn_dims): 

    model = MI_CLSTM(num_bands,
                     unet_out_channels,
                     crnn_input_size,
                     hidden_dims, 
                     lstm_kernel_sizes, 
                     conv_kernel_size, 
                     lstm_num_layers, 
                     avg_hidden_states, 
                     num_classes,
                     early_feats,
                     bidirectional,
                     max_timesteps,
                     satellites,
                     resize_planet,
                     grid_size,
                     main_attn_type,
                     attn_dims)
    return model

def make_MI_only_CLSTM_model(num_bands, crnn_input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, 
                             lstm_num_layers, avg_hidden_states, num_classes, bidirectional, max_timesteps,
                             satellites, main_attn_type, attn_dims):

    model = ONLY_CLSTM_MI(num_bands, crnn_input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size,
                          lstm_num_layers, avg_hidden_states, num_classes, bidirectional, max_timesteps,
                          satellites, main_attn_type, attn_dims)
    return model

def make_bidir_clstm_model(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, avg_hidden_states, main_attn_type, attn_dims):
    """ Defines a (bidirectional) CLSTM model 
    Args:
        input_size - (tuple) size of input dimensions 
        hidden_dims - (int or list) num features for hidden layers 
        lstm_kernel_sizes - (int) kernel size for lstm cells
        conv_kernel_size - (int) ketnel size for convolutional layers
        lstm_num_layers - (int) number of lstm cells to stack
        num_classes - (int) number of classes to predict
        bidirectional - (bool) if True, include reverse inputs and concatenate output features from forward and reverse models
                               if False, use only forward inputs and features
    
    Returns:
      returns the model! 
    """
    model = CLSTMSegmenter(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, 
                           with_pred=True, avg_hidden_states=avg_hidden_states, attn_type=main_attn_type, attn_dims=attn_dims) 
    return model

def make_fcn_model(n_class, n_channel, freeze=True):
    """ Defines a FCN8s model
    Args: 
      n_class - (int) number of classes to predict
      n_channel - (int) number of channels in input
      freeze - (bool) whether to use pre-trained weights
                TODO: unfreeze after x epochs of training

    Returns: 
      returns the model!
    """
    ## load pretrained model
    fcn8s_pretrained_model=torch.load("torchfcn.models.FCN8s.download()")
    fcn8s = FCN8(n_class, n_channel)
    fcn8s.load_state_dict(fcn8s_pretrained_model,strict=False)
    
    if freeze:
        ## Freeze the parameter you do not want to tune
        for param in fcn8s.parameters():
            if torch.sum(param==0)==0:
                param.requires_grad = False
    
    return fcn8s

def make_UNet_model(n_class, num_bands_dict, late_feats_for_fcn=False, pretrained=True, use_planet=False, resize_planet=False):
    """ Defines a U-Net model
    Args:
      n_class - (int) number of classes to predict
      n_channel - (int) number of channels in input
      for_fcn - (bool) whether or not U-Net is to be used for FCN + CLSTM, 
                 or false if just used as a U-Net. When True, the last conv and 
                 softmax layer is removed and features are returned. When False, 
                 the softmax layer is kept and probabilities are returned. 
      pretrained - (bool) whether to use pre-trained weights

    Returns: 
      returns the model!
    """
    model = UNet(n_class, num_bands_dict, late_feats_for_fcn, use_planet, resize_planet)
    
    if pretrained:
        # TODO: Why are pretrained weights from vgg13? 
        pre_trained = models.vgg13(pretrained=True)
        pre_trained_features = list(pre_trained.features)

        model.unet_encode.enc3.encode[3] = pre_trained_features[2] #  64 in,  64 out
        model.unet_encode.enc4.encode[0] = pre_trained_features[5] #  64 in, 128 out
        model.unet_encode.enc4.encode[3] = pre_trained_features[7] # 128 in, 128 out
        model.unet_encode.center[0] = pre_trained_features[10]     # 128 in, 256 out
        
    model = model.cuda()
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

    model = model.cuda()
    return model

def make_UNetDecoder_model(n_class, late_feats_for_fcn, use_planet, resize_planet):
    model = UNet_Decode(n_class, late_feats_for_fcn, use_planet, resize_planet)
    model = model.cuda()
    return model

def make_fcn_clstm_model(country, fcn_input_size, crnn_input_size, crnn_model_name, 
                         hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, avg_hidden_states,
                         num_classes, bidirectional, pretrained, early_feats, use_planet, resize_planet,
                         num_bands_dict, main_crnn, main_attn_type, attn_dims,
                         enc_crnn, enc_attn, enc_attn_type):
    """ Defines a fully-convolutional-network + CLSTM model
    Args:
      fcn_input_size - (tuple) input dimensions for FCN model
      fcn_model_name - (str) model name used as the FCN portion of the network
      crnn_input_size - (tuple) input dimensions for CRNN model
      crnn_model_name - (str) model name used as the convolutional RNN portion of the network 
      hidden_dims - (int or list) num features for hidden layers 
      lstm_kernel_sizes - (int) kernel size for lstm cells
      conv_kernel_size - (int) ketnel size for convolutional layers
      lstm_num_layers - (int) number of lstm cells to stack
      num_classes - (int) number of classes to predict
      bidirectional - (bool) if True, include reverse inputs and concatenate output features from forward and reverse models
                               if False, use only forward inputs and features
      pretrained - (bool) whether to use pre-trained weights

    Returns: 
      returns the model!
    """
    if early_feats:
        crnn_input_size += (GRID_SIZE[country] // 4, GRID_SIZE[country] // 4)
    else:
        crnn_input_size += (GRID_SIZE[country], GRID_SIZE[country]) 

    model = FCN_CRNN(fcn_input_size, crnn_input_size, crnn_model_name, hidden_dims, lstm_kernel_sizes, 
                     conv_kernel_size, lstm_num_layers, avg_hidden_states, num_classes, bidirectional, pretrained, 
                     early_feats, use_planet, resize_planet, num_bands_dict, main_crnn, main_attn_type, attn_dims, 
                     enc_crnn, enc_attn, enc_attn_type)
    model = model.cuda()

    return model

def make_UNet3D_model(n_class, n_channel, timesteps, dropout):
    """ Defined a 3d U-Net model
    Args: 
      n_class - (int) number of classes to predict
      n_channels - (int) number of input channgels

    Returns:
      returns the model!
    """

    model = UNet3D(n_channel, n_class, timesteps, dropout)
    model = model.cuda()
    return model
