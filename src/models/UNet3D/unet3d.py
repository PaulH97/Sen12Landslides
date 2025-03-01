"""
Taken from https://github.com/roserustowicz/crop-type-mapping/
Implementation by the authors of the paper :
"Semantic Segmentation of crop type in Africa: A novel Dataset and analysis of deep learning methods"
R.M. Rustowicz et al.

Slightly modified to support image sequences of varying length in the same batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarking.helperFunctions import get_params_values

def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model

def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True))
    return model

def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
    return model

def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model

class UNet3D(nn.Module):
    def __init__(self, config):
        super(UNet3D, self).__init__()
        in_channel = get_params_values(config, "num_channels")
        n_classes = get_params_values(config, "num_classes")
        self.timesteps = get_params_values(config, "timeseries_len")
        dropout = get_params_values(config, "dropout", 0.0)
        
        feats = 16
        self.en3 = conv_block(in_channel, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)
        
        self.final = nn.Conv3d(feats * 2, n_classes, kernel_size=3, stride=1, padding=1)
        self.fn = nn.Linear(self.timesteps, 1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
    
    def forward(self, x):
        # Permute to change shape from NxTxCxHxW to NxCxTxHxW
        x = x.permute(0, 2, 1, 3, 4)
        assert x.shape[2] == self.timesteps, f"Expected {self.timesteps}, but got {x.shape[2]}"

        en3 = self.en3(x)
        pool_3 = self.pool_3(en3)
        
        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)
        
        center_in = self.center_in(pool_4)
        center_out = self.center_out(center_in)
        
        # Upsample to match en4 dimensions before concatenation
        center_out = F.interpolate(center_out, size=en4.shape[2:], mode='trilinear', align_corners=True)
        concat4 = torch.cat([center_out, en4], dim=1)
        
        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)
        
        # Upsample to match en3 dimensions before concatenation
        trans3 = F.interpolate(trans3, size=en3.shape[2:], mode='trilinear', align_corners=True)
        concat3 = torch.cat([trans3, en3], dim=1)
        
        dc3 = self.dc3(concat3)
        final = self.final(dc3)
        
        final = final.permute(0, 1, 3, 4, 2)
        shape_num = final.shape[0:4]
        final = final.reshape(-1, final.shape[4])
        
        final = self.dropout(final)
        final = self.fn(final)
        final = final.reshape(shape_num)

        return final

if __name__ == "__main__":
    # works for t, h, w = 4n
    bs, c, t, h, w = 2, 15, 16, 24, 24
    inputs = torch.randn((bs, c, t, h, w))
    
    net = UNet3D({'num_channels': c, 'num_classes': 20, 'max_seq_len': t, 'dropout': 0.5})
    
    out = net(inputs)
    print(out.shape)
