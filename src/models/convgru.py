"""
Original from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
Modified from https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/convgru.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.in_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self.out_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        combined = torch.cat([input_tensor, cur_state], dim=1)
        z, r = torch.sigmoid(self.in_conv(combined)).chunk(2, dim=1)
        h = torch.tanh(self.out_conv(torch.cat([input_tensor, r * cur_state], dim=1)))
        new_state = (1 - z) * cur_state + z * h
        return new_state

    def init_hidden(self, batch_size, device):
        return Variable(
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        ).to(device)


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvGRUCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, pad_mask=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=input_tensor.size(0), device=input_tensor.device
            )

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=h
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            if pad_mask is not None:
                last_positions = (~pad_mask).sum(dim=1) - 1
                layer_output = layer_output[:, last_positions, :, :, :]

            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvGRU_Seg(nn.Module):
    """A ConvGRU model for semantic segmentation of image time series.

    Example:
        >>> model = ConvGRU_Seg(
        ...     num_classes=1,
        ...     img_res=128,
        ...     in_channels=10,
        ...     kernel_size=(3, 3),
        ...     hidden_dim=128
        ... )
        >>> batch = {"img": torch.randn(2, 15, 10, 128, 128)}
        >>> output = model(batch)
        >>> print(output["segmentation"].shape)
        torch.Size([2, 1, 128, 128])
    """
    def __init__(
        self,
        num_classes,
        img_res,
        in_channels,
        kernel_size,
        hidden_dim,
        pad_value=0,
    ):
        super(ConvGRU_Seg, self).__init__()

        self.num_classes = num_classes
        self.input_size = (img_res, img_res)
        self.in_channels = in_channels
        self.kernel_size = tuple(kernel_size)
        self.hidden_dim = hidden_dim
        self.pad_value = pad_value

        self.convgru_encoder = ConvGRU(
            input_dim=self.in_channels,
            input_size=self.input_size,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            return_all_layers=False,
        )
        
        self.classification_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.num_classes,
            kernel_size=self.kernel_size,
            padding=self.kernel_size[0] // 2,
        )

    def forward(self, x):
        """
        Args:
            x: dict or tensor
               x["img"]: [B, T, C, H, W]
               or direct tensor [B, T, C, H, W]
        
        Returns:
            dict: {"segmentation": [B, num_classes, H, W]}
        """
        # Extract tensor from dict if needed
        if isinstance(x, dict):
            input_tensor = x["img"]  # [B, T, C, H, W]
        else:
            input_tensor = x
        
        # Create pad mask based on pad_value
        pad_mask = (
            (input_tensor == self.pad_value)
            .all(dim=-1).all(dim=-1).all(dim=-1)
        )  # [B, T]
        pad_mask = pad_mask if pad_mask.any() else None
        
        # ConvGRU forward
        _, out = self.convgru_encoder(input_tensor, pad_mask=pad_mask)
        
        # Classification
        out = self.classification_layer(out)
        
        return {"segmentation": out}

    
if __name__ == "__main__":
    bs, t, c, h, w = 4, 15, 10, 128, 128
    
    # Test with dict input (new structure)
    batch = {
        "img": torch.randn(bs, t, c, h, w),
        "dates": torch.randn(bs, t),
        "msk": torch.randint(0, 2, (bs, h, w)),
    }
    
    model = ConvGRU_Seg(
        num_classes=1, 
        img_res=h, 
        in_channels=c, 
        kernel_size=(3, 3), 
        hidden_dim=16, 
        pad_value=0
    )
    
    out = model(batch)
    
    print(f"✓ Test passed")
    print(f"  Input:  {batch['img'].shape}")
    print(f"  Output: {out['segmentation'].shape}")
    
    out_direct = model(batch["img"])
    print(f"✓ Direct tensor input also works")