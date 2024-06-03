import torch
import torch.nn as nn

def get_conv_type(spatial_dims, is_transposed=False):
    if spatial_dims == 3:
        if is_transposed:
            return nn.ConvTranspose3d
        else:
            return nn.Conv3d
    elif spatial_dims == 2:
        if is_transposed:
            return nn.ConvTranspose2d
        else:
            return nn.Conv2d
    else:
        raise NotImplementedError
        
def get_norm_type(spatial_dims, type='batch'):
    if spatial_dims == 3:
        if type == 'batch':
            return nn.BatchNorm3d
        elif type == 'instance':
            return nn.InstanceNorm3d
        else:
            raise NotImplementedError
    elif spatial_dims == 2:
        if type == 'batch':
            return nn.BatchNorm2d
        elif type == 'instance':
            return nn.InstanceNorm2d
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError 

def get_dropout_type(spatial_dims):
    if spatial_dims == 3:
        return nn.Dropout3d
    elif spatial_dims == 2:
        return nn.Dropout2d

def create_conv_layer(spatial_dims, 
                      is_transposed, 
                      in_channels, 
                      out_channels, 
                      norm_type,
                      dropout,
                      kernel_size,
                      padding,
                      output_padding=None,
                      stride=1,
                      last_conv=False):
    if is_transposed:
        kwargs = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride,
            'output_padding': output_padding,
        }
    else:
        kwargs = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride,
        }
    return nn.Sequential(
                    get_conv_type(spatial_dims, is_transposed)(**kwargs),
                    get_norm_type(spatial_dims, norm_type)(out_channels),
                    nn.ReLU(),
                    get_dropout_type(spatial_dims)(dropout),
                    nn.Identity() if not last_conv else get_conv_type(spatial_dims, False)(in_channels=out_channels, 
                                                                                           out_channels=out_channels, 
                                                                                           kernel_size=3, 
                                                                                           padding=1,
                                                                                           stride=1)
                )

class ResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_type: str = 'batch',
        kernel_size: int = 3,
        num_convs: int = 2,
        dropout_rate: float = 0,
        last_conv: bool = False,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_convs):
            self.blocks.append(
                create_conv_layer(spatial_dims,
                                  is_transposed=False,
                                  in_channels=in_channels,
                                  out_channels=in_channels,
                                  norm_type=norm_type,
                                  dropout=dropout_rate,
                                  kernel_size=kernel_size,
                                  padding=(kernel_size - 1) // 2,
                                  stride=1)
            )
        if last_conv:
            self.blocks.append(get_conv_type(spatial_dims)(in_channels, 
                                                           in_channels,
                                                           kernel_size=kernel_size, 
                                                           padding=(kernel_size - 1) // 2))

    def forward(self, x):
        identity = x

        for block in self.blocks:
            x = block(x)

        x += identity

        return x