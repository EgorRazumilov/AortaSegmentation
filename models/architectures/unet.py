import torch
import torch.nn as nn
from models.blocks.unet import create_conv_layer, ResBlock

class UNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels,
        num_res_units: int = 0,
        norm_type: str = 'batch',
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_units = num_res_units
        self.norm_type = norm_type
        self.dropout = dropout

        self.down_blocks, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        for i in range(len(channels)):
            if i == 0:
                in_channels_down = self.in_channels
                out_channels_down = self.channels[0] 
            else:
                in_channels_down = self.channels[i - 1]
                out_channels_down = self.channels[i]
            down_block = nn.Sequential(
                *[ResBlock(spatial_dims, 
                           in_channels_down,
                           norm_type) 
                  for _ in range(num_res_units)],

                create_conv_layer(spatial_dims, 
                            is_transposed=False, 
                            in_channels=in_channels_down, 
                            out_channels=out_channels_down, 
                            norm_type=norm_type,
                            dropout=dropout,
                            kernel_size=3,
                            padding=1,
                            stride=2)
            )
            self.down_blocks.append(down_block)

            if i == 0:
                in_channels_up = self.channels[0]
                out_channels_up = self.out_channels
            else:
                in_channels_up = self.channels[i]
                out_channels_up = self.channels[i - 1]
            up_block = nn.Sequential(
                create_conv_layer(spatial_dims, 
                            is_transposed=True, 
                            in_channels=2 * in_channels_up, 
                            out_channels=out_channels_up, 
                            norm_type=norm_type,
                            dropout=dropout,
                            kernel_size=3,
                            padding=1,
                            output_padding=1,
                            stride=2),

                *[ResBlock(spatial_dims, 
                           out_channels_up,
                           norm_type,
                           last_conv=(i==0)) for _ in range(num_res_units)],
            )

            self.up_blocks.append(up_block)
        
        self.intermediate_block = ResBlock(spatial_dims, self.channels[-1], norm_type)


    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        encoder_outs = [x_in]
        for down_block in self.down_blocks:
            x_in = down_block(x_in)
            encoder_outs.append(x_in)
        encoder_outs = list(reversed(encoder_outs))

        x_inter = self.intermediate_block(x_in)
        for i, up_block in enumerate(reversed(self.up_blocks)):
            x_inter = up_block(torch.cat([x_inter, encoder_outs[i]], dim=1))
        return x_inter
