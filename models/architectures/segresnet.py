import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.unet import create_conv_layer, ResBlock

class SegResNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels,
        num_res_units: int = 0,
        norm_type: str = 'batch',
        dropout: float = 0.0,
        skip_type: str = 'cat'
    ) -> None:
        super().__init__()

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_units = num_res_units
        self.norm_type = norm_type
        self.dropout = dropout
        self.skip_type = skip_type


        bottleneck_size = self.channels[-1]
        fc_size = int(bottleneck_size * (image_size // 2 ** len(channels)) ** spatial_dims)
        fc_distr_size = 512
        
        self.vae_mean = nn.Linear(fc_size, fc_distr_size)
        self.vae_std = nn.Linear(fc_size, fc_distr_size)
        
        self.vae_linear_up = nn.Sequential(
            nn.Linear(fc_distr_size, fc_size),
            nn.ReLU(),
        )

        self.down_blocks, self.up_blocks, self.up_blocks_vae = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
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
            factor = 2 if self.skip_type == 'cat' else 1
            up_block = nn.Sequential(
                create_conv_layer(spatial_dims, 
                            is_transposed=True, 
                            in_channels=factor * in_channels_up, 
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
            
            up_block_vae = nn.Sequential(
                create_conv_layer(spatial_dims, 
                            is_transposed=True, 
                            in_channels=in_channels_up, 
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

            self.up_blocks_vae.append(up_block_vae)


        self.intermediate_block = ResBlock(spatial_dims, self.channels[-1], norm_type)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_in_orig = x_in.clone()
        encoder_outs = [x_in]
        for down_block in self.down_blocks:
            x_in = down_block(x_in)
            encoder_outs.append(x_in)
        encoder_outs = list(reversed(encoder_outs))

        x_inter = self.intermediate_block(x_in)

        if self.training:
            X_orig = x_inter.clone()
            X = X_orig.reshape(-1, self.vae_mean.in_features)
            z_mean = self.vae_mean(X)
            z_sigma = F.softplus(self.vae_std(X))
            z_mean_rand = torch.randn_like(z_mean)
            x_vae = z_mean + z_sigma * z_mean_rand
            x_vae = self.vae_linear_up(x_vae)
            x_vae = x_vae.reshape(X_orig.shape)
            for up_block_vae in reversed(self.up_blocks_vae):
                x_vae = up_block_vae(x_vae)
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)
            vae_mse_loss = F.mse_loss(x_in_orig, x_vae)
            vae_loss = vae_reg_loss + vae_mse_loss

        for i, up_block in enumerate(reversed(self.up_blocks)):
            if self.skip_type == 'cat':
                x_inter = torch.cat([x_inter, encoder_outs[i]], dim=1)
            else:
                x_inter = x_inter + encoder_outs[i]
            x_inter = up_block(x_inter)

        if self.training:
            return x_inter, vae_loss
        
        return x_inter
