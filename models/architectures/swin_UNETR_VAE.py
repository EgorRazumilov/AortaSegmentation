from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer, SwinUNETR

from models.blocks.unet import create_conv_layer, ResBlock


class VAESwinUNETR(SwinUNETR):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        super().__init__(img_size,
                         in_channels,
                         out_channels,
                         depths,
                         num_heads,
                         feature_size,
                         norm_name,
                         drop_rate,
                         attn_drop_rate,
                         dropout_path_rate,
                         normalize,
                         use_checkpoint,
                         spatial_dims,
                         downsample,
                         use_v2)
        
        bottleneck_size = 16 * feature_size
        fc_size = int(bottleneck_size * (img_size // 2 ** 5) ** spatial_dims)
        fc_distr_size = 512
        
        self.vae_mean = nn.Linear(fc_size, fc_distr_size)
        self.vae_std = nn.Linear(fc_size, fc_distr_size)
        
        self.vae_linear_up = nn.Sequential(
            nn.Linear(fc_distr_size, fc_size),
            nn.ReLU(),
        )
        self.vae_up_blocks = nn.ModuleList()

        for i in range(5):
            in_channels = bottleneck_size // 2 ** i
            out_channels = bottleneck_size // 2 ** (i + 1) if i < 4 else 1
            self.vae_up_blocks.append(nn.Sequential(
                create_conv_layer(spatial_dims, 
                            is_transposed=True, 
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            norm_type=norm_name,
                            dropout=drop_rate,
                            kernel_size=3,
                            padding=1,
                            output_padding=1,
                            stride=2),

                *[ResBlock(spatial_dims, 
                           out_channels,
                           norm_name,
                           last_conv=(i==0)) for _ in range(2)],
            ))

    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        if self.training:
            X_orig = hidden_states_out[-1]
            X = X_orig.reshape(-1, self.vae_mean.in_features)
            z_mean = self.vae_mean(X)
            z_sigma = F.softplus(self.vae_std(X))
            z_mean_rand = torch.randn_like(z_mean)
            x_vae = z_mean + z_sigma * z_mean_rand
            x_vae = self.vae_linear_up(x_vae)
            x_vae = x_vae.reshape(X_orig.shape)
            for up_sample_block in self.vae_up_blocks:
                x_vae = up_sample_block(x_vae)
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)
            vae_mse_loss = F.mse_loss(x_in, x_vae)
            vae_loss = vae_reg_loss + vae_mse_loss
            return logits, vae_loss
        return logits