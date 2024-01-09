import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        make_backbone=None,
    ):
        super(ResNetEncoder, self).__init__()

        self.running_mean_and_var = nn.Sequential()
        input_channels = 1
        self.backbone = make_backbone(input_channels, baseplanes, ngroups)
        final_spatial = int(
            spatial_size * self.backbone.final_spatial_compress
        )
        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial ** 2))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial,
            final_spatial,
        )

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):

        cnn_input = []

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        depth_observations = observations

        cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x