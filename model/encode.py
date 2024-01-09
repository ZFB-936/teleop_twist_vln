# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
from model.basic_model import resnet
from model.basic_model.ResNetEncoder import ResNetEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, numpy_embed):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_embed)).to('cuda')
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class BiEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout = 0.2):
        super(BiEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)

class ScanEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ScanEncoder, self).__init__()
        self.context = nn.Linear(input_size, hidden_size)
        # self.Squeeze = nn.Linear(hidden_size, hidden_size // 4)
        # self.Excitation = nn.Linear(hidden_size // 4, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, scan):

        output = self.relu(self.context(scan))
        # output = self.relu(self.Squeeze(output))
        # output = self.relu(self.Excitation(output))
        return output.view(1, 1, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        trainable=False,
        spatial_output = False,
    ):
        super(VlnResnetDepthEncoder, self).__init__()
        self.visual_encoder = ResNetEncoder(
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            spatial_size=128,
            make_backbone=getattr(resnet, backbone)
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        observations = observations
        x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)

class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.
    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """
    def __init__(self, output_size, device, spatial_output = False):
        super(TorchVisionResNet50, self).__init__()
        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 2048
        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()

        self.layer_extract = self.cnn._modules.get("avgpool")

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """
        def resnet_forward(observation):
            resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        rgb_observations = observations
        # rgb_observations = rgb_observations / 255.0  # normalize RGB
        resnet_output = resnet_forward(rgb_observations.contiguous())

        return self.activation(
            self.fc(torch.flatten(resnet_output, 1))
        )  # [BATCH x OUTPUT_DIM]

if __name__ == "__main__":

    scan = np.loadtxt('a.csv', delimiter=',')
    #scan = torch.randn(1, 64)
    scan = torch.tensor(scan, dtype=torch.float32, device=device).view(1, -1)
    Encoder = ScanEncoder(64, 128).to(device)
    out = Encoder(scan)
    print(out.shape)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # img = cv.imread('/home/zfb/catkin_ws/src/teleop_twist/data/image/go to the laptop_0/1/RGB.jpg')
    # transf = transforms.ToTensor()
    # rgb_input = transf(img).to(device)  # tensor数据格式是torch(C,H,W)
    # print(rgb_input.shape)
    # 
    # deep = cv.imread('/home/zfb/catkin_ws/src/teleop_twist/data/image/go to the laptop_0/1/Deep.jpg',0)
    # deep = cv.resize(deep, (256, 256), interpolation=cv.INTER_AREA)
    # transf = transforms.ToTensor()
    # deep_input = transf(transf).to(device)  # tensor数据格式是torch(C,H,W)
    # print(deep_input.shape)
    # 
    # prev_actions = torch.randn(2 * 4, 32)
    # masks = torch.randint(1, size=(2 * 4,)).float()
    # rgb_encoder = TorchVisionResNet50(
    #         256,
    #         device,
    #         spatial_output=False,
    #     ).to(device)
    # 
    # deep_encoder = VlnResnetDepthEncoder(
    #         128,
    #         '/home/zfb/catkin_ws/src/teleop_twist/checkpoints/gibson-2plus-resnet50.pth',
    #         backbone="resnet50",
    #         resnet_baseplanes=32,
    #         trainable=False,
    #         spatial_output=False,
    #     ).to(device)
    # 
    # rgb_out = rgb_encoder(rgb_input)
    # print(rgb_out.shape)
    # 
    # rgb_out = deep_encoder(deep_input)
    # print(rgb_out.shape)
