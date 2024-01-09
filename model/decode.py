# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encode import VlnResnetDepthEncoder, TorchVisionResNet50, ScanEncoder

from comcom import SoftDotAttention

device = torch.device("cuda:0")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = 0.3
        self.embedding = nn.Embedding(output_size, 128)
        self.RGB_embedding = TorchVisionResNet50(
            256,
            device,
            spatial_output=False,
        ).to(device)
        self.Deep_embedding = VlnResnetDepthEncoder(
            128,
            './checkpoints/gibson-2plus-resnet50.pth',
            backbone="resnet50",
            resnet_baseplanes=32,
            trainable=False,
            spatial_output=False,
        ).to(device)
        self.scan_embedding = ScanEncoder(32, self.hidden_size).to(device)
        self.dropout = nn.Dropout(self.dropout_p)
        #self.feat_att_layer = SoftDotAttention(hidden_size, hidden_size * 4 + 128, 128)
        self.cat_emb = nn.Linear(hidden_size * 4 + 100, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_hidden, RGB, Deep):
        embed = self.embedding(input).view(1, 1, -1)
        RGB = self.RGB_embedding(RGB)
        Deep = self.Deep_embedding(Deep)
        #Scan = self.scan_embedding(Scan)

        vision = torch.cat((embed[0], RGB, Deep, encoder_hidden.squeeze(0)), 1) #.squeeze(0), Scan.squeeze(0)
        vision = self.dropout(vision).view(1, 1, -1)

        #output, _ = self.feat_att_layer(hidden.squeeze(0), vision)
        output = self.cat_emb(vision)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        #output = torch.cat((Scan.squeeze(0), output[0]), 1)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

MAX_LENGTH = 30
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, 32)
        self.RGB_embedding = TorchVisionResNet50(
            256,
            device,
            spatial_output=False,
        ).to(device)
        self.Deep_embedding = VlnResnetDepthEncoder(
            128,
            './checkpoints/gibson-2plus-resnet50.pth',
            backbone="resnet50",
            resnet_baseplanes=32,
            trainable=False,
            spatial_output=False,
        ).to(device)
        self.scan_embedding = ScanEncoder(32, 128).to(device)
        self.cat_emb = nn.Linear(544, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, RGB, Deep, Scan):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        RGB = self.RGB_embedding(RGB)
        Deep = self.Deep_embedding(Deep)
        Scan = self.scan_embedding(Scan)
        vision = torch.cat((RGB, Deep, Scan.squeeze(0)), 1)
        vision = self.dropout(vision).view(1, 1, -1)

        output = torch.cat((embedded[0], vision[0]), 1)
        output = self.cat_emb(output).unsqueeze(0)
        output = F.relu(output)
        output = self.dropout(output)

        attn_weights = F.softmax(
            self.attn(torch.cat((output[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((output[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)