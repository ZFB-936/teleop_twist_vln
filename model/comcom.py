# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import os
import math
import time
from PIL import Image
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn

device = torch.device("cuda:0")

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_numpy_word_embed(word2ix):
    row = 0
    file = 'glove.6B.100d.txt'
    path = '/home/zfb/下载/glove_6B'
    whole = os.path.join(path, file)
    words_embed = {}
    with open(whole, mode='r')as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            # print(len(line.split()))
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            if row > 20000:
                break
            row += 1
    # word2ix = {}
    ix2word = {ix: w for w, ix in word2ix.items()}
    id2emb = {}
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 100
    data = [id2emb[ix] for ix in range(len(word2ix))]

    return data

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?_]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

transf_RGB = transforms.Compose([
    transforms.ColorJitter(contrast=0.5),
    #transforms.RandomRotation(5),
    #transforms.CenterCrop(270),
    transforms.ToTensor(),
])

transf_Deep = transforms.Compose([
    transforms.ToTensor(),
])

def red_image(path, Action, c):

    path = "./data/image/" + str(path) + '/' + str(Action)
    if c == 1:
        image = cv.imread(path + '/RGB.jpg', c)
        image = cv.resize(image, (360, 270), interpolation=cv.INTER_AREA)
        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        image_input = transf_RGB(image)

    else:
        image = cv.imread(path + '/Deep.jpg', c)
        image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
        image_input = transf_Deep(image)  # tensor数据格式是torch(C,H,W)

    return image_input.unsqueeze(dim=0).to(device)

def red_scan(path, Action):

    path = "./data/image/" + str(path) + '/' + str(Action)
    scan = np.loadtxt(path + '/scan.csv', delimiter=',')
    scan = scan[16:48]

    scan = torch.tensor(scan, dtype=torch.float32, device=device).view(1, -1)
    noise = torch.rand(1, 32) / 10.0
    scan = scan + noise.to(device)
    return scan

def red_image_test(path, Action, c):

    path = "./data/image/" + str(path) + '/' + str(Action)
    if c == 1:
        image = cv.imread(path + '/RGB.jpg', c)
        image = cv.resize(image, (360, 270), interpolation=cv.INTER_AREA)
        image_input = transforms.ToTensor()(image)
    else:
        image = cv.imread(path + '/Deep.jpg', c)
        image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
        image_input = transforms.ToTensor()(image)  # tensor数据格式是torch(C,H,W)

    return image_input.unsqueeze(dim=0).to(device)

def red_scan_test(path, Action):

    path = "./data/image/" + str(path) + '/' + str(Action)
    scan = np.loadtxt(path + '/scan.csv', delimiter=',')
    scan = scan[16:48]
    scan = torch.tensor(scan, dtype=torch.float32, device=device).view(1, -1)

    return scan

def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.01)

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim, out_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, out_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn