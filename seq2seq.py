# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
import time
from model.comcom import timeSince, red_scan, red_image, red_scan_test, red_image_test, get_numpy_word_embed
from model.encode import EncoderRNN, BiEncoder
from model.decode import AttnDecoderRNN, DecoderRNN
import torchvision.transforms as transforms
import numpy as np
from model.comcom import prepareData
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0")

SOS_token = 0
EOS_token = 1

input_lang, output_lang, pairs = prepareData('lan', 'act', False)
numpy_embed = get_numpy_word_embed(input_lang.word2index)
_, _, test_pairs = prepareData('Test_lan', 'Test_act', False)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    images = pair[2]
    return (input_tensor, target_tensor, images)

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, image_path, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=30):
    encoder_hidden = encoder.initHidden()
    decoder_hidden = decoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    target = target_tensor[0]

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0, 0]
    # encoder_output, encoder_hidden = encoder(input_tensor)
    #print(encoder_hidden.shape)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    # decoder_input = torch.zeros(1, 100, device=device)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # if use_teacher_forcing:
    #     # Teacher forcing: Feed the target as the next input
    #     for di in range(target_length):
    #         RGB = red_image(image_path, di, 1)
    #         Deep = red_image(image_path, di, 0)
    #         Scan = red_scan(image_path, di)
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs, RGB, Deep, Scan)
    #         # loss += criterion(decoder_output, target_tensor[di])
    #         loss_i = criterion(decoder_output, target_tensor[di])
    #         if target_tensor[di].tolist() != target.tolist():
    #             loss_i = loss_i * 5
    #             target = target_tensor
    #         loss += loss_i
    #         decoder_input = target_tensor[di]  # Teacher forcing
    #
    # else:
    #     # Without teacher forcing: use its own predictions as the next input
    #     for di in range(target_length):
    #         RGB = red_image(image_path, di, 1)
    #         Deep = red_image(image_path, di, 0)
    #         Scan = red_scan(image_path, di)
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs, RGB, Deep, Scan)
    #         topv, topi = decoder_output.topk(1)
    #         decoder_input = topi.squeeze().detach()  # detach from history as input
    #
    #         # loss += criterion(decoder_output, target_tensor[di])
    #         loss_i = criterion(decoder_output, target_tensor[di])
    #         if target_tensor[di].tolist() != target.tolist():
    #             loss_i = loss_i * 5
    #             target = target_tensor
    #         loss += loss_i
    #         if decoder_input.item() == EOS_token:
    #             break
    #print(image_path)
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            RGB = red_image(image_path, di, 1)
            Deep = red_image(image_path, di, 0)
            # Scan = red_scan(image_path, di)
            # print(di)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_hidden, RGB, Deep)
            loss_i = criterion(decoder_output, target_tensor[di])
            if target_tensor[di].tolist() != target.tolist():
                loss_i = loss_i * 5
                target = target_tensor

            loss += loss_i
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # print(di)
            RGB = red_image(image_path, di, 1)
            Deep = red_image(image_path, di, 0)
            # Scan = red_scan(image_path, di)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_hidden, RGB, Deep)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss_i = criterion(decoder_output, target_tensor[di])
            if target_tensor[di].tolist() != target.tolist():
                loss_i = loss_i * 5
                target = target_tensor[di]

            loss += loss_i
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.001):
    global teacher_forcing_ratio
    start = time.time()
    print_loss_total = 0
    loss_a = 0
    counts = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    decoder_scheduler = StepLR(decoder_optimizer, step_size=1, gamma=0.95)
    encoder_scheduler = StepLR(encoder_optimizer, step_size=1, gamma=0.95)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]

    testing_pairs = [test_pairs[i] for i in range(len(test_pairs))]

    criterion = nn.NLLLoss()

    for epoch in range(1, n_iters + 1):

        # if epoch == 1:
        #     for t in range(0, len(testing_pairs)):
        #
        #         testing_pair = testing_pairs[t - 1]
        #         test_input = testing_pair[0]
        #         test_target = testing_pair[1]
        #         test_image_path = testing_pair[2]
        #         output_words, loss_e, count = evaluate(encoder, decoder, sentence=test_input, test_target=test_target,
        #                         test_image_path=test_image_path, max_length=30)
        #         loss_a += loss_e
        #         counts += count

                # print(test_image_path, end=' ')
                # print(test_target)
                # print(output_words)

            #print('test:', loss_a.data.cpu().numpy() / len(testing_pairs), counts / len(testing_pairs))

        training_pair = training_pairs[epoch - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        image_path = training_pair[2]

        loss = train(input_tensor, target_tensor, image_path, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('train:','%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_iters),
                                         epoch, epoch / n_iters * 100, print_loss_avg), end=' ')

            decoder_scheduler.step()
            encoder_scheduler.step()
            print(decoder_optimizer.param_groups[0]['lr'])
            # teacher_forcing_ratio = teacher_forcing_ratio * 0.9

        # if epoch % plot_every == 0:
        #     for t in range(0, len(testing_pairs)):
        #
        #         testing_pair = testing_pairs[t - 1]
        #         test_input = testing_pair[0]
        #         test_target = testing_pair[1]
        #         test_image_path = testing_pair[2]
        #         output_words, loss_e, count = evaluate(encoder, decoder, sentence=test_input, test_target=test_target,
        #                         test_image_path=test_image_path, max_length=30)
        #         loss_a += loss_e
        #         counts += count
        #
        #         # print(test_image_path, end=' ')
        #         # print(test_target)
        #         # print(output_words)
        #
        #     print('test:', loss_a.data.cpu().numpy() / len(testing_pairs), counts / len(testing_pairs))

        loss_a = 0
        counts = 0

    torch.save(encoder, 'checkpoints/encoder.pt')
    torch.save(decoder, 'checkpoints/decoder.pt')

def evaluate(encoder, decoder, sentence, test_target, test_image_path, max_length=20):
    criterion = nn.NLLLoss()
    loss = 0

    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        target_tensor = tensorFromSentence(output_lang, test_target)
        target = target_tensor[0]
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        decoder_hidden = decoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        decoded_out_tensor = torch.zeros(target_tensor.size()[0], 1, dtype=torch.int64, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
            # encoder_output, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoded_words = []
        for di in range(max_length):

            RGB = red_image_test(test_image_path, di, 1)
            Deep = red_image_test(test_image_path, di, 0)
            #Scan = red_scan_test(test_image_path, di)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_hidden, RGB, Deep)
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

            decoded_out_tensor[di] = decoder_input

            if di < len(target_tensor)-1:
                loss_i = criterion(decoder_output, target_tensor[di])
                if target_tensor[di].tolist() != target.tolist():
                    loss_i = loss_i * 5
                    target = target_tensor[di]
                loss += loss_i
            else:
                break
            if decoder_input.item() == EOS_token:
                break

        mask = (decoded_out_tensor == target_tensor).data.cpu().numpy()
        count = 0
        for m in mask:
            if m:
                count += 1

        count = count / target_tensor.size()[0]
        return decoded_words, loss / len(target_tensor), count

def evaluate_text(sentence,test_target, image_path):

    encoderC = torch.load('checkpoints/encoder.pt')
    decoderC = torch.load('checkpoints/decoder.pt')     # if use CPU add:  , map_location='cpu'

    output_words = evaluate(encoderC, decoderC, sentence,test_target, image_path, 15)
    return output_words

if __name__ == '__main__':

    hidden_size = 128
    #encoder = EncoderRNN(input_lang.n_words, hidden_size,).to(device)  #BiEncoder, EncoderRNN
    encoder = EncoderRNN(input_lang.n_words, 100, numpy_embed).to(device)
    # decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.2).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    # encoder = torch.load('checkpoints/encoder.pt')
    # decoder = torch.load('checkpoints/decoder.pt')     # if use CPU add:  , map_location='cpu'
    trainIters(encoder, decoder, 2000, print_every=100, plot_every=100)

    # #                    输入的自然语言指令，                           观测到的视觉信息
    #print(evaluate_text("bypass the bucket stop by the microwave", "right forward forward forward forward left forward forward left forward forward forward", "bypass the bucket go to the microwave_4_2"))
    #print(evaluate_text("stop by the microwave", "forward forward left forward forward forward forward forward forward forward", "bypass the bucket go to the microwave_7_1"))
    # #删除中间的视觉信息，这表示到达目的地能即使中止
    # print(evaluate_text("go to the laptop", "1 147 386 1327 1 286 373 1538 1 374 385 1350 1 459 374 956 1 459 374 856 1 459 374 856")) # 1 448 416 1151
    # print(evaluate_text("go to the laptop", "1 481 426 1350 1 481 416 1250 1 488 416 1151 1 489 374 1034 1 489 374 956 1 489 374 856"))
    # # 离目的较远
    # print(evaluate_text("go to the laptop", "1 701 336 1901 1 598 336 1802 1 503 342 1700 1 508 342 1604 1 501 357 1515 1 500 365 1375 1 498 379 1285 1 504 387 1190 1 507 397 1012 1 454 369 929"))
    # # 并没有指明走到那个物体，物体的观测信息
    # print(evaluate_text("four steps forward turn right 30 degrees", "1 147 386 1327 1 286 373 1538 1 374 385 1350 1 461 390 1250 1 448 416 1151 1 459 374 1034 1 459 374 956 0 0 0 0"))
    # # 并没有指明走到那个物体，没有物体的观测信息
    # print(evaluate_text("turn left 30 degrees and two steps forward", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"))
    # print(evaluate_text("two steps backward", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"))
    # print(evaluate_text("turn right 45 degrees and two steps backward", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"))
    # print(time.time() - start)