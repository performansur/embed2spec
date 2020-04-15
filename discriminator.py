import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import NewDataset
from torch.utils.data import DataLoader
import datetime
import os
from multiscale_convolution import Decoder
from multiscale_convolution import get_logger
from torch.autograd import Variable
inputSize = 257

# discriminator module which is adapted by LSGAN
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(inputSize, 2*inputSize, kernel_size=5, stride=2)
        self.c2 = nn.Conv1d(2*inputSize, 4*inputSize, kernel_size=5, stride=2)
        self.c3 = nn.Conv1d(4*inputSize, 8*inputSize, kernel_size=5, stride=2)
        self.c4 = nn.Conv1d(8*inputSize, 16*inputSize, kernel_size=5, stride=2)

        self.bn1 = nn.BatchNorm1d(num_features=2*inputSize)
        self.bn2 = nn.BatchNorm1d(num_features=4*inputSize)
        self.bn3 = nn.BatchNorm1d(num_features=8*inputSize)

        self.lin = nn.Linear(in_features=16*inputSize, out_features=1, bias=True)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.c1(x)))
        x = self.bn2(F.leaky_relu(self.c2(x)))
        x = self.bn3(F.leaky_relu(self.c3(x)))
        x = F.leaky_relu(self.c4(x))
        x = x.view(-1, 16*inputSize)
        x = self.lin(x)
        return x


if __name__ == '__main__':
    arch = 'ms_cnn_gan'
    embed_type = 'softmaxRep'
    embed_crop = 'meanCrop'
    dataset = 'combined'
    run_name = ''
    log_loc = './logs/'
    now = datetime.datetime.now()
    time = str(now.day) + '.' + str(now.month) + '.' + str(now.year) + '__' + str(now.hour) + ':' + str(now.minute)
    logFileName = arch + '_' + embed_type + '_' + embed_crop + '_' + dataset + '_' + run_name + '_' + time + '.log'
    log = get_logger('zerospeech', logFileName)

    # server = 'gpu1'
    server = 'gpu2'
    if server == 'gpu1':
        prefix = '/mnt/gpu2'
    else:
        prefix = ''
    output_path = prefix + '/home/mansur/zerospeech/models/ms_cnn_gan_speaker_models/'


    device = "cuda"
    data = NewDataset(dataset)
    print('Data is ready')
    loader = DataLoader(data, batch_size=64, num_workers=4, shuffle=True)
    gen = Decoder().to(device)
    disc = Discriminator().to(device)
    criterion = nn.MSELoss()  # first try to reconstruct the spectrum
    optG = optim.Adam(gen.parameters())
    optD = optim.Adam(disc.parameters())
    Tensor = torch.cuda.FloatTensor

    print('Start Training')
    max_epoch = 150
    for epoch in range(max_epoch):
        totalLoss_g = 0.0
        totalLoss_d = 0.0
        coef = 2/(1 + np.exp(-10*epoch/max_epoch))
        for speaker, embedding, fft, lengths, _, _ in loader:

            if epoch % 2 == 0:  # train generator for one epoch
                max_len = max(lengths)
                embedding = embedding.to(device).long()[:, :max_len]
                speaker = speaker.to(device).long()
                fft = fft.to(device).float()[:, :max_len].transpose(2, 1)
                optG.zero_grad()
                gen_output = gen(embedding, speaker)
                loss_g1 = criterion(gen_output, fft)
                disc_out = disc(gen_output)
                real = Variable(Tensor(disc_out.size()[0], 1).fill_(1.0), requires_grad=False)
                loss_g2 = criterion(disc(gen_output), real)
                loss_g = loss_g1 + loss_g2
                loss_g.backward()
                optG.step()
                totalLoss_g += loss_g.item()
            else:
                # feed real spectrogram to discriminator
                max_len = max(lengths)
                fft = fft.to(device).float()[:, :max_len].transpose(2, 1)
                disc_output = disc(fft)
                real = Variable(Tensor(disc_output.size()[0], 1).fill_(1.0), requires_grad=False)
                loss_d1 = criterion(disc_output, real)
                # feed fake spectrogram to discriminator
                embedding = embedding.to(device).long()[:, :max_len]
                speaker = speaker.to(device).long()
                gen_output = gen(embedding, speaker)
                optD.zero_grad()
                disc_fake_out = disc(gen_output.detach())
                fake = Variable(Tensor(disc_fake_out.size()[0], 1).fill_(0.0), requires_grad=False)
                loss_d2 = criterion(disc_fake_out, fake)
                loss_d = coef*(loss_d1 + loss_d2)
                loss_d.backward()
                optD.step()
                totalLoss_d += loss_d.item()

        torch.cuda.empty_cache()
        lossRecord = "epoch: " + str(epoch + 1) + "  generator loss: " + str(totalLoss_g) + ' discriminator loss: ' + str(totalLoss_d)
        print(lossRecord)
        log.info(lossRecord)
        if (epoch + 1) % 5 == 0:
            torch.save(gen.state_dict(), output_path + 'gen_speaker_model_' + str(epoch + 1))
            torch.save(disc.state_dict(), output_path + 'disc_speaker_model_' + str(epoch + 1))
            log.info('Generator Model saved. ' + output_path + 'gen_speaker_model_' + str(epoch + 1))
            log.info('Discriminator Model saved. ' + output_path + 'disc_speaker_model_' + str(epoch + 1))

