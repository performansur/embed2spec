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
import logging
import datetime
import os




# multiscale CNN model with speaker embeddings
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        TOTAL_N_SPEAKERS = 102
        N = 512
        emb = N // 4
        out1 = N // 2
        out2 = N // 2
        out3 = N // 2
        out4 = N // 2
        out5 = N // 2
        out6 = N // 2
        out7 = N // 2 + 1
        self.speaker_embedding = torch.nn.Embedding(num_embeddings=TOTAL_N_SPEAKERS, embedding_dim=emb)
        self.audio_embedding = torch.nn.Embedding(num_embeddings=64, embedding_dim=emb)

        self.c1_1 = nn.Conv1d(2*emb, out1, 1, padding=0, padding_mode='same')
        self.c1_2 = nn.Conv1d(2*emb, out1, 3, padding=1, padding_mode='same')
        self.c1_3 = nn.Conv1d(2*emb, out1, 5, padding=2, padding_mode='same')
        self.c1_4 = nn.Conv1d(2*emb, out1, 7, padding=3, padding_mode='same')
        self.c1_5 = nn.Conv1d(2*emb, out1, 9, padding=4, padding_mode='same')
        self.c1_6 = nn.Conv1d(2*emb, out1, 11, padding=5, padding_mode='same')
        self.c1_7 = nn.Conv1d(2*emb, out1, 13, padding=6, padding_mode='same')
        self.c1_8 = nn.Conv1d(2*emb, out1, 15, padding=7, padding_mode='same')

        self.c2_1 = nn.Conv1d(out1, out2, 1, padding=0, padding_mode='same')
        self.c2_2 = nn.Conv1d(out1, out2, 3, padding=1, padding_mode='same')
        self.c2_3 = nn.Conv1d(out1, out2, 5, padding=2, padding_mode='same')
        self.c2_4 = nn.Conv1d(out1, out2, 7, padding=3, padding_mode='same')

        self.c3_1 = nn.Conv1d(out2, out3, 1, padding=0, padding_mode='same')
        self.c3_2 = nn.Conv1d(out2, out3, 3, padding=1, padding_mode='same')
        self.c3_3 = nn.Conv1d(out2, out3, 5, padding=2, padding_mode='same')
        self.c3_4 = nn.Conv1d(out2, out3, 7, padding=3, padding_mode='same')

        self.c4_1 = nn.Conv1d(out3, out4, 1, padding=0, padding_mode='same')
        self.c4_2 = nn.Conv1d(out3, out4, 3, padding=1, padding_mode='same')

        self.c5 = nn.Conv1d(out4, out5, 1, padding=0, padding_mode='same')
        self.c6 = nn.Conv1d(out5, out6, 1, padding=0, padding_mode='same')
        self.c7 = nn.Conv1d(out6, out7, 1, padding=0, padding_mode='same')

        self.bn1 = nn.BatchNorm1d(num_features=out1)
        self.bn2 = nn.BatchNorm1d(num_features=out2)
        self.bn3 = nn.BatchNorm1d(num_features=out3)
        self.bn4 = nn.BatchNorm1d(num_features=out4)
        self.bn5 = nn.BatchNorm1d(num_features=out5)
        self.bn6 = nn.BatchNorm1d(num_features=out6)

        self.W1 = torch.nn.Parameter(torch.randn(1, 8), requires_grad=True)
        self.W2 = torch.nn.Parameter(torch.randn(1, 4), requires_grad=True)
        self.W3 = torch.nn.Parameter(torch.randn(1, 4), requires_grad=True)
        self.W4 = torch.nn.Parameter(torch.randn(1, 2), requires_grad=True)

    def forward(self, x, speaker_id):
        x = self.audio_embedding(x).transpose(2, 1)
        x = torch.cat([x, self.speaker_embedding(speaker_id).unsqueeze(-1).expand(x.shape)], dim=1)

        x = torch.stack([F.leaky_relu(self.bn1(self.c1_1(x))), F.leaky_relu(self.bn1(self.c1_2(x))),
                         F.leaky_relu(self.bn1(self.c1_3(x))), F.leaky_relu(self.bn1(self.c1_4(x))),
                         F.leaky_relu(self.bn1(self.c1_5(x))), F.leaky_relu(self.bn1(self.c1_6(x))),
                         F.leaky_relu(self.bn1(self.c1_7(x))), F.leaky_relu(self.bn1(self.c1_8(x)))], dim=-1)
        x = F.linear(x, F.softmax(self.W1, dim=1)).squeeze(-1)

        x = torch.stack([F.leaky_relu(self.bn2(self.c2_1(x))), F.leaky_relu(self.bn2(self.c2_2(x))),
                         F.leaky_relu(self.bn2(self.c2_3(x))), F.leaky_relu(self.bn2(self.c2_4(x)))], dim=-1)
        x = F.linear(x, F.softmax(self.W2, dim=1)).squeeze(-1)

        x = torch.stack([F.leaky_relu(self.bn3(self.c3_1(x))), F.leaky_relu(self.bn3(self.c3_2(x))),
                         F.leaky_relu(self.bn3(self.c3_3(x))), F.leaky_relu(self.bn3(self.c3_4(x)))], dim=-1)
        x = F.linear(x, F.softmax(self.W3, dim=1)).squeeze(-1)

        x = torch.stack([F.leaky_relu(self.bn4(self.c4_1(x))), F.leaky_relu(self.bn4(self.c4_2(x)))], dim=-1)
        x = F.linear(x, F.softmax(self.W4, dim=1)).squeeze(-1)

        x = F.leaky_relu(self.bn5(self.c5(x)))
        x = F.leaky_relu(self.bn6(self.c6(x)))
        x = self.c7(x)
        return x

# simpler model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        in_channel, out = 256
        n_speakers = 102
        emb = 64
        self.speaker_embedding = torch.nn.Embedding(num_embeddings=n_speakers, embedding_dim=emb)
        self.audio_embedding = torch.nn.Embedding(num_embeddings=64, embedding_dim=emb)

        self.conv1 = nn.Conv1d(in_channel, in_channel, kernel_size=11, padding=5,
                                 padding_mode='same')
        self.conv2 = nn.Conv1d(in_channel, in_channel, kernel_size=7, padding=3,
                                 padding_mode='same')
        self.conv3 = nn.Conv1d(in_channel, in_channel, 3, padding=1, padding_mode='same')
        self.conv4 = nn.Conv1d(in_channel, in_channel, 3, padding=1, padding_mode='same')
        self.conv5 = nn.Conv1d(in_channel, in_channel, 1, padding=0, padding_mode='same')
        self.conv6 = nn.Conv1d(in_channel, in_channel, 1, padding=0, padding_mode='same')
        self.conv7 = nn.Conv1d(in_channel, in_channel, 1, padding=0, padding_mode='same')
        self.conv8 = nn.Conv1d(in_channel, in_channel+1, 1, padding=0, padding_mode='same')
        self.bn = nn.BatchNorm1d(num_features=out)  # not sure about this input
        self.lin = nn.Linear(in_features=in_channel, out_features=in_channel+1, bias=True)

    def forward(self, x, speaker_id):
        x = self.audio_embedding(x).transpose(2, 1)
        x = torch.cat([x, self.speaker_embedding(speaker_id).unsqueeze(-1).expand(x.shape)], dim=1)
        x = F.leaky_relu(self.bn(self.conv1(x)))
        x = F.leaky_relu(self.bn(self.conv2(x)))
        x = F.leaky_relu(self.bn(self.conv3(x)))
        x = F.leaky_relu(self.bn(self.conv4(x)))
        x = F.leaky_relu(self.bn(self.conv5(x)))
        x = F.leaky_relu(self.bn(self.conv6(x)))
        x = F.leaky_relu(self.bn(self.conv7(x)))
        x = self.conv8(x)

        return x



def get_logger(logger_name, filename, create_file=True):

    log = logging.getLogger(logger_name)
    log.setLevel(level=logging.INFO)

    if create_file:
        # create file handler for logger.
        os.chdir('./logs/')
        fh = logging.FileHandler(filename)
        fh.setLevel(level=logging.INFO)

    if create_file:
        log.addHandler(fh)

    return log


'''
Before each run SPECIFY followings:
1) Model Architecture, i.e ms_cnn or ms_cnn_gan (variable arch)
2) Embedding Type, i.e one-hot_rep, one-hot_norep, softmax_rep, softmax_norep (variable embed_type)
3) Embedding Crop, i.e max, min, whole (variable embed_crop)
4) Dataset, i.e voice, unit, train (unit + voice) (variable dataset)
5) Run Name. Default empty string. Fill it if you want to add additional info
'''

# Training without GAN

if __name__ == "__main__":
    arch = 'msCnn'
    embed_type = 'softmaxRep'
    embed_crop = 'meanCrop'
    dataset = 'combined'
    run_name = ''
    log_loc = './logs/'
    now = datetime.datetime.now()
    time = str(now.day) + '.' + str(now.month) + '.' + str(now.year) + '__' + str(now.hour) + ':' + str(now.minute)
    logFileName = arch + '_' + embed_type + '_' + embed_crop + '_' + dataset + '_' + run_name + '_' + time + '.log'
    log = get_logger('zerospeech', logFileName)

    #server = 'gpu1'
    server = 'gpu2'
    if server == 'gpu1':
        prefix = '/mnt/gpu2'
    else:
        prefix = ''
    output_path = prefix + '/home/mansur/zerospeech/models/ms_cnn_speaker_models/'
    device = "cuda"
    data = NewDataset(dataset)
    print('Data is ready')
    loader = DataLoader(data, batch_size=256, num_workers=16, shuffle=True)
    #model_path = output_path + 'ms_model_500'
    model = Decoder().to(device)
    #model.load_state_dict(torch.load(model_path,  map_location=device))
    criterion = nn.MSELoss()  # first try to reconstruct the spectrum
    optG = optim.Adam(model.parameters())
    max_epoch = 900
    totalLoss = 0

    print('Start Training')
    for epoch in range(max_epoch):
        totalLoss = 0
        lens = 0.0
        counter = 0.0
        for speaker, embedding, fft, lengths, _, _ in loader:
            max_len = int(lengths.float().mean())
            embedding = embedding.to(device).long()[:, :max_len]
            speaker = speaker.to(device).long()
            fft = fft.to(device).float()[:, :max_len].transpose(2, 1)
            optG.zero_grad()
            output = model(embedding, speaker)
            loss = criterion(output, fft)

            lens += max_len
            counter += 1
            loss.backward()
            optG.step()
            totalLoss += loss.item()
        torch.cuda.empty_cache()
        del loss

        lossRecord = "epoch: " + str(epoch + 1) + "  loss: " + str(totalLoss) + "  mean length: " + str(lens/counter)
        print(lossRecord)
        log.info(lossRecord)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), output_path + 'speaker_model_' + str(epoch + 1))
            log.info('Model saved. ' + output_path + 'speaker_model_' + str(epoch + 1))
