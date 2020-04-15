import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embed2spec import embed2spec
from dataset import NewDataset
from torch.utils.data import DataLoader
from multiscale_model import CNN
from multiscale_convolution import get_logger
import datetime

# Training of a simpler model, without GAN or multiscale convolution
# Just speaker embeddings and posteriograms are fed to 1D convolution layers
if __name__ == "__main__":
    arch = 'cnn'
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
    output_path = prefix + '/home/mansur/zerospeech/models/cnn_models/'
    device = "cuda"
    data = NewDataset(dataset)
    print('Data is ready')
    loader = DataLoader(data, batch_size=128, num_workers=8, shuffle=True)
    model = CNN().to(device)
    criterion = nn.MSELoss()  # first try to reconstruct the spectrum
    optG = optim.Adam(model.parameters())
    max_epoch = 300

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
            output = model(embedding, speaker)
            loss = criterion(output, fft)
            optG.zero_grad()
            loss.backward()
            optG.step()
            totalLoss += loss.item()
            lens += max_len
            counter += 1
        torch.cuda.empty_cache()
        del loss

        lossRecord = "epoch: " + str(epoch + 1) + "  loss: " + str(totalLoss) + "  mean length: " + str(lens / counter)
        print(lossRecord)
        log.info(lossRecord)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), output_path + 'speaker_model_' + str(epoch + 1))
            log.info('Model saved. ' + output_path + 'speaker_model_' + str(epoch + 1))

