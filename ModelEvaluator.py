import torch
import torch.nn as nn
from embed2spec import embed2spec
from dataset import NewDataset
import numpy as np
import librosa as lib
from scipy.io.wavfile import write
from multiscale_model import MultiScale
from multiscale_convolution import Decoder


class ModelEvaluator:  # This class creates audio file using trained model so that we can listen the result

    def __init__(self, model_path, dataset):
        self.model = Decoder()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.data = NewDataset(dataset)
        found = False
        i = 1
        while not found:
            if model_path[-i] == '_':
                self.name = model_path[-i+1:]
                found = True
            i += 1

    def play_output(self, output_path, key, speaker_id=None, iter=100, fs=16000, return_original=False):
        index = 0
        for k in self.data.keys:
            if key in k:
                break
            else:
                index += 1

        original_id, embedding, fft, n_frames_before_pad, mean, std = self.data.__getitem__(index)
        if speaker_id is None:
            speaker_id = int(original_id)
        speaker_id = torch.tensor([speaker_id]).long()
        embedding = torch.from_numpy(embedding[:n_frames_before_pad]).unsqueeze(0).long()
        output = self.model(embedding, speaker_id).squeeze(0)
        output = output.detach().numpy()
        mean = np.repeat(mean, n_frames_before_pad, axis=1)
        std = np.repeat(std, n_frames_before_pad, axis=1)
        output = np.multiply(output, std) + mean


        ''''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(output.shape[0]//20, 12))
        #plt.subplot(1, 2, 1)
        plt.matshow(output, fignum=False)
        #plt.subplot(1, 2, 2)
        #plt.matshow(embedding[0,:,:].T, fignum=False)
        plt.show()
        '''

        grif_out = lib.core.griffinlim(np.exp(output), n_iter=iter, hop_length=160, win_length=512)
        write(output_path + key + '_' + self.name + '.wav', fs, grif_out)

        """
        speaker = torch.tensor([int(speaker_id)])
        index = self.data.keys.index(key)
        current_embedding = self.data.embedding[index]
        current_embedding = torch.from_numpy(current_embedding).unsqueeze(0).transpose(2, 1).float()
        output = self.model.alt_forward(current_embedding, speaker)
        print(output.shape)
        output = output.reshape(output.shape[1], output.shape[2])
        output = output.detach().numpy()
        '''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(output.shape[0]//20, 12))
        plt.subplot(1, 2, 1)
        plt.matshow(output, fignum=False)
        plt.subplot(1, 2, 2)
        plt.matshow(self.current_embedding[0,:,:].T, fignum=False)
        plt.show()'''

        grif_out = lib.core.griffinlim(np.exp(output), n_iter=iter, hop_length=160, win_length=512)
        write(output_path + key + '_' + self.name + '.wav', fs, grif_out)
        """


if __name__=='__main__':
    key = 'V002_1725974875_001'
    output_path = '/home/mansur/zerospeech/model_outputs/ms_cnn_gan/'
    model_path = '/home/mansur/zerospeech/models/ms_cnn_gan_speaker_models/gen_speaker_model_' + str(120)
    evaluator = ModelEvaluator(model_path=model_path, dataset='voice')
    evaluator.play_output(output_path=output_path, key=key)



