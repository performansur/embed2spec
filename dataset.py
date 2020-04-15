from torch.utils import data
import torch
import os
import pandas
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

# TODO LEARN MAX SIZE OF THE DATA!
max_len = 860
print(os.getcwd())

#server = 'gpu1'
server = 'gpu2'


class DecoderDataset(data.Dataset):  # This dataset is similar to one below but it is only voice set and doesn't use speaker info
    def __init__(self, mode,
                 embedding_folder="/home/mansur/zerospeech/train/",
                 fft_folder="/home/mansur/zerospeech/preprocessed/",
                 segments_folder="/home/yusuf/exps/zerospeech19/data/", debug=False):

        if server == 'gpu1':
            prefix = '/mnt/gpu2'
        else:
            prefix = ''

        segments_folder = prefix + segments_folder
        self.fft_folder = prefix + fft_folder + mode + "/fft/"
        self.segments = pandas.read_csv(segments_folder + mode + "/segments",
                                        sep=" ", index_col=0, header=None)

        embedding_folder = prefix + embedding_folder + mode + "_english_train/"
        embeddings = np.load(embedding_folder + "batuhan34.npy")
        embedding_offsets = np.load(embedding_folder + "offsets_data.npy")
        segmented_embedding = []
        beg = 0
        for i in range(len(embedding_offsets)):
            segmented_embedding.append(embeddings[beg:embedding_offsets[i]])
            beg = embedding_offsets[i]
        segmented_embedding.append(embeddings[embedding_offsets[i]:])

        with open(embedding_folder + "keys_data.txt", "r") as k:
            self.keys = k.read().split("\n")
        self.speaker_ids = np.unique([int(key[1:4]) for key in self.keys]).__len__()  # TODO new convention txt format
        self.embedding = segmented_embedding

        self.debug = debug

        #self.data = []
        #for index in range(len(self.keys)):
        #    self.data.append(self._real_get_item(index))

    def _real_get_item(self, index):
        key = self.keys[index]
        speaker_type, speaker_id = key[0], int(key[1:4])  # TODO fix after the new speaker naming convention
        fft = np.load(self.fft_folder + key + ".npy").T
        embedding = self.embedding[index]
        # todo onehot code here
        if self.debug:
            self._debug(embedding, fft)
        t = embedding.shape[0]
        f, c_f = fft.shape
        dif = f-t
        if dif > 0:
            start = dif//2
            end = dif - start
            fft = fft[start:-end]
        try:
            assert fft.shape[0] == embedding.shape[0]
        except AssertionError:
            print(key)

        n_frames_before_pad = min(t, max_len)

        reps = int(np.ceil(max_len / t))
        embedding = np.repeat(embedding, repeats=reps, axis=0)
        fft = np.repeat(fft, repeats=reps, axis=0)
        embedding = embedding[:max_len, :]
        fft = fft[:max_len, :]

        return speaker_id, embedding.T, fft.T, n_frames_before_pad

    def __len__(self):
        return self.keys.__len__()

    def __getitem__(self, index):
        return self._real_get_item(index) #self.data[index]

    @staticmethod
    def _debug(embedding, fft):
        plt.figure(figsize=(fft.shape[0]//20, 12))
        plt.subplot(1, 2, 1)
        plt.matshow(fft.T, fignum=False)
        plt.subplot(1, 2, 2)
        plt.matshow(embedding.T, fignum=False)
        plt.show()

    @staticmethod
    def _get_one_hot(embedding):
        one_hot = np.argmax(embedding, axis=1)
        embed = np.zeros_like(embedding)
        embed[np.arange(one_hot.size), one_hot] = 1
        return embed


class NewDataset(data.Dataset):  # This dataset also uses speaker information
    def __init__(self, mode,
                 path="/ssd1/mansur/zerospeech/2019/dataset/train_normalized/", debug=False):

        if mode == "combined":
            mode = ["voice", "unit"]
        else:
            mode = [mode]

        if server == 'gpu1':
            prefix = '/mnt/gpu2'
        else:
            prefix = ''
        with open(path + "speaker_ids.txt", "r") as f:
            self.speaker_ids = f.read().split()

        self.segments = []
        self.embeddings = []
        self.keys = []
        self.speakers = []
        self.mean_keys = []
        self.std_keys = []
        for m in mode:
            self.segments.append(pandas.read_csv(path + m + "/segments", sep=" ", index_col=0, header=None))

            embeddings = np.load(path + m + "/batuhan77.npy")
            embedding_offsets = np.load(path + m + "/offsets_data.npy")
            segmented_embedding = []
            beg = 0
            for i in range(len(embedding_offsets)):
                segmented_embedding.append(embeddings[beg:embedding_offsets[i]])
                beg = embedding_offsets[i]
            segmented_embedding.append(embeddings[embedding_offsets[i]:])

            self.embeddings.extend(segmented_embedding)

            with open(path + m + "/keys_data.txt", "r") as k:
                keys = k.read().split("\n")
                speakers = [self.speaker_ids.index(k[:4]) for k in keys]
                mean_keys = [path + m + "/fft/" + k + "_mean.npy" for k in keys]
                std_keys = [path + m + "/fft/" + k + "_std.npy" for k in keys]
                keys = [path + m + "/fft/" + k + ".npy" for k in keys]
                self.keys.extend(keys)
                self.speakers.extend(speakers)
                self.mean_keys.extend(mean_keys)
                self.std_keys.extend(std_keys)

        self.debug = debug

    def _real_get_item(self, index):  # mean and std included
        speaker = self.speakers[index]
        fft = np.load(self.keys[index]).T
        embedding = self.embeddings[index]
        embedding = self._get_one_hot(embedding)
        mean = np.load(self.mean_keys[index])
        std = np.load(self.std_keys[index])
        # todo onehot code here
        if self.debug:
            self._debug(embedding, fft)
        t = embedding.shape[0]
        f, c_f = fft.shape
        dif = f-t
        if dif > 0:
            start = dif//2
            end = dif - start
            fft = fft[start:-end]

        try:
            assert fft.shape[0] == embedding.shape[0]
        except AssertionError:
            print(self.keys[index])

        #
        n_frames_before_pad = min(t, max_len)
        reps = int(np.ceil(max_len / t))
        embedding = np.repeat(embedding, repeats=reps, axis=0)
        fft = np.repeat(fft, repeats=reps, axis=0)
        embedding = embedding[:max_len]
        fft = fft[:max_len, :]


        return speaker, embedding, fft, n_frames_before_pad, mean, std

    def __len__(self):
        return self.keys.__len__()

    def __getitem__(self, index):
        return self._real_get_item(index) #self.data[index]

    @staticmethod
    def _debug(embedding, fft):
        plt.figure(figsize=(fft.shape[0]//20, 12))
        plt.subplot(1, 2, 1)
        plt.matshow(fft.T, fignum=False)
        plt.subplot(1, 2, 2)
        plt.matshow(embedding.T, fignum=False)
        plt.show()

    @staticmethod
    def _get_one_hot(embedding):
        one_hot = np.argmax(embedding, axis=1)
        return one_hot


if __name__ == "__main__":
    dataset = NewDataset("combined", debug=False)
    for s, e, f, n, m, v in dataset:
        print(n)
    e, f, n = dataset[random.randint(0, len(dataset))]
    print(e.shape)
    print(f.shape)
