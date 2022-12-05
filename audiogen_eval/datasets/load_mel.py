import librosa
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import json
import torchaudio
from tqdm import tqdm


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datadir,
        _stft,
        sr=16000,
        fbin_mean=None,
        fbin_std=None,
        augment=False,
        limit_num=None,
    ):
        self.datalist = [os.path.join(datadir, x) for x in os.listdir(datadir)]
        if limit_num is not None:
            self.datalist = self.datalist[:limit_num]
        self._stft = _stft
        self.sr = sr
        self.augment = augment

        if fbin_mean is not None:
            self.fbin_mean = fbin_mean[..., None]
            self.fbin_std = fbin_std[..., None]
        else:
            self.fbin_mean = None
            self.fbin_std = None

    def __getitem__(self, index):
        filename = self.datalist[index]
        mel, energy, waveform = self.get_mel_from_file(filename)

        # if(self.fbin_mean is not None):
        #     mel = (mel - self.fbin_mean) / self.fbin_std

        return waveform, filename

    def __len__(self):
        return len(self.datalist)

    def get_mel_from_file(self, audio_file):
        audio, file_sr = torchaudio.load(audio_file)
        audio = audio - audio.mean()

        if file_sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=file_sr, new_freq=self.sr
            )

        if self._stft is not None:
            melspec, energy = self.get_mel_from_wav(audio[0, ...])
        else:
            melspec, energy = None, None

        return melspec, energy, audio

    def get_mel_from_wav(self, audio):
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)

        # =========================================================================
        # Following the processing in https://github.com/v-iashin/SpecVQGAN/blob/5bc54f30eb89f82d129aa36ae3f1e90b60e73952/vocoder/mel2wav/extract_mel_spectrogram.py#L141
        melspec, energy = self._stft.mel_spectrogram(audio, normalize_fun=torch.log10)
        melspec = (melspec * 20) - 20
        melspec = (melspec + 100) / 100
        melspec = torch.clip(melspec, min=0, max=1.0)
        # =========================================================================
        # Augment
        # if(self.augment):
        #     for i in range(1):
        #         random_start = int(torch.rand(1) * 950)
        #         melspec[0,:,random_start:random_start+50] = 0.0
        # =========================================================================
        melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        return melspec, energy


def load_npy_data(loader):
    new_train = []
    for waveform, filename in tqdm(loader):
        batch = batch.float().numpy()
        new_train.append(
            batch.reshape(
                -1,
            )
        )
    new_train = np.array(new_train)
    return new_train


if __name__ == "__main__":
    path = "/scratch/combined/result/ground/00294 harvest festival rumour 1_mel.npy"
    temp = np.load(path)
    print("temp", temp.shape)
