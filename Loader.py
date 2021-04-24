import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import librosa
import numpy as np

class Loader(Dataset):
    def __init__(self, 
        folds_location: List
        ) -> None:
        sample_rate = 22050
        n_fft = 1024
        win_length = 512
        hop_length = 512
        n_mels = 128 
        self._walker = []
        self.length = []

        self.mel_spectrogram = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min = 0,
                f_max = 22050,
                power = 1,
                n_mels=n_mels,
        )
        self._parse_file(folds_location)

    def __len__(self):
        return len(self._walker)
    
    def _parse_file(self, folds_location: str) -> None:
        for fold_location in folds_location:
            archive = Path(fold_location)
            self._walker.extend(sorted(str(p) for p in Path(archive).glob("*.wav")))
    
    def __getitem__(self, index) -> [torch.Tensor,int]:
        # return self._walker[index]
        fileid = self._walker[index]
        label = torch.tensor(int(fileid.split('-')[1]))
        waveform, sample_rate = torchaudio.load(fileid)
        waveform = waveform.mean(0, keepdim=True)

        # self.length.append(waveform.shape[1])
        
        # padd = torch.zeros(1,512)
        # print(fileid)
        # print(waveform.shape)
        # if padd.shape[-1] < waveform.shape[-1]:
        #     indices = torch.arange(0, 512)
        #     padd = torch.index_select(waveform, 1, indices)
        # else:
        #     indices = torch.arange(0, waveform.shape[-1])
        #     padd.index_copy_(1,indices,waveform)
        spec =  self.mel_spectrogram(waveform)
        # padd = torch.zeros(1,128,128)

        padd = torch.zeros(1,128,128)
        # I have to check whether this is something that is correct.
        if padd.shape[-1] < spec.shape[-1]:
            indices = torch.arange(0, 128)
            padd = torch.index_select(spec, 2, indices)
        else:
            indices = torch.arange(0, spec.shape[-1])
            padd.index_copy_(2,indices,spec)

        return padd, label

if __name__ == "__main__":
    y = Loader(["fold1_20","fold2_20/"])
    dataload = DataLoader(y,batch_size=10)

    for i,j in dataload:
        print(i.shape)
        break
    # length_arr = np.array(y.length)
    # print(y.length)
    # print(np.min(length_arr))
    # print(np.max(length_arr))