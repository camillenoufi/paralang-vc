from torch.utils import data
import torch.nn.functional as F
import torch
import numpy as np
import crepe
import librosa
import pickle 
import os
from spec_utils import get_mspec_from_array
from hp import hp
import random
       
class AutoVCDataset(data.Dataset):

    def __init__(self, paths, spk_embs, len_crop, scale=None, shift=None) -> None:
        super().__init__()
        self.paths = paths
        self.spk_embs = spk_embs
        self.len_crop = len_crop
        # assert jitter % 32 == 0, "Jitter must be divisible by 32"
        # self.jitter_choices = list(range(0, jitter+1, 32))
        if scale is not None and shift is not None:
            self.norm_mel = lambda x: (x + shift) / scale
            self.denorm_mel = lambda x: (x*scale) - shift
        else:
            self.norm_mel = lambda x: x
            self.denorm_mel = lambda x: x

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        pth = self.paths[index]
        if pth.suffix == '.pt': mspec = torch.load(str(pth)) # (N, n_mels)
        else: 
            x, fs = librosa.load(pth, sr=hp.sample_rate)
            mspec = get_mspec_from_array(x=x, input_sr=fs, is_hifigan=True, return_waveform=True) # (N, n_mels)
        mspec, y = self.random_crop(mspec)
        spk_id = pth.parent.stem
        spk_emb = self.spk_embs[spk_id]
        mspec = self.norm_mel(mspec)

        #variables added by Camille:
        #f0 track
        step_size = int(1e3*hp.hop_length/hp.sampling_rate) #in ms
        _, f0, _, _ = crepe.predict(y, fs, viterbi=True, step_size=step_size)
        #extract RMSE
        rmse = librosa.feature.rmse(x, frame_length=hp.fft_length, hop_length=hp.hop_length, center=True)
        #add some sort of assert that the f0 and rmse vectors are the same length as the mspec
        assert mspec.shape[0] == len(f0)
        assert mspec.shape[0] == len(rmse)

        return mspec, spk_emb

    def random_crop(self, mspec):
        N, _ = mspec.shape
        clen = self.len_crop
        if N < clen:
            # pad mspec
            n_pad = clen - N
            mspec = F.pad(mspec, (0, 0, 0, n_pad), value=mspec.min())
        elif N > clen:
            crop_start = random.randint(0, N - clen)
            mspec = mspec[crop_start:crop_start+clen]
        return mspec

def get_loader(files, spk_embs, len_crop, batch_size=16, 
                num_workers=8, shuffle=False, scale=None, shift=None):
    """Build and return a data loader."""
    
    dataset = AutoVCDataset(files, spk_embs, len_crop, scale=scale, shift=shift)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=shuffle, pin_memory=shuffle) # set pin memory to True if training.
    return data_loader


