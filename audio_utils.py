import os
import speechbrain as sb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
from hp import hp
import numpy as np
from spec_utils import get_mspec_from_array
from IPython.display import (
    Audio, display, clear_output)

class Vocoder(nn.Module):
    """Hi-Fi GAN Vocoder network."""
    def __init__(self):
        super(Vocoder, self).__init__()
        self.vocoder = sb.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
        self.resampler = torchaudio.transforms.Resample(hp.sampling_rate, hp.original_sr)

    def forward(self, x):  # x is batch of mel-spectograms
      wavs = self.vocoder.decode_batch(x.transpose(2,1)) # x needs to have shape [batch x mels x T]
      wavs = wavs.squeeze(1)
        #after the above line, shape is (samples, channels)
      return wavs
    
    def resample(self,wav):
      return self.resampler(wav)

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {torchaudio.info(path)}")

def save_wavs(wav, path):
  torchaudio.save(path, wav, hp.original_sr)
  # inspect_file(path)

def format_wavs(wavs_true, wavs_pred):
  c = wavs_pred.shape[0]
  pairs = [torch.stack((wavs_true[i,:], wavs_pred[i,:]),dim=0) for i in range(c)]
  return pairs

def resample(wav):
   torchaudio.resample
