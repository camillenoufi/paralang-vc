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

    def forward(self, x):  # x is batch of mel-spectograms   
      wavs = self.vocoder.decode_batch(x) # x needs to have shape [batch x mels x T]
      wavs = torch.transpose(wavs.squeeze(1))
        #after the above line, shape is (samples, channels)
      return wavs

def save_wavs(wavs, path):
  torchaudio.save(path, wavs, hp.sampling_rate)
