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
    def __init__(self, out_path, writer):
        super(Vocoder, self).__init__()
        self.vocoder = sb.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
        # self.resampler = torchaudio.transforms.Resample(hp.sampling_rate, hp.original_sr)
        self.writer = writer
        self.out_path = out_path

    def forward(self,x):
      wavs = self.vocoder.decode_batch(x.transpose(2,1)) # x needs to have shape [batch x mels x T]
      wavs = wavs.squeeze(1)
        #after the above line, shape is (samples, channels)
      return wavs

    def vocode(self, x_src, x_tgt, epoch, i, valid=False):  # x is batch of mel-spectograms
      wavs_true = self.forward(x_src)
      wavs_pred = self.forward(x_tgt.squeeze(1))
      wavs = self.format_wavs(wavs_true, wavs_pred)
      for j,wav in enumerate(wavs):
          self.writer.add_audio(f'valid/wavs_E{self.epoch}_{i}_true', wav[0,:].unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
          self.writer.add_audio(f'valid/wavs_E{self.epoch}_{i}_pred', wav[1,:].unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
          if valid:
             fn = os.path.join(self.out_path,f'wavs_E{epoch}_v{j}.wav')
          else:
             fn = os.path.join(self.out_path,f'wavs_E{epoch}_{j}.wav')
          self.save_wavs(wav, fn)
    
    def inspect_file(self, path):
      print("-" * 10)
      print("Source:", path)
      print("-" * 10)
      print(f" - File size: {os.path.getsize(path)} bytes")
      print(f" - {torchaudio.info(path)}")

    def save_wavs(self, wav, path):
      torchaudio.save(path, wav, hp.original_sr)
      # self.inspect_file(path)

    def format_wavs(self, wavs_true, wavs_pred):
      c = wavs_pred.shape[0]
      pairs = [torch.stack((wavs_true[i,:], wavs_pred[i,:]),dim=0) for i in range(c)]
      return pairs

    
    # def resample(self,wav):
    #   return self.resampler(wav)









