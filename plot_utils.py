import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display
from hp import hp
import numpy as np
import os


def plot_spec(mspec, title_str, out_path):
  mspec = np.squeeze(mspec)
  fig, ax = plt.subplots()
  img = librosa.display.specshow(mspec, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax)
  ax.set(title=title_str)
  fig.colorbar(img, ax=ax, format="%+2.f dB")
  savepath = os.path.join(out_path, title_str + '.png')
  plt.savefig(savepath)
