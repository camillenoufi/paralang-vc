import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display
from hp import hp
import numpy as np
import os
import io


def plot_spec(mspec, title_str, out_path):
  mspec = np.squeeze(mspec)
  fig, ax = plt.subplots()
  img = librosa.display.specshow(mspec, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax)
  ax.set(title=title_str)
  fig.colorbar(img, ax=ax, format="%+2.f dB")
  savepath = os.path.join(out_path, title_str + '.png')
  plt.savefig(savepath)


def spec_to_tensorboard(mspec):
  mspec = np.squeeze(mspec)
  fig, ax = plt.subplots()
  img = librosa.display.specshow(mspec, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax)
  fig.colorbar(img, ax=ax, format="%+2.f dB")
  fig.canvas.draw()
  image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  image = image / 255.0
  image = np.swapaxes(image, 0, 2)
  return image
