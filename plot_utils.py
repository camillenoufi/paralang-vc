import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display
from hp import hp
import numpy as np
import os
import io


def plot_spec(mspecs, title_str, out_path):
  
  mspecs = [np.transpose(np.squeeze(m.to('cpu').detach().numpy())) for m in mspecs]
  (speech, EGG, tEGG, pred_EGG, pred_tEGG, postnet) = tuple(mspecs)

  fig, [[ax1, ax2],[ax3, ax4], [ax5, ax6]] = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(16,16))

  img = librosa.display.specshow(speech, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax1)
  ax1.set(title='Speech')
  fig.colorbar(img, ax=ax1, format="%+2.f dB")

  img = librosa.display.specshow(EGG, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax2)
  ax2.set(title='EGG')
  fig.colorbar(img, ax=ax2, format="%+2.f dB")

  img = librosa.display.specshow(tEGG, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax3)
  ax3.set(title='tEGG')
  fig.colorbar(img, ax=ax3, format="%+2.f dB")

  img = librosa.display.specshow(pred_EGG, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax4)
  ax4.set(title='EGG Prediction')
  fig.colorbar(img, ax=ax4, format="%+2.f dB")
  
  img = librosa.display.specshow(pred_tEGG, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax5)
  ax5.set(title='tEGG Prediction')
  fig.colorbar(img, ax=ax5, format="%+2.f dB")

  img = librosa.display.specshow(postnet, hop_length=hp.hop_length, x_axis='time', y_axis='mel', ax=ax6)
  ax6.set(title='PostNet')
  fig.colorbar(img, ax=ax6, format="%+2.f dB")
  
  fig.suptitle(title_str)
  savepath = os.path.join(out_path, title_str + '.png')
  plt.savefig(savepath)

  return fig


def spec_to_tensorboard(mspecs, title_str, out_path):
  fig = plot_spec(mspecs, title_str, out_path)
  fig.canvas.draw()
  image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  image = image / 255.0
  image = np.swapaxes(image, 0, 2)
  return image
