from torch.utils import data
import torch.nn.functional as F
import torch
import torchaudio
import numpy as np
import crepe
import librosa
import pickle 
import os
from pathlib import Path

from spec_utils import get_mspec_from_array
from hp import hp
import random
       

class AudioTransformPipeline():
    def __init__(self, sample_rate, transform_dict=None, transform_args=None) -> None:
      self.sr = sample_rate
      self.vad = torchaudio.transforms.Vad(sample_rate=hp.sampling_rate, trigger_level=7.0)
      if transform_dict is not None:
        self.transform_dict = transform_dict
        self.transforms = self.init_transforms()
        if transform_args == None:
            self.transform_args = self.init_transform_args()
        else:
            assert self.transform_args['speed'] != None
            # assert(self.transform_args['speed']['min'] != None and self.transform_args['speed']['max'] != None)
            assert self.transform_args['noise'] != None
            assert(self.transform_args['scramble']['min'] != None and self.transform_args['scramble']['max'] != None)
            assert(self.transform_args['gain']['min'] != None and self.transform_args['gain']['max'] != None)
        self.speed_perturb = torchaudio.transforms.SpeedPerturbation(orig_freq=self.sr, factors=self.transform_args['speed'])
      else:
         self.transforms = None

    def Vad(self, x):
       return self.reverse(self.vad(self.reverse(self.vad(x))))
    
    def reverse(self, x):
        x, sr = torchaudio.sox_effects.apply_effects_tensor(x, sample_rate=self.sr, effects=[['reverse']], channels_first=True)
        return x
     
    def perturb_speed(self, x):
        # rate = random.uniform(self.transform_args['speed']['min'], self.transform_args['speed']['max'])
        # if rate == 1.0: # no change
        #     return x
        # # change speed and resample to original rate:
        # sox_effects = [
        #     ["speed", str(rate)],
        #     ["rate", str(self.sr)],
        # ]
        # x_speed_perturbed, _ = torchaudio.sox_effects.apply_effects_tensor(
        #     x, self.sr, sox_effects)
        # return x_speed_perturbed
        return  self.speed_perturb(x)[0]  #leave off returned "new time"

    def scramble(self, x):
      if random.random() <= self.transform_args['scramble']['prob']:
         return x
      #else: scramble
      num_slices = random.randint(self.transform_args['scramble']['min'], self.transform_args['scramble']['max'])
      slice_size = int(x.shape[1] / num_slices)
      slices = []
      for i in range(num_slices):
        start = i * slice_size
        end = x.shape[-1] if i == num_slices - 1 else (i + 1) * slice_size
        slices.append(x[:,start:end])
      random.shuffle(slices)
      return torch.cat(slices, 1)
    
    def adjust_gain(self, x):
    #   self.gain_adjust = lambda a,b: torchaudio.transforms.Vol(gain=random.uniform(a, b), gain_type="amplitude")
    # would be nice to figure out how to not declare the Vol object every time, but have randomness in the gain factor
      gain_adjuster = torchaudio.transforms.Vol(gain=random.uniform(self.transform_args['gain']['min'], self.transform_args['gain']['max']), 
                                            gain_type="amplitude")
      return gain_adjuster(x)
    
    def add_noise(self, x): 
       #x is a Tensor of shape [channels, samples]
       noise = torch.randn(size=x.size())
       idx = random.randint(a=0, b=len(self.transform_args['noise'])-1)
       snr = self.transform_args['noise'][idx].repeat(x.shape[0])
       return torchaudio.functional.add_noise(x, noise, snr=snr)   

    def init_transforms(self):
        transforms = []
        # The order of these if statements determines the order in which the augmentations are applied
        if self.transform_dict['speed']:
            transforms.append(self.perturb_speed)
        if self.transform_dict['reverse']:
            transforms.append(self.scramble)
        if self.transform_dict['gain']:
            transforms.append(self.adjust_gain)
        if self.transform_dict['scramble']:
            transforms.append(self.scramble)
        if self.transform_dict['noise']:
            transforms.append(self.add_noise)
        return transforms
    
    def init_transform_args(self):
        gain = {'min':0.9,
                'max':1.0
                }
        scramble = {'min':1,
                    'max':4,
                    'prob':0.5
                    }
        snr_dbs = torch.Tensor([80,60,20,10,3])
        speed = [0.9, 1.1, 1.0, 1.0, 1.0]
        # speed = {'min':0.9,
        #         'max':1.0
        #         }
        transform_args = {'speed':speed,
                          'gain':gain,
                          'scramble':scramble,
                          'noise':snr_dbs}
        return transform_args
    
    def compose_transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x
       

class AutoVCDataset(data.Dataset):

    def __init__(self, paths, spk_embs_root, len_crop, scale=None, shift=None, transform_dict=None) -> None:
        super().__init__()
        self.paths = paths
        self.spk_embs_root = spk_embs_root
        self.sr = hp.sampling_rate
        self.len_crop = len_crop
        self.ATP = AudioTransformPipeline(sample_rate=self.sr, transform_dict=transform_dict)
        if transform_dict:
           self.transform = True
        else:
           self.transform = False
        self.min_c = np.log2(50)
        self.max_c = np.log2(1000)
        
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        pth = self.paths[index]
        spk_id = pth.parent.stem
        if pth.suffix == '.wav':
          x, fs = torchaudio.load(pth) #resampling happens when converting to melspec
          if fs!=self.sr:
             x = torchaudio.functional.resample(x, orig_freq=fs, new_freq=self.sr)
          
          # Apply VAD to remove silences
          x = self.ATP.Vad(x)
          
          # apply augmentation transforms to .wav file
          if self.transform:
            x = self.ATP.compose_transform(x).detach().numpy()
          
          # Get source, TEGG, and EGG spectrograms using the different channels in x
          mspec_src, x_src = self.get_mspec(x[0, :], fs) # (N, n_mels)
          mspec_tegg, _ = self.get_mspec(x[1, :], fs) # (N, n_mels)
          mspec_egg, _ = self.get_mspec(x[2, :], fs) # (N, n_mels)
        
        elif pth.suffix == '.pt': mspec = torch.load(str(pth)) # (N, n_mels), no additional transforms
        else: print('file type not supported')
        
        # load speaker embedding
        emb_pth = self.spk_embs_root/spk_id/f"{pth.stem}_sse_emb.pt"
        try:
            if emb_pth.is_file():
              spk_emb = torch.load(emb_pth, map_location=torch.device('cpu'))
        except:
          spk_emb = torch.empty((1,hp.dim_emb))
          print('Embedding file not found. Embedding will be empty tensor')
        
        #f0 track
        step_size = int(1e3*hp.hop_length/hp.sampling_rate) #in ms
        _, f0, _, _ = crepe.predict(x_src, fs, viterbi=True, step_size=step_size)
        f0 = self.f0_normalization(f0)
        #extract RMSE
        rmse = librosa.feature.rms(y=x_src, frame_length=hp.fft_length, hop_length=hp.hop_length, center=True)[0]
        rmse = np.clip(rmse, 0, 1) #remove rmse outliers
        
        # try:
        #     assert mspec_src.shape == mspec_tegg.shape
        #     assert mspec_src.shape == mspec_egg.shape
        #     assert mspec_src.shape[0] == len(f0)
        #     assert mspec_src.shape[0] == len(rmse)
        # except:
        #    print('mel specs and conditioning vectors are not of the same length')

        # random crop everything in the same way
        mspec_src, mspec_tegg, mspec_egg, f0, rmse = self.random_crop(mspec_src, mspec_tegg, mspec_egg, torch.Tensor(f0), torch.Tensor(rmse))

        #one-hot encode conditioning vars
        f0_1hot, f0_i = self.quantize_f0_numpy(f0.detach().numpy())
        rmse_1hot, rmse_i = self.quantize_rmse_numpy(rmse.detach().numpy())

        #mconvert all to tensors
        f0_1hot = torch.Tensor(f0_1hot)
        f0_i = torch.Tensor(f0_i)
        rmse_1hot = torch.Tensor(rmse_1hot)
        rmse_i = torch.Tensor(rmse_i)

        return mspec_src, mspec_tegg, mspec_egg, spk_emb, (f0_1hot, f0_i), (rmse_1hot,rmse_i)

    def get_mspec(self, x, fs):
        m, x = get_mspec_from_array(x, input_sr=self.sr, is_hifigan=True, return_waveform=True) # (N, n_mels)
        return m, x
    
    def random_crop(self, mspec_src, mspec_tegg, mspec_egg, f0, rmse):
        #cprint(mspec.shape) 
        N, _ = mspec_src.shape # the same for both
        clen = self.len_crop
        if N < clen:
            # pad mspec, f0, and rmse
            n_pad = clen - N
            mspec_src = F.pad(mspec_src, (0, 0, 0, n_pad), value=mspec_src.min())
            mspec_tegg = F.pad(mspec_tegg, (0, 0, 0, n_pad), value=mspec_tegg.min())
            mspec_egg = F.pad(mspec_egg, (0, 0, 0, n_pad), value=mspec_egg.min())
            f0 = F.pad(f0, (0, n_pad), value=f0.min())
            rmse = F.pad(rmse, (0, n_pad), value=rmse.min())
        elif N > clen:
            crop_start = random.randint(0, N - clen)
            mspec_src = mspec_src[crop_start:crop_start+clen]
            mspec_tegg = mspec_tegg[crop_start:crop_start+clen]
            mspec_egg = mspec_egg[crop_start:crop_start+clen]
            f0 = f0[crop_start:crop_start+clen]
            rmse = rmse[crop_start:crop_start+clen]
        return mspec_src, mspec_tegg, mspec_egg, f0, rmse

    def f0_cent_normalization(self, x):
        c = np.log2(x+1e-6) - self.min_c
        index_nonzero = (c > 0)
        c[index_nonzero] = c[index_nonzero]/self.max_c
        c[index_nonzero] = np.clip(c[index_nonzero], 0, 1)
        return c
    
    def f0_normalization(self, f0):
        f0 = np.log(f0.astype(float).copy())
        index_nonzero = (f0 > 0)
        mean_f0 = np.mean(f0[index_nonzero])
        std_f0 = np.std(f0[index_nonzero])
        f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
        f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
        f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0 #shift to be between 0 and 1
        return f0

    def quantize_f0_numpy(self, x, num_bins=256):
        # x is normalized to be between 0 and 1
        x = x.astype(float).copy()
        uv = (x<=0)
        x[uv] = 0.0
        assert (x >= 0).all() and (x <= 1).all()
        x = np.round(x * (num_bins-1))
        x = x + 1 #make range from 1-257 for voiced
        x[uv] = 0.0 #make 0 for unvoiced / no pitch
        enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
        enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
        return enc, x.astype(np.int64)

    def quantize_rmse_numpy(self, x, num_bins=256):
        # x is normalized to be between 0 and 1
        x = x.astype(float).copy()
        assert (x >= 0).all() and (x <= 1).all()
        x = np.round(x * (num_bins-1))
        enc = np.zeros((len(x), num_bins), dtype=np.float32)
        enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
        return enc, x.astype(np.int64)

    def quantize_f0_torch(x, num_bins=256):
        # x is normalized to be between 0 and 1
        B = x.size(0)
        x = x.view(-1).clone()
        uv = (x<=0)
        x[uv] = 0
        assert (x >= 0).all() and (x <= 1).all()
        x = torch.round(x * (num_bins-1))
        x = x + 1
        x[uv] = 0
        enc = torch.zeros((x.size(0), num_bins+1), device=x.device)
        enc[torch.arange(x.size(0)), x.long()] = 1
        return enc.view(B, -1, num_bins+1), x.view(B, -1).long()

def get_loader(files, spk_embs_root, len_crop, batch_size=16, 
                num_workers=0, shuffle=False, scale=None, shift=None, transform_dict=None): # had to change num_workers to 0
    """Build and return a data loader."""
    dataset = AutoVCDataset(files, spk_embs_root, len_crop, scale=scale, shift=shift, transform_dict=transform_dict)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=shuffle,
                                  pin_memory=True) # set pin memory to True if training.
    # something is up with the default collate function in the dataloader and the way data samples are getting batched.  did a dimension flip?
    # see https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading for more info.
    # if you set batch size to 1 in hp.py (bypassing the current issue) you'll also see there is a dimensions-related bug further downstream
    return data_loader



def precompute_sse(spk_folders, device='cpu'):
    sse = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder').to(device)
    sse.eval()
    
    # mb = master_bar(spk_folders)
    spk_out_path = Path(hp.speaker_embedding_dir)
    spk_embs = {}
    os.makedirs(spk_out_path, exist_ok=True)
    print("[SPEAKER EMBEDDING] Precomputing/identifying speaker embeddings")
    # loop through all speakers
    for spk_folder in spk_folders:
        # mb.child.comment = f"processing speaker {spk_folder.stem} ({i} of {len(sample_uttrs)})"
        # print(f"processing speaker {spk_folder.stem})")
        os.makedirs(spk_out_path/f"{spk_folder.stem}", exist_ok=True)
        uttrs = list(spk_folder.iterdir())
        # sample_uttrs = random.sample(list(spk_folder.iterdir()), k=hp.n_uttr_per_spk_embedding)
        # embs = []
        for i, uttr_pth in enumerate(uttrs):
          # if precomputed, load embedding
          if (spk_out_path/f"{spk_folder.stem}"/f"{uttr_pth.stem}_sse_emb.pt").is_file(): 
              continue
              #embedding = torch.load(spk_out_path/f"{uttr_pth.stem}_sse_emb.pt")
          else: #compute embedding and save to file
            mel = sse.melspec_from_file(uttr_pth).to(device)
            if str(uttr_pth).endswith('.pt'): 
              raise NotImplementedError(("If spectrograms are not precomputed, please do not use pre-computed mel-spectrograms in args."))
            with torch.no_grad():
                embedding = sse(mel[None])[0] #get embedding from mspec
            torch.save(embedding, spk_out_path/f"{spk_folder.stem}"/f"{uttr_pth.stem}_sse_emb.pt")
          # embs.append(embedding.cpu())

    del sse
    torch.cuda.empty_cache()
        


