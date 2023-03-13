from torch.utils import data
import torch.nn.functional as F
import torch
import numpy as np
import crepe
import librosa
import pickle 
import os
from pathlib import Path

from spec_utils import get_mspec_from_array
from hp import hp
import random
       
class AutoVCDataset(data.Dataset):

    def __init__(self, paths, spk_embs_root, len_crop, scale=None, shift=None, transform=False) -> None:
        super().__init__()
        self.paths = paths
        self.spk_embs_root = spk_embs_root
        self.len_crop = len_crop
        self.transform = transform
        # assert jitter % 32 == 0, "Jitter must be divisible by 32"
        # self.jitter_choices = list(range(0, jitter+1, 32))

        # Replacements for lambda function to fix AttributeError
        global norm_mel
        def norm_mel(x):
            return (x + shift) / scale
        global denorm_mel
        def denorm_mel(x):
            return (x * scale) - shift
        global identity
        def identity(x):
            return x

        if scale is not None and shift is not None:
            self.norm_mel = norm_mel
            self.denorm_mel = denorm_mel
        else:
            self.norm_mel = identity
            self.denorm_mel = identity     
        

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        pth = self.paths[index]
        spk_id = pth.parent.stem
        if pth.suffix == '.wav':
          x, fs = librosa.load(pth, sr=hp.sampling_rate, mono=False)
          if self.transform: # do augmentations to .wav files directly so they will propogate to mspecs and conditioning vars
            x = self.apply_transforms(x,fs)
          
          # Get source, TEGG, and EGG spectrograms using the different channels in x
          mspec_src, x_src = get_mspec_from_array(x=x[0, :], input_sr=fs, is_hifigan=True, return_waveform=True) # (N, n_mels)
          mspec_tegg, x_tegg = get_mspec_from_array(x=x[1, :], input_sr=fs, is_hifigan=True, return_waveform=True) # (N, n_mels)
          mspec_egg, x_egg = get_mspec_from_array(x=x[2, :], input_sr=fs, is_hifigan=True, return_waveform=True) # (N, n_mels)
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
        
        assert mspec_src.shape == mspec_tegg.shape
        assert mspec_src.shape == mspec_egg.shape
        assert mspec_src.shape[0] == len(f0)
        assert mspec_src.shape[0] == len(rmse)

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

    def apply_transforms(self, x, fs, transforms=None):
      # TO-DO directly on .wav. (ALL CHANNELS!)
      # scale / stretch / shrink by minor factor
      # boost / lower volume
      # time-reversal
      # simple chop / scramble
      return x

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

    def quantize_one_hot_torch(x, num_bins=256):
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
                num_workers=0, shuffle=False, scale=None, shift=None): # had to change num_workers to 0
    """Build and return a data loader."""
    dataset = AutoVCDataset(files, spk_embs_root, len_crop, scale=scale, shift=shift)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=shuffle, pin_memory=True) # set pin memory to True if training.
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
        


