import datetime
import math
import os
import random
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from fastprogress import master_bar, progress_bar
from torch.nn import utils
from torch.utils import tensorboard
import torchvision

from data import AutoVCDataset, get_loader, precompute_sse
from hp import hp
from model_vc import Generator
from plot_utils import plot_spec, spec_to_tensorboard
from audio_utils import Vocoder, save_wavs, format_wavs


def train(args):
    print("[BACKEND] Setting up paths and training.")
    out_path = Path(hp.output_path)
    os.makedirs(out_path, exist_ok=True) # Make output directory

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #also can manually set in hp.py

    # For now, no precomputed spectrograms
    if args.mel_path is None:
        ds_root = Path(hp.data_root) # /subdir if applicable
    else:
        print("[DATA] Using precomputed mels from ", args.mel_path)
        ds_root = Path(args.mel_path)

    # Get speaker folders (only directories with numeric names)
    spk_folders = []
    for path in sorted(list(ds_root.iterdir())):
        if path.is_dir():
            if path.parts[-1].isnumeric():
                spk_folders.append(path)
    print(f"[DATA] Found a total of {len(spk_folders)} speakers")

    # Getting embeddings:
    print("[SPEAKER EMBEDDING] Precomputing/identifying speaker embeddings")
    precompute_sse(spk_folders, device)
    spk_embs_root = Path(hp.speaker_embedding_dir)

    # Gather training / testing paths.
    random.seed(hp.seed)
    train_spk_folders = sorted(random.sample(spk_folders, k=hp.n_train_speakers))
    test_spk_folders = sorted(list(set(spk_folders) - set(train_spk_folders)))
    train_files = []
    for pth in train_spk_folders: train_files.extend(list(pth.iterdir()))
    test_files = []
    for pth in test_spk_folders: test_files.extend(list(pth.iterdir()))
    print(f"[DATA] Split into {len(train_spk_folders)} train speakers ({len(train_files)} files)")
    print(f"[DATA] and {len(test_spk_folders)} test speakers ({len(test_files)} files)")

    print("[DATA] Constructing final dataloaders")
    train_dl = get_loader(train_files, spk_embs_root, hp.len_crop, hp.bs, 
                        shuffle=True, shift=hp.mel_shift, scale=hp.mel_scale)
    test_dl = get_loader(test_files, spk_embs_root, hp.len_crop, hp.bs, 
                        shuffle=False, shift=hp.mel_shift, scale=hp.mel_scale)

    print(f'  Num Train samples: {len(train_files)}')
    print(f'  Num Test samples: {len(test_files)}')

    print("[LOGGING] Setting up logger")
    writer = tensorboard.writer.SummaryWriter(out_path)
    keys = ['G/loss_spec','G/loss_spec_psnt','G/loss_cd']

    print("[MODEL] Setting up model")
    G = Generator(hp.dim_neck, hp.sse_dim, hp.dim_pre, hp.f0_dim, hp.amp_dim, hp.freq).to(device)
    opt = torch.optim.Adam(G.parameters(), hp.lr)
    if args.fp16: 
        print("[TRAIN] Using fp16 training.")
        scaler = GradScaler()

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        epoch = ckpt['epoch']
        ite = ckpt['iter']
        G.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['opt_state_dict'])
        print(f"[CHECKPOINT] Loaded checkpoint starting from epoch {epoch} (iter {ite})",
                f" with last known loss {ckpt['loss']:6.5f}")

    if args.vocode:
      print("[MODEL] Setting up neural vocoder.")
      vocode = Vocoder()

    print("[TRAIN] Beginning training")
    start_time = time.time()
    running_loss = 0.0
    n_epochs = math.ceil(hp.n_iters / len(train_dl))
    # if args.checkpoint is not None:
    #   iter = ite
    # else:
    iter = 0
    # mb = master_bar(range(n_epochs))
    for epoch in range(n_epochs):
        print(f'--------------- Epoch {epoch} --------------------')
        G.train()
        # pb = progress_bar(enumerate(train_dl), total=len(train_dl), parent=mb)
        for i, (x_src, x_tgt, s_src, f0_src, amp_src) in enumerate(train_dl):
            x_src = x_src.to(device)
            x_tgt = x_tgt.to(device)
            s_src = s_src.to(device)
            f0_src = f0_src[0].to(device) #0 is 1-hot vec, 1 is index of 1.0 location in vec
            amp_src = amp_src[0].to(device)

            # print(x_src.shape, x_tgt.shape, s_src.shape, f0_src.shape, amp_src.shape)
            # yields: torch.Size([BS, 128, 80]) torch.Size([BS, 128, 80]) torch.Size([BS, 256]) torch.Size([BS, 128, 257]) torch.Size([BS, 128, 256])
            opt.zero_grad()

            # fp16 enable
            if args.fp16:
              pass
                # with autocast():
                #     # Conversion mapping loss
                #     x_pred, x_pred_psnt, code_src = G(x_src, f0_src, amp_src, s_src, s_src)
                #     g_loss_spec = F.mse_loss(x_tgt, x_pred.squeeze(1))   
                #     g_loss_spec_psnt = F.mse_loss(x_tgt, x_pred_psnt.squeeze(1))   
                    
                #     # Code semantic loss.
                #     code_pred = G(x_pred_psnt.squeeze(1), None, None, s_src, None)
                #     g_loss_cd = F.l1_loss(code_src, code_pred)

                #     g_loss = g_loss_spec + hp.mu*g_loss_spec_psnt + hp.lamb*g_loss_cd
                # scaler.scale(g_loss).backward()
                # scaler.step(opt)
                # scaler.update()
            else:
                # Conversion mapping loss
                x_pred, x_pred_psnt, code_src = G(x_src, f0_src, amp_src, s_src, s_src)
                g_loss_spec = F.mse_loss(x_tgt, x_pred.squeeze(1))   
                g_loss_spec_psnt = F.mse_loss(x_tgt, x_pred_psnt.squeeze(1))   
                
                # Code semantic loss.
                code_pred = G(x_pred_psnt.squeeze(1), None, None, s_src, None)
                g_loss_cd = F.l1_loss(code_src, code_pred)

                g_loss = g_loss_spec + hp.mu*g_loss_spec_psnt + hp.lamb*g_loss_cd
                g_loss.backward()
                opt.step()

            loss = {}
            loss['G/loss_spec'] = g_loss_spec.item()
            loss['G/loss_spec_psnt'] = g_loss_spec_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
            
            # lerp smooth running loss
            running_loss = running_loss + 0.1*(float(g_loss) - running_loss)
            # mb.child.comment = f"loss = {float(running_loss):6.5f}"
            print(f"loss = {float(running_loss):6.5f}")

            if iter % hp.tb_log_interval == 0:
                for tag in keys: writer.add_scalar(tag, loss[tag], iter)
                writer.add_scalar('G/loss', g_loss.item(), iter)

            if iter % hp.print_log_interval == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, iter+1, hp.n_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                # mb.write(log)
                print(log)
            
            if iter % hp.media_log_interval == 0:
                # Plot, Save and Add Spectograms to tensorboard
                image = spec_to_tensorboard( (x_src[0], x_tgt[0], x_pred[0], x_pred_psnt[0]), f'E{epoch}', out_path)
                writer.add_image(f'G/MelSpec_E{epoch}_I{iter}', image, epoch)    
                if args.vocode:
                    wavs_true = vocode(x_src)
                    wavs_pred = vocode(x_pred_psnt.squeeze(1))
                    wavs = format_wavs(wavs_true, wavs_pred)
                    for j,wav in enumerate(wavs):
                        wav = vocode.resample(wav)
                        writer.add_audio(f'valid/wavs_E{epoch}_{i}_true', wav[0,:].unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
                        writer.add_audio(f'valid/wavs_E{epoch}_{i}_pred', wav[1,:].unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
                        save_wavs(wav, os.path.join(out_path,f'wavs_E{epoch}_{j}.wav'))
                    
            iter += 1
            if iter >= hp.n_iters:
                print("[TRAIN] Training completed.")
                break

        
        # mb.write(f"[TRAIN] epoch {epoch} completed. Beginning eval.")
        print(f"[TRAIN] epoch {epoch} completed. Beginning eval.")
        G.eval()
        # pb = progress_bar(enumerate(test_dl), total=len(test_dl), parent=mb)
        valid_losses = {tag: [] for tag in keys}
        valid_losses['G/loss'] = []
        for i, (x_src, x_tgt, s_src, f0_src, amp_src) in enumerate(test_dl):
            x_src = x_src.to(device)
            x_tgt = x_tgt.to(device)
            s_src = s_src.to(device)
            f0_src = f0_src[0].to(device) #0 is 1-hot vec, 1 is index of 1.0 location in vec
            amp_src = amp_src[0].to(device)

            with torch.no_grad():
                # Identity mapping loss
                x_pred, x_pred_psnt, code_src = G(x_src, f0_src, amp_src, s_src, s_src)
                g_loss_spec = F.mse_loss(x_src, x_pred.squeeze(1))   
                g_loss_spec_psnt = F.mse_loss(x_src, x_pred_psnt.squeeze(1))   
                
                # Code semantic loss.
                code_pred = G(x_pred_psnt.squeeze(1), None, None, s_src, None)
                g_loss_cd = F.l1_loss(code_src, code_pred)

                g_loss = g_loss_spec + hp.mu*g_loss_spec_psnt + hp.lamb*g_loss_cd

            valid_losses['G/loss_spec'].append(g_loss_spec.item())
            valid_losses['G/loss_spec_psnt'].append(g_loss_spec_psnt.item())
            valid_losses['G/loss_cd'].append(g_loss_cd.item())
            valid_losses['G/loss'].append(g_loss.item())
            # mb.child.comment = f"loss = {float(g_loss):6.5f}"
        
            image = spec_to_tensorboard( (x_src[0], x_tgt[0], x_pred[0], x_pred_psnt[0]), f'E{epoch}_v{i}', out_path)
            writer.add_image(f'valid/MelSpec_E{epoch}_{i}', image, epoch)
            if args.vocode:
                    wavs_true = vocode(x_src)
                    wavs_pred = vocode(x_pred_psnt.squeeze(1))
                    wavs = format_wavs(wavs_true, wavs_pred)
                    for j,wav in enumerate(wavs):
                        wav = vocode.resample(wav)
                        writer.add_audio(f'valid/wavs_E{epoch}_{i}_true', wav[0,:].unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
                        writer.add_audio(f'valid/wavs_E{epoch}_{i}_pred', wav[1,:].unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
                        save_wavs(wav, os.path.join(out_path,f'wavs_E{epoch}_{j}.wav'))

        valid_losses = {k: np.mean(valid_losses[k]) for k in valid_losses.keys()}
        for tag in valid_losses.keys(): writer.add_scalar('valid/' + tag, valid_losses[tag], iter)
        pst = [f"{k}: {valid_losses[k]:5.4f}" for k in valid_losses.keys()]
        # mb.write(f"[TRAIN] epoch {epoch} eval metrics: " + '\t'.join(pst))
        print(f"[TRAIN] epoch {epoch} eval metrics: " + '\t'.join(pst))

        print("[EPOCH COMPLETE] Saving model")
        torch.save({
            'epoch': epoch,
            'iter': iter,
            'model_state_dict': G.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'loss': valid_losses['G/loss']
        }, out_path/'checkpoint_last.pth')
    
    print("[CLEANUP] Saving model")
    torch.save({
        'epoch': epoch,
        'iter': iter,
        'model_state_dict': G.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'loss': valid_losses['G/loss']
    }, out_path/'checkpoint_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='autovc trainer')
    parser.add_argument('--checkpoint', action='store', required=False, default=None,
                        help='checkpoint to restore from')
    parser.add_argument('--fp16', required=False, default=False, action='store', type=bool,
                        help='use fp16 in training')
    parser.add_argument('--mel_path', required=False, default=None, action='store',
                        help='path to precomputed spectrograms. Compute them on the fly if not.')
    parser.add_argument('--vocode', required=False, default=False, action='store',
                        help='boolean flag to indicate if hifi-gan vocoder is envoked during train/test')


    args = parser.parse_args()
    train(args)
