import random
import numpy as np
import torch
from torch import t
from torchaudio.transforms import Spectrogram, MelSpectrogram

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# cosine scheduler
class CosineScheduler:
    def __init__(self, epochs, lr, warmup_length=5):
        """
        Arguments
        ---------
        epochs : int
            学習のエポック数．
        lr : float
            学習率．
        warmup_length : int
            warmupを適用するエポック数．
        """
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup_length

    def __call__(self, epoch):
        """
        Arguments
        ---------
        epoch : int
            現在のエポック数．
        """
        progress = (epoch - self.warmup) / (self.epochs - self.warmup)
        progress = np.clip(progress, 0.0, 1.0)
        lr = self.lr * 0.5 * (1. + np.cos(np.pi * progress))

        if self.warmup:
            lr = lr * min(1., (epoch+1) / self.warmup)

        return lr


# 学習率変更関数
def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_meanspect(X, y):
    melspectgram = MelSpectrogram(sample_rate=1200)

    uniq_list=[]
    for i, raw in enumerate(y):
        if raw not in uniq_list:
            uniq_list.append(raw)
            tmp=melspectgram(t(X[i,:,:]))
            if i==0:
                X_spct0=tmp[:,:,0].unsqueeze(2)
                X_spct1=tmp[:,:,1].unsqueeze(2)
            else:
                X_spct0=torch.cat((X_spct0, tmp[:,:,0].unsqueeze(2)),2)
                X_spct1=torch.cat((X_spct1, tmp[:,:,1].unsqueeze(2)),2)

    Spct_mean0=torch.mean(X_spct0, 2)
    Spct_mean1=torch.mean(X_spct1, 2)
    Spct_mean=torch.cat((Spct_mean0.unsqueeze(2), Spct_mean1.unsqueeze(2)), 2)
    return Spct_mean
