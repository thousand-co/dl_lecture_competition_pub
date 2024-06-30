import torch
from  torchvision.transforms import v2
from torchvision import transforms
from torchaudio.transforms import MelSpectrogram
import torch.nn as nn
from torch import t
from torch.nn import functional as F
from scipy.signal import butter, lfilter
import numpy as np

def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    return mu_x  # quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class DatAugmentation(torch.utils.data.Dataset):
    def __init__(self, aug_sel='normal'):
        super().__init__()
        self.spectgram = MelSpectrogram(sample_rate=1200)
        #self.normalize = v2.Normalize(mean=[0.5], std=[0.5])
        #self.resize = v2.Resize(size=512)
        #self.transf = transforms.Normalize(mean=[0.5], std=[0.5])
        #self.totensor = transforms.ToTensor()
        self.aug_sel = aug_sel

    def __call__(self, X):  # 271x281
        X = X + 100
        if self.aug_sel=='_normal':
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram':
            X = self.spectgram(t(X))  # 281x128x2
            X0 = t(X[:,:,0])  # 281x128
            X1 = t(X[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram_log':
            X = self.spectgram(t(X))  # 281x128x2
            X0 = t(X.log2()[:,:,0])  # 281x128
            X1 = t(X.log2()[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_bandpass_l':
            X0=t(torch.tensor(butter_bandpass_filter(t(X), 1, 50, 1200)))  # 271x281
            X1=t(torch.tensor(butter_bandpass_filter(t(X), 50, 100, 1200)))
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X) + torch.ones(512,512)*50
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_bandpass_h':
            X0=t(torch.tensor(butter_bandpass_filter(t(X), 100, 200, 1200)))  # 271x281
            X1=t(torch.tensor(butter_bandpass_filter(t(X), 200, 400, 1200)))
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X) + torch.ones(512,512)*50
            X = F.normalize(X)
            return X.float()
