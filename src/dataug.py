import torch
from  torchvision.transforms import v2
from torchvision import transforms
from torchaudio.transforms import Spectrogram, MelSpectrogram
import torch.nn as nn
from torch import t
from torch.nn import functional as F
from scipy.signal import butter, lfilter
import numpy as np
from sklearn.preprocessing import RobustScaler
import torchaudio.transforms as T

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
        self.melspectgram = MelSpectrogram(sample_rate=200)
        self.melspectgram40hz = MelSpectrogram(sample_rate=200, f_min=0.1, f_max=40)
        self.melspectgram99hz = MelSpectrogram(sample_rate=200, f_min=0.1, f_max=99)
        self.spectgram = Spectrogram()
        self.aug_sel = aug_sel

    def __call__(self, X):  # 271x281
        if self.aug_sel=='_normal':
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram':
            X = self.melspectgram(t(X))  # 281x128x2
            X0 = t(X[:,:,0])  # 281x128
            X1 = t(X[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram_log':
            X = self.melspectgram(t(X))  # 281x128x2
            X0 = t(X.log2()[:,:,0])  # 281x128
            X1 = t(X.log2()[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_bandpass_l':
            X0=t(torch.tensor(butter_bandpass_filter(t(X), 1, 50, 200)))  # 271x281
            X1=t(torch.tensor(butter_bandpass_filter(t(X), 50, 99, 200)))
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_bandpass_40':
            X0=t(torch.tensor(butter_bandpass_filter(t(X), 0.1, 20, 200)))  # 271x281
            X1=t(torch.tensor(butter_bandpass_filter(t(X), 20, 40, 200)))
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_filter99+spect':
            X=torch.tensor(butter_bandpass_filter(t(X), 1, 99, 200)).float()
            X = self.melspectgram(X)  # 281x128x2
            X0 = t(X[:,:,0])  # 281x128
            X1 = t(X[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram99':
            X = self.melspectgram99hz(t(X))  # 281x128x2
            X0 = t(X[:,:,0])  # 281x128
            X1 = t(X[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram40':
            X = self.melspectgram40hz(t(X))  # 281x128x2
            X0 = t(X[:,:,0])  # 281x128
            X1 = t(X[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_basel_scale_clip':
            shape=X.shape[1]
            #resampler=T.Resample(200, 200)
            #X=resampler(X)
            X_ave=torch.mean(X[:,:50], axis=1, keepdims=True)
            X_ave=torch.tile(X_ave, (1,shape))
            X=X-X_ave
            transformer = RobustScaler().fit(X)
            X=transformer.transform(X)
            X=np.clip(X, -20, 20)
            X=torch.tensor(X)  # 271x281
            X = F.adaptive_avg_pool1d(X, 512)
            X = F.adaptive_avg_pool1d(t(X), 512)
            X = t(X)
            X = F.normalize(X)
            return X.float()
