import torch
#from  torchvision.transforms import v2
#from torchvision import transforms
from torchaudio.transforms import Spectrogram, MelSpectrogram
#import torch.nn as nn
from torch import t
from torch.nn import functional as F
from scipy.signal import butter, lfilter
import numpy as np
from sklearn.preprocessing import RobustScaler
#import torchaudio.transforms as T
#from CWT.cwt import ComplexMorletCWT
from src.cwt import CWT

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

class DatPreprocess(torch.utils.data.Dataset):
    def __init__(self, aug_sel='normal'):
        super().__init__()
        self.melspectgram = MelSpectrogram(sample_rate=200)
        #self.melspectgram40hz = MelSpectrogram(sample_rate=200, f_min=0.1, f_max=40)
        #self.melspectgram99hz = MelSpectrogram(sample_rate=200, f_min=0.1, f_max=99)
        self.spectgram = Spectrogram()
        self.aug_sel = aug_sel

    def __call__(self, X):  # 271x281
        if self.aug_sel=='_normal':
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_baseline':
            shape = X.shape[1]  # 281x128x2
            X_ave=torch.mean(X[:,:28], axis=1, keepdims=True)
            X_ave=torch.tile(X_ave, (1,shape))
            X=X-X_ave
            #X=torch.clip(X, -20, 20)
            X=F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram':
            X = self.melspectgram(t(X))  # 281x128x2
            X0 = t(X[:,:,0])  # 281x128
            X1 = t(X[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_spectgram_log':
            X = self.melspectgram(t(X))  # 281x128x2
            X0 = t(X.log2()[:,:,0])  # 281x128
            X1 = t(X.log2()[:,:,1])  # 281x128
            X = torch.cat((X0, X1), 0)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_bandpass_40':
            X0=t(torch.tensor(butter_bandpass_filter(t(X), 0.1, 20, 200)))  # 271x281
            X1=t(torch.tensor(butter_bandpass_filter(t(X), 20, 40, 200)))
            X = torch.cat((X0, X1), 0)
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_scale_clip':
            transformer = RobustScaler().fit(X)
            X=transformer.transform(X)
            X=np.clip(X, -20, 20)
            X=torch.tensor(X)  # 271x281
            X = F.normalize(X)
            return X.float()
        elif self.aug_sel=='_cwt':
            # hop_length=1はメモリ不足で実行不可,5は可
            x_ten = X.unsqueeze(0)
            pycwt = CWT(fmin=1, fmax=40, hop_length=5, dt=1/200)
            out = pycwt(x_ten).squeeze(0)
            return out
        else:
            return X
        
'''
        elif self.aug_sel=='_cwt':
            prepared_signal = X.reshape((1, -1, 1))
            fs = 200
            lower_freq = 1
            upper_freq = 40
            n_scales = 42
            wavelet_width = 1
            cwt = ComplexMorletCWT(wavelet_width, fs, lower_freq, upper_freq, n_scales, border_crop=0, stride=5)
            np_scalogram = cwt(prepared_signal)
            scalogram_real = np_scalogram[0, :, :, 0]
            scalogram_imag = np_scalogram[0, :, :, 1]
            scalogram_magn = np.sqrt(scalogram_real ** 2 + scalogram_imag ** 2)
            scalogram_magn = torch.tensor(scalogram_magn)
            return scalogram_magn
        else:
            return X
'''