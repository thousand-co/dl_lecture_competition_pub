#from  torchvision import transforms
from torchaudio.transforms import MelSpectrogram
import torch.nn as nn
from torch import t

class DatAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectgram = MelSpectrogram(sample_rate=1200)

    def __call__(self, X):  # 271x281
        data = self.spectgram(t(X))  # 281x128x2
        return t(data.log2()[:,:,0])  # 281x128