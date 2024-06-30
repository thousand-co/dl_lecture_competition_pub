import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data_spl", id: str = "_0") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X"+id+".pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y"+id+".pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i]
        else:
            return self.X[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

class ThingsMEGDataset_aug1(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data_spl", id: str = "_0", transform=None) -> None:
        super().__init__()
        self.transform = transform
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        # 4108x271x281
        self.X = torch.load(os.path.join(data_dir, f"{split}_X"+id+".pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y"+id+".pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if self.transform is not None:
            out_X=self.transform(self.X[i])
        else:
            out_X=self.X[i]

        if hasattr(self, "y"):
            return out_X, self.y[i]
        else:
            return out_X
        
    @property
    def num_channels(self) -> int:  #271-->128
        return 128
        #return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:  #281-->281
        return 281
        #return self.X.shape[2]