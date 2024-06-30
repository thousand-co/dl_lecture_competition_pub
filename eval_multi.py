import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset_aug1
from src.models import BasicConvClassifier
from src.utils import set_seed
from src.dataug import DatAugmentation
from src.models_res import ResNet, Bottleneck


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
#def run(args: DictConfig):
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    model_path = savedir + '/'

    #model_name='model_best'
    model_name='model_last'

    id_list=['_0', '_1', '_2', '_3']
    #aug_list=['_normal', '_spectgram', '_spectgram_log', '_bandpass_l', '_bandpass_h']
    aug_list=['_normal']
    for aug in aug_list:
        for id in id_list:
            ### Data Augmentation ###
            transform_test=DatAugmentation(aug_sel=aug)

            # ------------------
            #    Dataloader
            # ------------------    
            test_set = ThingsMEGDataset_aug1("test", args.data_dir, id, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(
                test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
            )

            # ------------------
            #       Model
            # ------------------
            #model = BasicConvClassifier(
            #    test_set.num_classes, test_set.seq_len, test_set.num_channels
            #).to(args.device)
            model = ResNet(Bottleneck, [3, 4, 32, 3], num_classes=test_set.num_classes).to(args.device)
            model.load_state_dict(torch.load(model_path+model_name+aug+id+'.pt', map_location=args.device))

            # ------------------
            #  Start evaluation
            # ------------------ 
            preds = [] 
            model.eval()
            #for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
            for X in tqdm(test_loader, desc="Validation"):        
                preds.append(model(X.to(args.device)).detach().cpu())
                
            preds = torch.cat(preds, dim=0).numpy()
            np.save(os.path.join(savedir, "submission"+aug+id), preds)
            cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()