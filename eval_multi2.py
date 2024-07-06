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
from src.datprep_i import DatPreprocess
from src.models_res import ResNet, Bottleneck
import yaml

with open('./configs/config.yaml', 'r') as yml:
    args = yaml.safe_load(yml)

@torch.no_grad()
#@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(model_path=sys.argv[1].split('=')[1]):
    set_seed(args['seed'])
    savedir = os.path.dirname(model_path)
    model_path = savedir + '/'

    model_name='model_best'
    #model_name='model_last'

    id_list=['_0', '_1', '_2', '_3']
    #aug_list=['_normal', '_spectgram', '_spectgram_log', '_bandpass_l', '_bandpass_h']
    aug_list=['_normal']
    for aug in aug_list:
        for id in id_list:
            ### Data Pre-process ###
            transform_test=DatPreprocess(aug_sel=aug)

            # ------------------
            #    Dataloader
            # ------------------    
            loader_args = {"batch_size": args['batch_size'], "num_workers": args['num_workers']}

            val_set = ThingsMEGDataset_aug1("val", args['data_dir'], id, transform=transform_test)
            test_set = ThingsMEGDataset_aug1("test", args['data_dir'], id, transform=transform_test)

            val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=args['batch_size'], num_workers=args['num_workers'])

            # ------------------
            #       Model
            # ------------------
            model0 = BasicConvClassifier(test_set.num_classes, test_set.seq_len, test_set.num_channels).to(args['device'])
            model1 = BasicConvClassifier(test_set.num_classes, test_set.seq_len, test_set.num_channels).to(args['device'])
            model2 = BasicConvClassifier(test_set.num_classes, test_set.seq_len, test_set.num_channels).to(args['device'])
            model3 = BasicConvClassifier(test_set.num_classes, test_set.seq_len, test_set.num_channels).to(args['device'])
            #model = ResNet(Bottleneck, [3, 4, 32, 3], num_classes=test_set.num_classes).to(args.device)
            model0.load_state_dict(torch.load(model_path+model_name+aug+'_0.pt', map_location=args['device']))
            model1.load_state_dict(torch.load(model_path+model_name+aug+'_1.pt', map_location=args['device']))
            model2.load_state_dict(torch.load(model_path+model_name+aug+'_2.pt', map_location=args['device']))
            model3.load_state_dict(torch.load(model_path+model_name+aug+'_3.pt', map_location=args['device']))

            # --------------------
            #   Start validating
            # --------------------  
            max_val_acc = 0
            accuracy = Accuracy(
                task="multiclass", num_classes=val_set.num_classes, top_k=10
            ).to(args['device'])
            
            for epoch in range(1):
                print(f"Epoch {epoch+1}/{args['epochs']}")
                for model in [model0, model1, model2, model3]:
                    val_loss, val_acc = [], []
                    
                    model.eval()
                    for X, y in tqdm(val_loader, desc="Validation"):
                        X, y = X.to(args['device']), y.to(args['device'])
                        
                        with torch.no_grad():
                            y_pred = model(X)
                        
                        val_loss.append(F.cross_entropy(y_pred, y).item())
                        val_acc.append(accuracy(y_pred, y).item())

                    print(f"Epoch {epoch+1}/{args['epochs']} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
                    if args['use_wandb']:
                        wandb.log({"val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
                    

            # ------------------
            #  Start evaluation
            # ------------------ 
            #preds = [] 
            #model.eval()
            ###for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
            #for X in tqdm(test_loader, desc="Validation"):        
            #    preds.append(model(X.to(args['device'])).detach().cpu())
                
            #preds = torch.cat(preds, dim=0).numpy()
            #np.save(os.path.join(savedir, "submission"+aug+id), preds)
            #cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()