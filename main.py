import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import tensorflow as tf

from src.datprep_i import DatPreprocess
#from src.datasets import ThingsMEGDataset
from src.datasets import ThingsMEGDataset_aug1
#from src.models import BasicConvClassifier
from src.models2 import BasicConvClassifier, MEGnetClassifier  # with glu
from src.models2D import BasicConvClassifier2D  # with glu
#from src.models_res import Bottleneck, ResNet
#from src.models_tran import Classifier
#from src.models_wres import DoBottleneck, WideResNet
#from src.models_effnet import EfficientNet_V2
from src.utils import set_seed, set_lr, CosineScheduler, get_meanspect

#from torchvision import transforms, models
from torchvision.models import efficientnet_v2_l
#import torchaudio
#import mne
#from scipy.signal import stft
#from sklearn.decomposition import FastICA, PCA


# hydra:設定管理ライブラリ
# confディレクトリconfig.yaml設定ファイルを読み込む
# outputsディレクトリへタイムスタンプ毎に実行結果を保存
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # wandb:実行結果の可視化ツール(要ログイン)
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")
    id_list=['_0', '_1', '_2', '_3']
    #id_list=['_0']
    #aug_list=['_without', '_normal', '_baseline', '_spectgram', '_spectgram_log', '_bandpass_40', '_scale_clip', '_cwt']
    aug_list=['_baseline']
    for aug in aug_list:
        for id in id_list:
            # Data Pre-process
            transform_train=DatPreprocess(aug_sel=aug)
            #transform_valid=DatPreprocess(aug_sel=aug)
            #transform_test=DatPreprocess(aug_sel=aug)

            #transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
            #                          transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            #                          transforms.ToTensor(), transforms.Normalize(0,5, 0.5)])

            # ------------------
            #    Dataloader
            # ------------------
            loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

            #train_set = ThingsMEGDataset("train", args.data_dir, id)
            #val_set = ThingsMEGDataset("val", args.data_dir, id)
            #test_set = ThingsMEGDataset("test", args.data_dir, id)

            train_set = ThingsMEGDataset_aug1("train", args.data_dir, id, transform=transform_train)
            val_set = ThingsMEGDataset_aug1("val", args.data_dir, id, transform=transform_train)
            test_set = ThingsMEGDataset_aug1("test", args.data_dir, id, transform=transform_train)

            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args, pin_memory=True)

            # ------------------
            #       Model
            # ------------------
            if aug=='_cwt':
                model = BasicConvClassifier2D(train_set.num_classes, train_set.seq_len, train_set.num_channels).to(args.device)
            else:
                model = BasicConvClassifier(train_set.num_classes, train_set.seq_len, train_set.num_channels).to(args.device)
                #model = MEGnetClassifier(train_set.num_classes, train_set.seq_len, train_set.num_channels).to(args.device)
                #model = ResNet(Bottleneck, [3, 4, 8 , 3], train_set.num_classes, train_set.seq_len, train_set.num_channels).to(args.device)
                #model = WideResNet(DoBottleneck, [3, 4, 8, 3], train_set.num_classes, train_set.seq_len, train_set.num_channels).to(args.device)  # have error
                #model = EfficientNet_V2(128).to(args.device)
                #model = Classifier(train_set.num_classes, train_set.seq_len, train_set.num_channels).to(args.device)

            # ------------------
            #     Optimizer
            # ------------------
            #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            ### Scheduler ###
            scheduler = CosineScheduler(epochs=args.epochs, warmup_length=int(args.epochs*0.05), lr=args.lr)
            # ------------------
            #   Start training
            # ------------------  
            max_val_acc = 0
            accuracy = Accuracy(
                task="multiclass", num_classes=train_set.num_classes, top_k=10
            ).to(args.device)
            
            for epoch in range(args.epochs):
                print(f"Epoch {epoch+1}/{args.epochs}")

                new_lr = scheduler(epoch)
                set_lr(new_lr, optimizer)

                train_loss, train_acc, val_loss, val_acc = [], [], [], []
                
                model.train()
                scaler = torch.cuda.amp.GradScaler()  # added
                for X, y in tqdm(train_loader, desc="Train"):
                    X, y = X.to(args.device), y.to(args.device)

                    with torch.cuda.amp.autocast():
                        y_pred = model(X)
                        loss = F.cross_entropy(y_pred, y)

                    train_loss.append(loss.item())
                    
                    optimizer.zero_grad()

                    #loss.backward()
                    scaler.scale(loss).backward()  # added

                    #optimizer.step()
                    scaler.step(optimizer)  # added

                    acc = accuracy(y_pred, y)
                    train_acc.append(acc.item())
 
                    scaler.update()  # added

                model.eval()
                for X, y in tqdm(val_loader, desc="Validation"):
                    X, y = X.to(args.device), y.to(args.device)
                    
                    with torch.no_grad():
                        y_pred = model(X)
                    
                    val_loss.append(F.cross_entropy(y_pred, y).item())
                    val_acc.append(accuracy(y_pred, y).item())

                print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
                torch.save(model.state_dict(), os.path.join(logdir, "model_last"+aug+id+".pt"))
                if args.use_wandb:
                    wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
                
                if np.mean(val_acc) > max_val_acc:
                    cprint("New best.", "cyan")
                    torch.save(model.state_dict(), os.path.join(logdir, "model_best"+aug+id+".pt"))
                    max_val_acc = np.mean(val_acc)
                    
            # ----------------------------------
            #  Start evaluation with best model
            # ----------------------------------
            model.load_state_dict(torch.load(os.path.join(logdir, "model_best"+aug+id+".pt"), map_location=args.device))

            preds = [] 
            model.eval()
            for X in tqdm(test_loader, desc="Validation"):        
                preds.append(model(X.to(args.device)).detach().cpu())
                
            preds = torch.cat(preds, dim=0).numpy()
            np.save(os.path.join(logdir, "submission"+aug+id), preds)
            cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
