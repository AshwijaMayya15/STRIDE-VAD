from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
import numpy as np
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import summary
from tqdm import tqdm
import option_enhanced as option
args = option.parse_args()
from model_enhanced import EnhancedModel
from dataset import SpatioTemporalDataset
from train_enhanced import train
from test_enhanced import test
import datetime
import os
import random
import sys
import json


def save_config(save_path):
    path = save_path + '/'
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f = open(os.path.join(path, f"config_{timestamp}.txt"), 'w')
    for key in vars(args).keys():
        f.write('{}: {}'.format(key, vars(args)[key]))
        f.write('\n')
    f.close()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class EarlyStopping:
    """Early stopping to stop training when validation performance stops improving"""
    def __init__(self, patience=20, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # min
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False

    
if __name__ == '__main__':
    args = option.parse_args()
    
    # Set random seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda')

    # Setup save path
    savepath = './ckpt1/{}_{}_{}_seed{}'.format(args.lr, args.batch_size, args.comment, seed)
    save_config(savepath)

    # Determine input channels
    if args.use_flow:
        input_channels = 198
        print("=" * 60)
        print("Training Enhanced Model with Early Fusion: X3D + Flow")
        print(f"Input channels: {input_channels} (192 X3D + 6 Flow)")
    else:
        input_channels = 192
        print("=" * 60)
        print("Training Enhanced Model with X3D only")
        print(f"Input channels: {input_channels}")
    
    print(f"Seed: {seed}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epoch}")
    print(f"Early stop patience: {args.early_stop_patience}")
    print(f"Using Focal Loss: {args.use_focal}")
    print(f"Using Mixup: {args.use_mixup}")
    print(f"Using SE blocks: {args.use_se}")
    print(f"Using TPP: {args.use_tpp}")
    print(f"Using LSTM: {args.use_lstm}")
    if args.use_lstm:
        print(f"  LSTM Hidden: {args.lstm_hidden}")
        print(f"  LSTM Layers: {args.lstm_layers}")
    print(f"Using ConvLSTM: {args.use_convlstm}")  # ADD THIS LINE
    print("=" * 60)

    # Create dataloaders
    train_loader = DataLoader(SpatioTemporalDataset(args, test_mode=False),
                               batch_size=args.batch_size // 2, shuffle=False)
    test_loader = DataLoader(SpatioTemporalDataset(args, test_mode=True),
                             batch_size=args.batch_size)

    # Initialize enhanced model
    if args.model_arch == 'base':
        model = EnhancedModel(
            dropout=args.dropout_rate, 
            attn_dropout=args.attn_dropout_rate,
            input_channels=input_channels,
            dims=(192, 192, 128, 128),
            depths=(4, 3, 3, 2),
            block_types=('c', 'c', 'a', 'a'),
            use_se=args.use_se,
            use_tpp=args.use_tpp,
            use_lstm=args.use_lstm,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            use_convlstm=args.use_convlstm, # ADD THIS LINE
            use_decoupled_lstm=args.use_decoupled_lstm
        )
    elif args.model_arch == 'fast' or args.model_arch == 'tiny':
        model = EnhancedModel(
            dropout=args.dropout_rate, 
            attn_dropout=args.attn_dropout_rate, 
            ff_mult=1, 
            dims=(32, 32), 
            depths=(1, 1),
            input_channels=input_channels,
            use_se=False,
            use_tpp=False,
            use_lstm=False,
            use_convlstm=False  # ADD THIS LINE
        )
    else:
        print("Model architecture not recognized")
        sys.exit()
    
    model.apply(init_weights)

    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(model_ckpt)
        print("Pretrained checkpoint loaded from:", args.pretrained_ckpt)

    model = model.to(device)

    # Print model summary
    print("\nModel Summary:")
    summary(model, (1, input_channels, 16, 10, 10), depth=3)

    if not os.path.exists('./ckpt1'):
        os.makedirs('./ckpt1')

    # Optimizer with configurable weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.max_epoch * num_steps,
        cycle_mul=1.,
        lr_min=args.lr * 0.1,
        warmup_lr_init=args.lr * 0.01,
        warmup_t=args.warmup * num_steps,
        cycle_limit=20,
        t_in_epochs=False,
        warmup_prefix=True,
        cycle_decay=0.95,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=0.0001,
        mode='max'
    )

    # Training tracking
    test_info = {"epoch": [], "test_AUC": [], "test_PR": []}
    best_auc = 0.0
    best_pr_auc = 0.0
    best_epoch_auc = 0
    best_epoch_pr = 0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    for step in tqdm(
            range(0, args.max_epoch),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        # Training
        cost = train(train_loader, model, optimizer, scheduler, device, step)

        # Testing
        auc, pr_auc = test(test_loader, model, args, device)

        # Track results
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)
        
        # Save checkpoint
        if args.use_flow:
            checkpoint_name = savepath + '/' + args.model_name + f'_epoch{step}_fusion.pkl'
        else:
            checkpoint_name = savepath + '/' + args.model_name + f'_epoch{step}_x3d.pkl'
        
        torch.save(model.state_dict(), checkpoint_name)
        save_best_record(test_info, os.path.join(savepath + "/", f'epoch{step}_results.txt'))

        # Track best performance
        if auc > best_auc:
            best_auc = auc
            best_epoch_auc = step
            best_checkpoint = savepath + '/' + args.model_name + '_best_auc.pkl'
            torch.save(model.state_dict(), best_checkpoint)
            print(f"\n🏆 New best AUC: {best_auc:.4f} at epoch {step} - Model saved!")
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch_pr = step
            best_pr_checkpoint = savepath + '/' + args.model_name + '_best_pr.pkl'
            torch.save(model.state_dict(), best_pr_checkpoint)
            print(f"🏆 New best PR-AUC: {best_pr_auc:.4f} at epoch {step} - Model saved!")
        
        # Early stopping check
        if early_stopping(auc):
            print(f"\n⚠️  Early stopping triggered at epoch {step}")
            print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch_auc}")
            print(f"Best PR-AUC: {best_pr_auc:.4f} at epoch {best_epoch_pr}")
            break
        
        # Print current status
        print(f"Current: AUC={auc:.4f}, PR-AUC={pr_auc:.4f} | Best: AUC={best_auc:.4f}, PR-AUC={best_pr_auc:.4f}")

    # Save final model
    if args.use_flow:
        final_model_path = savepath + '/' + args.model_name + '_final_fusion.pkl'
    else:
        final_model_path = savepath + '/' + args.model_name + '_final.pkl'
    
    torch.save(model.state_dict(), final_model_path)
    
    # Save final results summary
    results_summary = {
        'seed': seed,
        'best_auc': float(best_auc),
        'best_auc_epoch': int(best_epoch_auc),
        'best_pr_auc': float(best_pr_auc),
        'best_pr_auc_epoch': int(best_epoch_pr),
        'final_auc': float(test_info['test_AUC'][-1]),
        'final_pr_auc': float(test_info['test_PR'][-1]),
        'total_epochs': len(test_info['epoch']),
        'config': vars(args)
    }
    
    with open(savepath + '/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"Best AUC: {best_auc:.4f} (Epoch {best_epoch_auc})")
    print(f"Best PR-AUC: {best_pr_auc:.4f} (Epoch {best_epoch_pr})")
    print(f"Final model saved to: {final_model_path}")
    print(f"Results summary saved to: {savepath}/results_summary.json")
    print("=" * 60)