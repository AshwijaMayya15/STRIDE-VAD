import torch
import torch.nn.functional as F
import option_enhanced as option
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

args = option.parse_args()
torch.autograd.set_detect_anomaly(True)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TripletLoss(nn.Module):
    """Label-driven Triplet Loss with adaptive margin"""
    def __init__(self):
        super().__init__()
    
    def forward(self, feats, labels, margin=100.0):
        """
        feats: (B, D)
        labels: (B,) with values in [0,1]
        """
        labels = labels.view(-1)

        # Normal = 0, Anomaly = 1
        n_feats = feats[labels < 0.5]
        a_feats = feats[labels >= 0.5]

        # Safety check (can happen with mixup or small batches)
        if n_feats.size(0) < 2 or a_feats.size(0) < 1:
            return torch.tensor(0.0, device=feats.device)

        # Intra-normal distance (compact normals)
        n_d = torch.cdist(n_feats, n_feats, p=2)
        n_d_max = n_d.max(dim=1)[0]

        # Normal-to-anomaly distance (push anomalies away)
        a_d = torch.cdist(n_feats, a_feats, p=2)
        a_d_min = a_d.min(dim=1)[0]

        triplet = torch.clamp(margin - a_d_min, min=0.0)
        return (n_d_max + triplet).mean()



class EnhancedLoss(nn.Module):
    """Enhanced loss combining Focal Loss and Triplet Loss"""
    def __init__(self, alpha=0.005, use_focal=True, focal_gamma=2.0):
        super().__init__()
        self.use_focal = use_focal
        if use_focal:
            self.criterion = FocalLoss(alpha=0.25, gamma=focal_gamma)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()
        self.alpha = alpha
    
    def forward(self, scores, feats, targets, margin=100.0):
        loss_ce = self.criterion(scores, targets)
        loss_triplet = self.triplet(feats, targets, margin=margin)
        return loss_ce, self.alpha * loss_triplet


def mixup_data(x1, x2, alpha=0.2):
    """Mixup augmentation for better regularization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    mixed = lam * x1 + (1 - lam) * x2
    return mixed, lam


def smooth_labels(labels, epsilon=0.1):
    """Label smoothing for better generalization"""
    return labels * (1 - epsilon) + epsilon * 0.5


def adaptive_margin(epoch, base=100, max_margin=200, warmup=5):
    """Adaptive margin that increases during training"""
    if epoch < warmup:
        return base
    return min(base + (epoch - warmup) * 5, max_margin)


def train(loader, model, optimizer, scheduler, device, epoch):
    """
    Enhanced training function with:
    - Focal Loss option
    - Mixup augmentation
    - Adaptive margin for triplet loss
    - Gradient clipping
    - Label smoothing
    """
    model.train()
    pred, label = [], []
    
    # Initialize loss with options from args
    use_focal = getattr(args, 'use_focal', True)
    focal_gamma = getattr(args, 'focal_gamma', 2.0)
    loss_fn = EnhancedLoss(alpha=args.alpha, use_focal=use_focal, focal_gamma=focal_gamma)
    
    # Mixup and label smoothing parameters
    mixup_alpha = getattr(args, 'mixup_alpha', 0.2)
    label_smoothing = getattr(args, 'label_smoothing', 0.1)
    use_mixup = getattr(args, 'use_mixup', True) and np.random.random() > 0.3
    
    # Adaptive margin
    margin = adaptive_margin(epoch, base=100, max_margin=150)
    
    running_loss, running_ce, running_triplet = 0.0, 0.0, 0.0
    
    for step, (ninput, nlabel, ainput, alabel) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"):
        # Move inputs to device
        ninput = ninput.to(device)
        ainput = ainput.to(device)
        
        # Apply mixup augmentation randomly
        if use_mixup and np.random.random() > 0.5:
            mixed_normal, lam_n = mixup_data(ninput, ainput, alpha=mixup_alpha)
            mixed_anomaly, lam_a = mixup_data(ainput, ninput, alpha=mixup_alpha)
            input_batch = torch.cat((mixed_normal, mixed_anomaly), 0)
            
            # Adjust labels for mixup
            nlabel_mixed = lam_n * nlabel + (1 - lam_n) * alabel
            alabel_mixed = lam_a * alabel + (1 - lam_a) * nlabel
            labels = torch.cat((nlabel_mixed, alabel_mixed), 0).to(device)
        else:
            # Standard concatenation
            input_batch = torch.cat((ninput, ainput), 0)
            labels = torch.cat((nlabel, alabel), 0).to(device)
        
        # Apply label smoothing
        if label_smoothing > 0:
            labels = smooth_labels(labels, epsilon=label_smoothing)
        
        # Forward pass
        scores, feats = model(input_batch)
        
        # Compute losses with adaptive margin
        loss_ce, loss_triplet = loss_fn(scores.squeeze(), feats, labels, margin=margin)
        loss = loss_ce + loss_triplet
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        scheduler.step_update(epoch * len(loader) + step)
        
        # Track metrics
        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_triplet += loss_triplet.item()
        
        # Apply sigmoid to logits for predictions
        pred += torch.sigmoid(scores).detach().cpu().tolist()
        # Use original labels (before smoothing) for metrics - round to binary
        original_labels = torch.cat((nlabel, alabel), 0)
        label += original_labels.cpu().tolist()
    
    # Epoch summary
    avg_loss = running_loss / len(loader)
    avg_ce = running_ce / len(loader)
    avg_triplet = running_triplet / len(loader)
    
    # Convert to numpy arrays and ensure proper format
    label = np.array(label)
    pred = np.array(pred)
    
    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label, pred)
    pr_auc = auc(recall, precision)
    
    print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | Triplet: {avg_triplet:.4f} | Margin: {margin:.1f}")
    print(f"[Epoch {epoch}] train_pr_auc: {pr_auc:.4f} | train_roc_auc: {roc_auc:.4f}")
    
    return avg_loss