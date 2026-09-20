from torch.utils.data import DataLoader
import option_enhanced as option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    auc, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from tqdm import tqdm
from model_enhanced import EnhancedModel
from dataset import SpatioTemporalDataset
from torchinfo import summary
import umap.umap_ as umap
import seaborn as sns
import numpy as np
import sys
import os
import glob
import pandas as pd
import json
import warnings

# Suppress sklearn's UndefinedMetricWarning for single-class categories
warnings.filterwarnings('ignore', message='Only one class is present in y_true')


MODEL_LOCATION = 'saved_models/'
MODEL_NAME = 'STRIDE_BiLSTM_DECOUPLED_LSTM'
MODEL_EXTENSION = '.pkl'
CONF_THRESHOLD = 0.5  # Fixed threshold for confusion matrix

class VideoAnalysisResults:
    """Store and analyze per-video classification results"""
    
    def __init__(self):
        self.video_paths = []
        self.predictions = []
        self.labels = []
        self.features = []
    
    def add_result(self, video_path, prediction, label, feature):
        self.video_paths.append(video_path)
        self.predictions.append(prediction)
        self.labels.append(label)
        self.features.append(feature)
    
    def get_video_name(self, path):
        basename = os.path.basename(path)
        return basename.replace('.npy', '').replace('_x264', '')
    
    def get_category(self, path):
        parts = path.split('/')
        return parts[-2] if len(parts) >= 2 else "Unknown"
    
    def analyze_results(self, threshold=0.5):
        results = {
            'true_negatives': [],
            'false_positives': [],
            'false_negatives': [],
            'true_positives': [],
        }
        
        for path, pred, label in zip(self.video_paths, self.predictions, self.labels):
            video_name = self.get_video_name(path)
            category = self.get_category(path)
            prediction_binary = 1 if pred >= threshold else 0
            label_binary = int(label)
            
            video_info = {
                'video_name': video_name,
                'category': category,
                'true_label': 'Normal' if label_binary == 0 else 'Anomaly',
                'predicted_label': 'Normal' if prediction_binary == 0 else 'Anomaly',
                'anomaly_score': float(pred),
                'confidence': float(abs(pred - 0.5) * 2),
                'path': path
            }
            
            if label_binary == 0 and prediction_binary == 0:
                results['true_negatives'].append(video_info)
            elif label_binary == 0 and prediction_binary == 1:
                results['false_positives'].append(video_info)
            elif label_binary == 1 and prediction_binary == 0:
                results['false_negatives'].append(video_info)
            elif label_binary == 1 and prediction_binary == 1:
                results['true_positives'].append(video_info)
        
        return results
    
    def save_detailed_report(self, results, save_dir='video_analysis'):
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("📊 DETAILED VIDEO CLASSIFICATION ANALYSIS")
        print(f"{'='*80}\n")
        
        tn_count = len(results['true_negatives'])
        fp_count = len(results['false_positives'])
        fn_count = len(results['false_negatives'])
        tp_count = len(results['true_positives'])
        total = tn_count + fp_count + fn_count + tp_count
        
        print(f"Total Videos: {total}")
        print(f"  ✅ Correct:   {tn_count + tp_count} ({(tn_count + tp_count)/total*100:.2f}%)")
        print(f"  ❌ Incorrect: {fp_count + fn_count} ({(fp_count + fn_count)/total*100:.2f}%)")
        print(f"\nBreakdown:")
        print(f"  True Negatives (TN):  {tn_count:4d}  (Normal → Correct)")
        print(f"  True Positives (TP):  {tp_count:4d}  (Anomaly → Correct)")
        print(f"  False Positives (FP): {fp_count:4d}  (Normal → WRONG)")
        print(f"  False Negatives (FN): {fn_count:4d}  (Anomaly → WRONG)")
        
        # Save summary
        with open(os.path.join(save_dir, 'classification_summary.txt'), 'w') as f:
            f.write("="*80 + "\n")
            f.write("VIDEO CLASSIFICATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total: {total}\n")
            f.write(f"Correct: {tn_count + tp_count}\n")
            f.write(f"TN: {tn_count}, TP: {tp_count}, FP: {fp_count}, FN: {fn_count}\n")
        
        # Save each category
        categories = [
            ('true_negatives', 'Correctly Classified Normal', '✅'),
            ('true_positives', 'Correctly Classified Anomaly', '✅'),
            ('false_positives', 'MISCLASSIFIED: Normal as Anomaly', '❌'),
            ('false_negatives', 'MISCLASSIFIED: Anomaly as Normal', '❌')
        ]
        
        for key, name, emoji in categories:
            videos = results[key]
            if len(videos) == 0:
                continue
            
            print(f"\n{emoji} {name}: {len(videos)}")
            print("-" * 80)
            
            sorted_videos = sorted(videos, key=lambda x: x['confidence'], reverse=True)
            
            for i, v in enumerate(sorted_videos[:10], 1):
                print(f"  {i:2d}. {v['video_name']:<30} | Score: {v['anomaly_score']:.4f}")
            
            if len(videos) > 10:
                print(f"  ... and {len(videos) - 10} more")
            
            # Save to file
            with open(os.path.join(save_dir, f'{key}.txt'), 'w') as f:
                f.write(f"{name}\n{'='*80}\n\n")
                for i, v in enumerate(sorted_videos, 1):
                    f.write(f"{i}. {v['video_name']}\n")
                    f.write(f"   Category: {v['category']}\n")
                    f.write(f"   Score: {v['anomaly_score']:.6f}\n")
                    f.write(f"   Path: {v['path']}\n\n")
            
            # Save CSV
            df = pd.DataFrame(sorted_videos)
            df.to_csv(os.path.join(save_dir, f'{key}.csv'), index=False)
            print(f"  📄 Saved: {key}.txt and {key}.csv")
        
        # Save JSON
        with open(os.path.join(save_dir, 'all_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"All files saved in: {save_dir}/")
        print(f"{'='*80}\n")


def test_time_augmentation(model, input_tensor, args, device='cuda'):
    predictions, features = [], []
    
    model.eval()
    with torch.no_grad():
        logits, feat = model(input_tensor)
        predictions.append(torch.sigmoid(logits))
        features.append(feat)
        
        input_t = torch.flip(input_tensor, dims=[2])
        logits_t, feat_t = model(input_t)
        predictions.append(torch.sigmoid(logits_t))
        features.append(feat_t)
        
        if not args.use_flow:
            input_h = torch.flip(input_tensor, dims=[4])
            logits_h, feat_h = model(input_h)
            predictions.append(torch.sigmoid(logits_h))
            features.append(feat_h)
    
    return torch.mean(torch.stack(predictions), 0), torch.mean(torch.stack(features), 0)


def plot_roc_curve(labels, scores, name, use_flow=False, use_tta=False):
    labels, scores = np.array(labels, dtype=np.int32), np.array(scores, dtype=np.float32)
    
    print("\n📈 ROC curve...")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # OPTIONAL: Youden's J optimal point (uncomment to show)
    optimal_idx = np.argmax(tpr - fpr)
    opt_thr, opt_tpr, opt_fpr = thresholds[optimal_idx], tpr[optimal_idx], fpr[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'darkorange', lw=3, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
    
    # OPTIONAL: Show optimal threshold point (Youden's J statistic)
    plt.plot(opt_fpr, opt_tpr, 'ro', ms=12, label=f'Optimal={opt_thr:.3f}')
    
    plt.xlabel('FPR', fontsize=14, fontweight='bold')
    plt.ylabel('TPR', fontsize=14, fontweight='bold')
    title = 'ROC Curve' + (' (X3D+Flow)' if use_flow else '')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.savefig(name + "_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name}_roc.png (AUC={roc_auc:.4f})")


def plot_precision_recall_curve(labels, scores, name, use_flow=False, use_tta=False):
    labels, scores = np.array(labels, dtype=np.int32), np.array(scores, dtype=np.float32)
    
    print("📊 PR curve...")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'green', lw=3, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    title = 'Precision-Recall Curve' + (' (X3D+Flow)' if use_flow else '')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.savefig(name + "_pr.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name}_pr.png (PR-AUC={pr_auc:.4f})")


def plot_threshold_performance(labels, scores, name):
    """Plot metrics vs threshold"""
    labels, scores = np.array(labels, dtype=np.int32), np.array(scores, dtype=np.float32)
    
    print("📈 Threshold analysis...")
    thresholds = np.linspace(0, 1, 100)
    accuracies, precisions, recalls, f1_list = [], [], [], []
    
    for t in thresholds:
        preds = (scores >= t).astype(int)
        acc = accuracy_score(labels, preds)
        
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        p = tp/(tp+fp) if tp+fp>0 else 0
        r = tp/(tp+fn) if tp+fn>0 else 0
        f1 = 2*p*r/(p+r) if p+r>0 else 0
        
        accuracies.append(acc)
        precisions.append(p)
        recalls.append(r)
        f1_list.append(f1)
    
    opt_idx = np.argmax(f1_list)
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, lw=2.5, label='Accuracy')
    plt.plot(thresholds, precisions, lw=2.5, label='Precision')
    plt.plot(thresholds, recalls, lw=2.5, label='Recall')
    plt.plot(thresholds, f1_list, lw=2.5, label='F1-Score')
    plt.axvline(thresholds[opt_idx], color='red', linestyle='--', lw=2, 
                label=f'Optimal = {thresholds[opt_idx]:.3f}')
    plt.axvline(0.5, color='gray', linestyle=':', lw=2, label='Default = 0.5')
    
    plt.xlabel('Threshold', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Performance Metrics vs Threshold - STEAD Base', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.savefig(name + "_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name}_threshold_analysis.png")
    return thresholds[opt_idx]


def plot_confusion_matrix(labels, scores, name, threshold=CONF_THRESHOLD):
    """Confusion matrix — single colorbar, no overlap, all bold"""

    labels = np.array(labels, dtype=np.int32)
    scores = np.array(scores, dtype=np.float32)

    print(f"📊 Confusion matrix (threshold={threshold})...")

    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    fig, ax = plt.subplots(figsize=(9, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Anomaly"]
    )

    # ✅ LET sklearn handle the colorbar (ONLY ONE)
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True
    )

    # -------- FORCE ALL TEXT BOLD --------
    for text in ax.texts:
        text.set_fontweight("bold")
        text.set_fontsize(16)

    # Axis tick labels
    ax.set_xticklabels(
        ["Normal", "Anomaly"],
        fontsize=14,
        fontweight="bold"
    )

    ax.set_yticklabels(
        ["Normal", "Anomaly"],
        fontsize=14,
        fontweight="bold",
        rotation=90,
        va="center"
    )

    # Axis labels
    ax.set_xlabel("Predicted label", fontsize=15, fontweight="bold")
    ax.set_ylabel("True label", fontsize=15, fontweight="bold")

    # Title — spacing fixed
    ax.set_title(
        f"Confusion Matrix - STRIDE BiLSTM DECOUPLEDLSTM (threshold={threshold})\n"
        f"Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}",
        fontsize=16,
        fontweight="bold",
        pad=22
    )

    # ✅ Increase top margin (NO bbox_inches tight)
    plt.subplots_adjust(top=0.82)

    plt.savefig(name + "_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"✅ Saved: {name}_confusion_matrix.png")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"  Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")


def plot_tsne(features, labels, name, use_flow=False, use_tta=False):
    print("🎨 t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[labels==0, 0], reduced[labels==0, 1],
               c='blue', label='Normal', alpha=0.6)
    plt.scatter(reduced[labels==1, 0], reduced[labels==1, 1],
               c='red', label='Anomaly', marker='*', s=100, alpha=0.6)
    plt.title('t-SNE Feature Embedding', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(name + "_tsne.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name}_tsne.png")


def plot_score_distribution(scores, labels, name):
    print("📊 Score distribution...")
    scores, labels = np.array(scores), np.array(labels)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.figure(figsize=(12, 6))
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
    plt.xlabel('Anomaly Score', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.title('Score Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, axis='y')
    
    plt.savefig(name + "_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name}_dist.png")


def plot_score_boxplot(labels, scores, name):
    print("📦 Boxplot...")
    labels, scores = np.array(labels), np.array(scores)
    data = [scores[labels == 0], scores[labels == 1]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bp = ax.boxplot(data, positions=[1, 2], widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    for i, (scores_data, pos) in enumerate(zip(data, [1, 2])):
        x = np.random.normal(pos, 0.04, size=len(scores_data))
        ax.scatter(x, scores_data, alpha=0.3, s=20, color=colors[i])
    
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Anomaly Score', fontsize=14, fontweight='bold')
    ax.set_title('Score Distribution by Class', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(name + "_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name}_boxplot.png")


# ============================================================================
# NEW ENHANCED VISUALIZATIONS
# ============================================================================

def plot_score_distribution_violin(pred, labels, name):
    """
    Create publication-quality violin plot showing score distributions
    for Normal vs Anomaly classes with statistical annotations
    """
    print("🎻 Creating Score Distribution Violin Plot...")
    
    pred = np.array(pred)
    labels = np.array(labels)
    
    # Separate scores by class
    normal_scores = pred[labels == 0]
    anomaly_scores = pred[labels == 1]
    
    # Calculate statistics
    normal_mean = normal_scores.mean()
    normal_std = normal_scores.std()
    anomaly_mean = anomaly_scores.mean()
    anomaly_std = anomaly_scores.std()
    separation = anomaly_mean - normal_mean
    
    # Prepare data for violin plot
    data_list = [normal_scores, anomaly_scores]
    
    # Create figure - INCREASED HEIGHT for better legend placement
    fig, ax = plt.subplots(figsize=(10,7))  # Changed from (10, 7) to (12, 9)
    
    # Create violin plot
    positions = [1, 2]
    parts = ax.violinplot(
        data_list,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )
    
    # Customize violin colors
    colors = ['#3498db', '#e74c3c']  # Blue for Normal, Red for Anomaly
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Customize other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(2)
    
    # Add scatter points for individual samples (with jitter)
    for i, (scores_data, pos, color) in enumerate(zip(data_list, positions, colors)):
        # Sample points if too many (for clarity)
        if len(scores_data) > 100:
            indices = np.random.choice(len(scores_data), 100, replace=False)
            scores_sample = scores_data[indices]
        else:
            scores_sample = scores_data
        
        x = np.random.normal(pos, 0.04, size=len(scores_sample))
        ax.scatter(x, scores_sample, alpha=0.3, s=30, color=color, edgecolors='black', linewidths=0.5)
    
    # Add horizontal lines for means
    ax.hlines(normal_mean, 0.7, 1.3, colors='blue', linestyles='--', linewidth=2, label=f'Normal Mean: {normal_mean:.3f}')
    ax.hlines(anomaly_mean, 1.7, 2.3, colors='red', linestyles='--', linewidth=2, label=f'Anomaly Mean: {anomaly_mean:.3f}')
    
    # Formatting
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Normal Videos', 'Anomaly Videos'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Anomaly Score', fontsize=14, fontweight='bold')
    ax.set_title('Score Distribution by Class - Violin Plot', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.2)  # Keep y-axis at 0-1
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Legend - TOP LEFT CORNER (with increased height, no overlap)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add statistics text box - TOP RIGHT CORNER
    stats_text = (
        f'Normal: μ={normal_mean:.3f}, σ={normal_std:.3f}\n'
        f'Anomaly: μ={anomaly_mean:.3f}, σ={anomaly_std:.3f}\n'
        f'Separation: {separation:.3f}\n'
        f'Normal samples: {len(normal_scores)}\n'
        f'Anomaly samples: {len(anomaly_scores)}'
    )
    
    ax.text(
        0.98, 0.97, stats_text,  # TOP RIGHT corner
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
    )
    
    plt.tight_layout()
    plt.savefig(name + "_score_distribution_violin.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {name}_score_distribution_violin.png")
    print(f"   Normal: {len(normal_scores)} videos, μ={normal_mean:.3f}, σ={normal_std:.3f}")
    print(f"   Anomaly: {len(anomaly_scores)} videos, μ={anomaly_mean:.3f}, σ={anomaly_std:.3f}")
    print(f"   Separation: {separation:.3f}")


def plot_per_category_performance(video_results, name, threshold=0.5):
    """
    Publication-quality per-category performance analysis

    FIXES:
    ✔ Correct overall AUC computation
    ✔ Correct per-category AUC (category vs normal)
    ✔ Removes invalid anomaly-only AUC logic
    ✔ Handles edge cases safely
    """

    print("📊 Creating Per-Category Performance Analysis...")

    # Extract data
    video_paths = video_results.video_paths
    predictions = np.array(video_results.predictions)
    labels = np.array(video_results.labels)

    categories = [video_results.get_category(path) for path in video_paths]

    # -----------------------------
    # Separate NORMAL videos (needed for proper AUC)
    # -----------------------------
    normal_preds = predictions[labels == 0]
    normal_labels = labels[labels == 0]

    # -----------------------------
    # Group anomaly videos by category
    # -----------------------------
    category_data = {}

    for path, pred, label, cat in zip(video_paths, predictions, labels, categories):

        if label == 0:
            continue  # Skip normal here (handled separately)

        if cat not in category_data:
            category_data[cat] = {'predictions': [], 'labels': []}

        category_data[cat]['predictions'].append(pred)
        category_data[cat]['labels'].append(label)

    # -----------------------------
    # Compute per-category metrics
    # -----------------------------
    category_metrics = {}

    for cat, data in category_data.items():

        anomaly_preds = np.array(data['predictions'])
        anomaly_labels = np.array(data['labels'])

        # ✔ Combine with normal samples → Proper AUC
        combined_preds = np.concatenate([normal_preds, anomaly_preds])
        combined_labels = np.concatenate([normal_labels, anomaly_labels])

        # Safe AUC computation
        if len(np.unique(combined_labels)) < 2:
            auc_score = np.nan
        else:
            auc_score = roc_auc_score(combined_labels, combined_preds) * 100

        detection_rate = (anomaly_preds >= threshold).mean() * 100
        count = len(anomaly_preds)
        mean_score = anomaly_preds.mean()

        category_metrics[cat] = {
            'auc': auc_score,
            'detection_rate': detection_rate,
            'count': count,
            'mean_score': mean_score
        }

    # -----------------------------
    # Sort by AUC
    # -----------------------------
    sorted_categories = sorted(
        category_metrics.items(),
        key=lambda x: (np.nan_to_num(x[1]['auc'], nan=0)),
        reverse=True
    )

    cats = [item[0] for item in sorted_categories]
    aucs = [item[1]['auc'] for item in sorted_categories]
    detection_rates = [item[1]['detection_rate'] for item in sorted_categories]
    counts = [item[1]['count'] for item in sorted_categories]

    # -----------------------------
    # ✔ CORRECT OVERALL AUC
    # -----------------------------
    if len(np.unique(labels)) < 2:
        overall_auc = np.nan
    else:
        overall_auc = roc_auc_score(labels, predictions) * 100

    overall_detection = (predictions[labels == 1] >= threshold).mean() * 100

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # ========== AUC Plot ==========
    colors = ['#2ecc71' if auc >= 90 else '#f39c12' if auc >= 85 else '#e74c3c' for auc in aucs]

    bars1 = ax1.bar(range(len(cats)), aucs, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)

    if not np.isnan(overall_auc):
        ax1.axhline(overall_auc, color='blue', linestyle='--', linewidth=2,
                    label=f'Overall AUC: {overall_auc:.2f}%')

    for bar, auc, count in zip(bars1, aucs, counts):
        height = bar.get_height()
        label = "NaN" if np.isnan(auc) else f"{auc:.1f}%"
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{label}\n(n={count})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(range(len(cats)))
    ax1.set_xticklabels(cats, rotation=45, ha='right')
    ax1.set_ylabel('AUC (%)')
    ax1.set_title('Per-Category AUC Performance')
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========== Detection Rate ==========
    colors2 = ['#2ecc71' if dr >= 85 else '#f39c12' if dr >= 70 else '#e74c3c' for dr in detection_rates]

    bars2 = ax2.bar(range(len(cats)), detection_rates, color=colors2,
                    alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.axhline(overall_detection, color='purple', linestyle='--', linewidth=2,
                label=f'Overall Detection: {overall_detection:.2f}%')

    for bar, dr in zip(bars2, detection_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{dr:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(range(len(cats)))
    ax2.set_xticklabels(cats, rotation=45, ha='right')
    ax2.set_ylabel('Detection Rate (%)')
    ax2.set_title(f'Detection Rate (Threshold={threshold})')
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(name + "_per_category_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {name}_per_category_performance.png")



def plot_combined_score_analysis(pred, labels, name):
    """
    Create a comprehensive 2x2 grid showing multiple score distribution views
    """
    print("📊 Creating Combined Score Analysis...")
    
    pred = np.array(pred)
    labels = np.array(labels)
    
    normal_scores = pred[labels == 0]
    anomaly_scores = pred[labels == 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Violin Plot (Top Left)
    ax1 = axes[0, 0]
    parts = ax1.violinplot([normal_scores, anomaly_scores], positions=[1, 2], 
                           widths=0.7, showmeans=True, showmedians=True)
    colors = ['#3498db', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Normal', 'Anomaly'], fontweight='bold')
    ax1.set_ylabel('Anomaly Score', fontweight='bold')
    ax1.set_title('Violin Plot', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Histogram (Top Right)
    ax2 = axes[0, 1]
    ax2.hist(normal_scores, bins=30, alpha=0.6, color='blue', label='Normal', edgecolor='black')
    ax2.hist(anomaly_scores, bins=30, alpha=0.6, color='red', label='Anomaly', edgecolor='black')
    ax2.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax2.set_xlabel('Anomaly Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Score Histogram', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. KDE Plot (Bottom Left)
    ax3 = axes[1, 0]
    kde_normal = gaussian_kde(normal_scores)
    kde_anomaly = gaussian_kde(anomaly_scores)
    x_range = np.linspace(0, 1, 200)
    ax3.plot(x_range, kde_normal(x_range), color='blue', lw=2, label='Normal')
    ax3.plot(x_range, kde_anomaly(x_range), color='red', lw=2, label='Anomaly')
    ax3.fill_between(x_range, kde_normal(x_range), alpha=0.3, color='blue')
    ax3.fill_between(x_range, kde_anomaly(x_range), alpha=0.3, color='red')
    ax3.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold')
    ax3.set_xlabel('Anomaly Score', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Kernel Density Estimation', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Distribution (Bottom Right)
    ax4 = axes[1, 1]
    sorted_normal = np.sort(normal_scores)
    sorted_anomaly = np.sort(anomaly_scores)
    ax4.plot(sorted_normal, np.linspace(0, 1, len(sorted_normal)), 
            color='blue', lw=2, label='Normal CDF')
    ax4.plot(sorted_anomaly, np.linspace(0, 1, len(sorted_anomaly)), 
            color='red', lw=2, label='Anomaly CDF')
    ax4.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_xlabel('Anomaly Score', fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontweight='bold')
    ax4.set_title('Cumulative Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(name + "_combined_score_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {name}_combined_score_analysis.png")


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test(dataloader, model, args, device='cuda', name="model", main=False, 
         use_tta=False, video_analysis=False):
    """Enhanced testing with all features"""
    model.to(device)
    
    if video_analysis and main:
        video_results = VideoAnalysisResults()
        video_paths = [p.strip() for p in dataloader.dataset.x3d_list]
    
    with torch.no_grad():
        model.eval()
        pred, labels, feats = [], [], []
        batch_idx = 0  # Track overall sample index
        
        for i, (feat_input, label) in tqdm(enumerate(dataloader), desc="Testing"):
            batch_size = feat_input.size(0)
            labels += label.cpu().tolist()
            input = feat_input.to(device)
            
            if use_tta:
                scores, feat = test_time_augmentation(model, input, args, device)
            else:
                logits, feat = model(input)
                scores = torch.sigmoid(logits)
            
            # Handle both single samples and batches
            scores_list = scores.squeeze().cpu().tolist()
            if not isinstance(scores_list, list):
                scores_list = [scores_list]
            pred += scores_list
            
            feat_np = feat.cpu().numpy()
            if len(feat_np.shape) == 1:
                feats.append(feat_np)
            else:
                feats.extend(list(feat_np))
            
            # Video analysis - handle batches
            if video_analysis and main:
                label_list = label.cpu().tolist()
                if not isinstance(label_list, list):
                    label_list = [label_list]
                
                for j in range(batch_size):
                    if batch_idx < len(video_paths):
                        video_results.add_result(
                            video_paths[batch_idx],
                            scores_list[j],
                            label_list[j],
                            feat_np[j] if len(feat_np.shape) > 1 else feat_np
                        )
                        batch_idx += 1
        
        # Metrics
        fpr, tpr, _ = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labels, pred)
        pr_auc = auc(recall, precision)
        
        print(f'\n{"="*60}')
        print(f'RESULTS: ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}')
        print(f'{"="*60}')
        
        if main:
            print("\n🎨 Generating visualizations...")
            
            feats_np = np.array(feats)
            labels_np = np.array(labels)
            
            # ORIGINAL 8 VISUALIZATIONS
            # 1. ROC Curve
            plot_roc_curve(labels, pred, name, args.use_flow, use_tta)
            
            # 2. PR Curve
            plot_precision_recall_curve(labels, pred, name, args.use_flow, use_tta)
            
            # 3. Threshold Analysis (MOST IMPORTANT!)
            optimal_threshold = plot_threshold_performance(labels, pred, name)
            print(f"   📊 Optimal threshold: {optimal_threshold:.3f}")
            
            # 4. Confusion Matrix
            plot_confusion_matrix(labels, pred, name, CONF_THRESHOLD)
            
            # 5. t-SNE
            plot_tsne(feats_np, labels_np, name, use_flow=args.use_flow, use_tta=use_tta)
            
            # 6. Score Distribution
            plot_score_distribution(pred, labels, name)
            
            # 7. UMAP
            print("🗺️  UMAP...")
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
            reduced = reducer.fit_transform(feats_np)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced[labels_np==0, 0], reduced[labels_np==0, 1],
                       c='blue', label='Normal', alpha=0.6)
            plt.scatter(reduced[labels_np==1, 0], reduced[labels_np==1, 1],
                       c='red', label='Anomaly', marker='*', s=100, alpha=0.6)
            plt.title('UMAP Embedding', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.savefig(name + "_umap.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {name}_umap.png")
            
            # 8. Boxplot
            plot_score_boxplot(labels, pred, name)
            
            print("\n✅ All 8 original visualizations complete!")
            print("  1. ROC Curve")
            print("  2. PR Curve")
            print("  3. Threshold Analysis ⭐")
            print("  4. Confusion Matrix")
            print("  5. t-SNE")
            print("  6. Score Distribution")
            print("  7. UMAP")
            print("  8. Boxplot")
            
            # ============ NEW ENHANCED VISUALIZATIONS ============
            print("\n🌟 Generating ENHANCED visualizations...")
            
            # 9. Score Distribution Violin Plot
            plot_score_distribution_violin(pred, labels, name)
            
            # 10. Combined Score Analysis (4-in-1)
            plot_combined_score_analysis(pred, labels, name)
            
            # 11. Per-Category Performance (if video analysis enabled)
            if video_analysis:
                plot_per_category_performance(video_results, name, threshold=CONF_THRESHOLD)
            
            print("\n✅ All 11 visualizations complete!")
            print("  Original 8 + Enhanced 3:")
            print("  9. Violin Plot with Statistics ⭐")
            print("  10. Combined Score Analysis (4-in-1) ⭐")
            print("  11. Per-Category Performance ⭐")
            
            print("\n" + "="*60)
            #print(f"🎯 OPTIMAL THRESHOLD: {optimal_threshold:.3f}")
            print(f"   (Based on maximum F1-Score)")
            print("="*60)
            
            # Video analysis
            if video_analysis:
                results = video_results.analyze_results(CONF_THRESHOLD)
                video_results.save_detailed_report(results)
        
        return roc_auc, pr_auc


if __name__ == '__main__':
    args = option.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("🚀 ENHANCED MODEL TESTING")
    print("="*60)
    
    input_channels = 198 if args.use_flow else 192
    print(f"Mode: {'X3D+Flow' if args.use_flow else 'X3D only'}")
    print(f"Channels: {input_channels}")
    print("="*60)
    
    use_tta = getattr(args, 'use_tta', False)
    
    test_loader = DataLoader(
        SpatioTemporalDataset(args, test_mode=True),
        batch_size=1 if use_tta else args.batch_size,
        shuffle=False
    )
    
    # Load model
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
            use_convlstm=args.use_convlstm,
            use_decoupled_lstm=args.use_decoupled_lstm,
        )
    else:
        model = EnhancedModel(
            dropout=args.dropout_rate,
            attn_dropout=args.attn_dropout_rate,
            ff_mult=1,
            dims=(32, 32),
            depths=(1, 1),
            input_channels=input_channels
        )
    
    model = model.to(device)
    
    if hasattr(args, 'pretrained_ckpt') and args.pretrained_ckpt:
        model_path = args.pretrained_ckpt
    else:
        model_path = MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION
    
    print(f"\n📂 Loading: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded!\n")
    
    # Test with video analysis
    auc_score, pr_auc_score = test(
        test_loader, model, args, device,
        name=MODEL_NAME, main=True, use_tta=use_tta,
        video_analysis=True
    )
    
    print("\n" + "="*60)
    print("🎉 COMPLETE!")
    print(f"ROC-AUC: {auc_score:.4f} | PR-AUC: {pr_auc_score:.4f}")
    print("="*60)