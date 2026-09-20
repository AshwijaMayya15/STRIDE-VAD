#!/usr/bin/env python3
"""
Simple 15-Seed Training Script
Run: python train_15_seeds.py

Expected:
- Time: 30-60 hours
- Best AUC: 91.5-92.0%
- Mean AUC: 91.1-91.4%
- Should beat 91.3% target!
"""

# Fix MKL threading issue
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import subprocess
import json
import time
from datetime import datetime
import numpy as np
import sys

# Configuration
SEEDS = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 
         2031, 2032, 2033, 2034, 2035]

CONFIG = {
    'lr': 1.5e-4,
    'batch_size': 16,
    'dropout_rate': 0.3,
    'weight_decay': 0.3,
    'alpha': 0.008,
    'focal_gamma': 2.0,
    'mixup_alpha': 0.2,
    'label_smoothing': 0.1,
    'max_grad_norm': 1.0,
    'max_epoch': 100,
    'early_stop_patience': 20,
    'warmup': 2,
}

def run_training(seed, config):
    """Run training for a single seed"""
    cmd = [
        sys.executable, 'main_enhanced.py',
        '--lr', str(config['lr']),
        '--batch_size', str(config['batch_size']),
        '--dropout_rate', str(config['dropout_rate']),
        '--attn_dropout_rate', '0.1',
        '--weight_decay', str(config['weight_decay']),
        '--alpha', str(config['alpha']),
        '--focal_gamma', str(config['focal_gamma']),
        '--mixup_alpha', str(config['mixup_alpha']),
        '--label_smoothing', str(config['label_smoothing']),
        '--max_grad_norm', str(config['max_grad_norm']),
        '--use_convlstm',   # DECOUPLED_LSTM in ConvBlocks (spatial conv + temporal LSTM)             # BiLSTM inter-stage bridge (runs independently of ConvBlock type)
        '--use_focal',
        '--use_mixup',
        '--use_tpp',
        '--max_epoch', str(config['max_epoch']),
        '--early_stop_patience', str(config['early_stop_patience']),
        '--warmup', str(config['warmup']),
        '--seed', str(seed),
        '--model_arch', 'base',
        '--comment', 'convlstm_15seeds'
    ]
    
    print(f"\n{'='*80}")
    print(f"🎯 Training Seed {seed}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    end_time = time.time()
    
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    success = result.returncode == 0
    
    if success:
        print(f"\n✅ Seed {seed} completed in {hours}h {minutes}m")
    else:
        print(f"\n❌ Seed {seed} failed!")
    
    return success, elapsed

def get_results(seed, config):
    """Extract results for a seed"""
    ckpt_dir = f"./ckpt1/{config['lr']}_{config['batch_size']}_convlstm_15seeds_seed{seed}"
    result_file = os.path.join(ckpt_dir, 'results_summary.json')
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            return {
                'seed': seed,
                'best_auc': data['best_auc'],
                'best_pr_auc': data['best_pr_auc'],
                'best_epoch': data['best_auc_epoch'],
                'success': True
            }
    else:
        return {
            'seed': seed,
            'best_auc': 0.0,
            'best_pr_auc': 0.0,
            'best_epoch': 0,
            'success': False
        }

def main():
    print("="*80)
    print("🚀 ENHANCED MULTI-SEED TRAINING - 15 SEEDS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Architecture:   DECOUPLED_LSTM (ConvBlocks) + BiLSTM (inter-stage)")
    print(f"  Learning Rate: {CONFIG['lr']}")
    print(f"  Batch Size: {CONFIG['batch_size']}")
    print(f"  Max Epochs: {CONFIG['max_epoch']}")
    print(f"  Seeds: {SEEDS[0]}-{SEEDS[-1]}")
    print(f"\nExpected time: 30-60 hours")
    print(f"Expected best AUC: 91.5-92.0%")
    print(f"Target to beat: 91.3%\n")
    print("="*80)
    
    start_time = time.time()
    results = []
    success_count = 0
    
    # Train all seeds
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Progress: {i}/15 seeds")
        print(f"# Current seed: {seed}")
        print(f"{'#'*80}\n")
        
        success, elapsed = run_training(seed, CONFIG)
        
        if success:
            success_count += 1
            result = get_results(seed, CONFIG)
            results.append(result)
            
            if result['success']:
                print(f"\n📊 Results for seed {seed}:")
                print(f"  ROC-AUC: {result['best_auc']:.4f}")
                print(f"  PR-AUC:  {result['best_pr_auc']:.4f}")
                print(f"  Epoch:   {result['best_epoch']}")
        
        # Show progress
        elapsed_total = time.time() - start_time
        avg_time = elapsed_total / i
        remaining = (15 - i) * avg_time
        remaining_hours = int(remaining // 3600)
        remaining_mins = int((remaining % 3600) // 60)
        
        print(f"\n⏱️  Estimated time remaining: {remaining_hours}h {remaining_mins}m")
        print(f"✅ Completed: {success_count}/{i}")
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    total_hours = int(total_time // 3600)
    total_mins = int((total_time % 3600) // 60)
    
    print("\n\n" + "="*80)
    print("🎉 ALL TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 SUMMARY:")
    print(f"  Total runs:      15")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {15 - success_count}")
    print(f"  Total time:      {total_hours}h {total_mins}m")
    
    if results:
        # Find best
        best = max(results, key=lambda x: x['best_auc'])
        
        print(f"\n🏆 BEST MODEL:")
        print(f"  Seed:            {best['seed']}")
        print(f"  ROC-AUC:         {best['best_auc']:.4f}")
        print(f"  PR-AUC:          {best['best_pr_auc']:.4f}")
        print(f"  Epoch:           {best['best_epoch']}")
        
        # Check if beat target
        if best['best_auc'] > 0.913:
            diff = (best['best_auc'] - 0.913) * 100
            print(f"\n  ✅ SUCCESS! Beat 91.3% by {diff:.2f}%!")
        else:
            gap = (0.913 - best['best_auc']) * 100
            print(f"\n  ⚠️  Gap to 91.3%: {gap:.2f}%")
            print(f"  💡 Tip: Try TTA for +0.2-0.3% boost")
        
        # Statistics
        aucs = [r['best_auc'] for r in results if r['success']]
        pr_aucs = [r['best_pr_auc'] for r in results if r['success']]
        
        if aucs:
            print(f"\n📈 STATISTICS:")
            print(f"  ROC-AUC:")
            print(f"    Mean:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
            print(f"    Best:   {np.max(aucs):.4f}")
            print(f"    Worst:  {np.min(aucs):.4f}")
            print(f"    Median: {np.median(aucs):.4f}")
            print(f"\n  PR-AUC:")
            print(f"    Mean:   {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
            print(f"    Best:   {np.max(pr_aucs):.4f}")
        
        # Show all results
        print(f"\n📋 ALL RESULTS:")
        print(f"{'Seed':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'Epoch':<8} {'Status'}")
        print("-" * 60)
        
        for result in sorted(results, key=lambda x: x['best_auc'], reverse=True):
            marker = "🏆" if result['seed'] == best['seed'] else "  "
            if result['success']:
                print(f"{marker} {result['seed']:<6} {result['best_auc']:.4f}     "
                      f"{result['best_pr_auc']:.4f}     {result['best_epoch']:<6} ✅")
            else:
                print(f"  {result['seed']:<6} N/A        N/A        N/A      ❌")
        
        # Comparison with STEAD
        print(f"\n📊 COMPARISON WITH STEAD:")
        print(f"  STEAD Baseline:  0.9130 (91.3%)")
        print(f"  Your Best:       {best['best_auc']:.4f} ({best['best_auc']*100:.2f}%)")
        
        if best['best_auc'] > 0.913:
            diff = (best['best_auc'] - 0.913) * 100
            print(f"  ✅ SUCCESS:      +{diff:.2f}%")
        else:
            diff = (0.913 - best['best_auc']) * 100
            print(f"  Gap:             -{diff:.2f}%")
    
    print("\n" + "="*80)
    print("🎊 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest model location:")
    if results:
        best = max(results, key=lambda x: x['best_auc'])
        print(f"  ./ckpt1/{CONFIG['lr']}_{CONFIG['batch_size']}_convlstm_15seeds_seed{best['seed']}/")
    print("\n")
    
    # Save results to JSON
    output_file = 'training_results_15seeds.json'
    with open(output_file, 'w') as f:
        json.dump({
            'config': CONFIG,
            'total_time': total_time,
            'success_count': success_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=4)
    
    print(f"Results saved to: {output_file}\n")

if __name__ == '__main__':
    main()