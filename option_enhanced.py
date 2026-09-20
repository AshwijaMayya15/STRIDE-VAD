import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='STEAD Enhanced - Spatial-Temporal Anomaly Detection')
    
    # Flow usage flag
    parser.add_argument('--use_flow', action='store_true', default=False, help='Use fused features (X3D + Optical Flow)')
    # X3D feature file lists
    parser.add_argument('--rgb_list', default='ucf_x3d_train.txt', help='List of X3D training features')
    parser.add_argument('--test_rgb_list', default='ucf_x3d_test.txt', help='List of X3D test features')
    
    # Flow feature file lists
    parser.add_argument('--flow_list', default='ucf_flow_train.txt', help='List of Flow training features')
    parser.add_argument('--test_flow_list', default='ucf_flow_test.txt', help='List of Flow test features')
    
    # Training parameters
    parser.add_argument('--comment', default='enhanced', help='Comment for the ckpt name')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate (default: 1.5e-4)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--alpha', type=float, default=0.008, help='Weight for triplet loss (default: 0.008)')
    
    # Model parameters
    parser.add_argument('--model_name', default='model', help='Name to save model')
    parser.add_argument('--model_arch', default='base', help='Model architecture: base or fast or tiny')
    parser.add_argument('--pretrained_ckpt', default=None, help='Checkpoint for pretrained model')
    
    # Enhanced model features
    parser.add_argument('--use_se', action='store_true', default=False, help='Use Squeeze-and-Excitation blocks')
    parser.add_argument('--use_tpp', action='store_true', default=False, help='Use Temporal Pyramid Pooling')
    parser.add_argument('--use_lstm', action='store_true', default=False, help='Use Bidirectional LSTM layer')
    parser.add_argument('--lstm_hidden', type=int, default=64, help='LSTM hidden dimension (default: 64)')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    parser.add_argument('--use_convlstm', action='store_true', default=False, help='Use ConvLSTM in conv blocks (heavy)')
    parser.add_argument('--use_decoupled_lstm', action='store_true', default=False, help='Use hybrid DECOUPLED_LSTM (spatial conv + temporal LSTM)')

    # Training schedule
    parser.add_argument('--max_epoch', type=int, default=100, help='Max training epochs (default: 100)')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for reproducibility')
    
    # Enhanced training techniques
    parser.add_argument('--use_focal', action='store_true', default=True, help='Use Focal Loss instead of BCE')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    parser.add_argument('--use_mixup', action='store_true', default=True, help='Use Mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--weight_decay', type=float, default=0.3, help='Weight decay (default: 0.3)')
    parser.add_argument('--attn_heads', type=int, default=16, help='Performer attention heads')
    
    # Testing enhancements
    parser.add_argument('--use_tta', action='store_true', default=False, help='Use Test-Time Augmentation')
    parser.add_argument('--use_ensemble', action='store_true', default=False, help='Use model ensemble')
    parser.add_argument('--ensemble_dir', type=str, default=None, help='Directory containing models for ensemble')
    
    args = parser.parse_args()
    
    return args
