# STRIDE: Spatial-Temporal Recurrent Intelligence for Detecting Events in Weakly-Supervised Video Anomaly Detection

Weakly-supervised video anomaly detection. X3D-L backbone with a configurable
temporal block (plain Conv, ConvLSTM, or DECOUPLED-LSTM), optional BiLSTM
bridge, SE channel attention, Performer attention, temporal pyramid pooling,
and a Focal + Triplet training objective.

This repo contains **code only** — no trained checkpoints. Pre-extracted
X3D features are available for download (see `Data Preprocessing/`), or
extract your own from raw video.

## Repository layout

```
model_enhanced.py       Backbone + head (Conv / ConvLSTM / DECOUPLED-LSTM / Attn blocks)
dataset.py               Video-level dataset, MIL pairing (normal/anomaly)
option_enhanced.py       CLI arguments
utils.py                 Shared helpers
main_enhanced.py          Training entry point
train_enhanced.py         Training loop
test_enhanced.py          Inference + evaluation visualizations

train_15_seeds.py         Multi-seed launcher (runs one config across a fixed seed set)
Data Preprocessing/       X3D feature extraction + split lists (UCF-Crime, XD-Violence)
```

## Requirements

```
torch
torchvision
pytorchvideo
performer-pytorch
torchinfo
scikit-learn
numpy
tqdm
umap-learn
matplotlib
seaborn
```

See `requirements.txt` for the full list; pin versions to your CUDA/PyTorch
setup. All commands below assume a Windows shell (PowerShell / cmd); adjust
path separators if running elsewhere.

## Data

See **`Data Preprocessing/README.md`** for full details. Two options:

- **Use pre-extracted X3D features (recommended)** — hosted on Hugging Face
  for both UCF-Crime and XD-Violence; download and point the split lists at
  the local directory.
- **Extract your own** from raw video with `Data Preprocessing/x3dextract.py`.

Either way, features are per-video `.npy`, shape `(192, 16, 10, 10)`
(X3D-only) or `(198, 16, 10, 10)` (X3D + 6-channel optical flow, fused), and
`dataset.py` reads train/test splits from plain-text list files
(`ucf_x3d_train.txt`, `ucf_x3d_test.txt`, `xd_x3d_train.txt`,
`xd_x3d_test.txt`, plus `AnomalyTrain.txt` / `AnomalyTest.txt`) already
provided in `Data Preprocessing/`. The label is inferred from the path: any
path containing `Normal` is the normal class; UCF-Crime paths follow
`<Category>\<video>.npy`, XD-Violence paths carry the label in the filename
suffix (`..._label_B1-B5.npy`).

## Training

```powershell
python main_enhanced.py `
    --rgb_list "Data Preprocessing\ucf_x3d_train.txt" --test_rgb_list "Data Preprocessing\ucf_x3d_test.txt" `
    --use_se --use_tpp --use_lstm --use_decoupled_lstm `
    --use_focal --use_mixup `
    --lr 1.5e-4 --batch_size 16 --dropout_rate 0.3 --weight_decay 0.3 `
    --alpha 0.008 `
    --seed 2021 --comment my_run
```

Key architecture flags:

| Flag | Effect |
|---|---|
| `--use_decoupled_lstm` | DECOUPLED-LSTM conv block (grouped depthwise conv + per-location BiLSTM) |
| `--use_convlstm` | Full ConvLSTM conv block (heavier alternative) |
| `--use_lstm` | BiLSTM bridge between backbone and pooling head |
| `--use_se` | Squeeze-and-Excitation channel attention per stage |
| `--use_tpp` | Temporal pyramid pooling (scales {1,2,4,8}); omit for plain max-pool |
| `--attn_heads N` | Performer attention heads (`dim_head = stage_dim // N`) |
| `--use_flow` | Fuse 6-channel optical flow with X3D features (198-ch input) |
| `--alpha` | Triplet loss weight (`0` disables triplet, focal-only) |

Checkpoints (`model_best_auc.pkl`, `model_best_pr.pkl`, per-epoch, and a
`config_*.txt` snapshot of every flag) are written under
`<RUNDIR>\ckpt1\<lr>_<batch>_<comment>_seed<seed>\`. Set the `RUNDIR`
environment variable to control the output root (defaults to the working
directory):

```powershell
$env:RUNDIR = "D:\STRIDE_runs"
python main_enhanced.py ...
```

## Multi-seed training

`train_15_seeds.py` launches one configuration across a fixed seed set
(2021–2035 by convention) for robust mean±std reporting rather than
single-run numbers. Edit the `CONFIG` dict at the top of the script for
your architecture flags and hyperparameters, then run:

```powershell
python train_15_seeds.py
```

Each seed trains sequentially in one process (no job scheduler assumed);
for parallel execution on a cluster, adapt the seed loop to your own
scheduler.

**Running many configurations:** edit the `CONFIG` dict (and the
architecture flags in the launch command) inside `train_15_seeds.py` for
each configuration you want to sweep, then rerun the script — one edit +
run per configuration.

**Running a single seed / one-off experiment:** skip `train_15_seeds.py`
and just pass the flags directly to `main_enhanced.py`, as in the
*Training* example above (e.g. `--use_decoupled_lstm --use_lstm --use_se
--use_tpp --alpha 0.008 --seed 2021`).

## Citation

```bibtex
@article{mayya6628603stride,
  title={STRIDE: Spatial-Temporal Recurrent Intelligence for Detecting Events in Weakly-Supervised Video Anomaly Detection},
  author={Mayya, Ashwija and Khaire, Pushpajit A},
  journal={Available at SSRN 6628603}
}
```

