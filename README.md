# STRIDE: Spatial-Temporal Recurrent Intelligence for Detecting Events in Weakly-Supervised Video Anomaly Detection

STRIDE is a weakly-supervised video anomaly detection (VAD) model evaluated on UCF-Crime and XD-Violence. It combines a DECOUPLED-LSTM module, Squeeze-and-Excitation blocks, Temporal Pyramid Pooling, Performer attention, and a Focal + Triplet loss over X3D features.

## Highlights

- Video-level ROC-AUC **92.13 ± 0.57** / PR-AUC **92.18 ± 0.71** on UCF-Crime (15-seed mean)
- Video-level ROC-AUC **89.52 ± 0.54** / PR-AUC **93.09 ± 0.49** on XD-Violence (15-seed mean)

## Repository Structure

```
.
├── dataset.py              # Dataset loading / preprocessing
├── model_enhanced.py        # STRIDE model architecture
├── option_enhanced.py       # Training/eval config & CLI args
├── main_enhanced.py         # Entry point
├── train_enhanced.py        # Training loop
├── test_enhanced.py         # Evaluation loop
├── train_15_seeds.py        # Multi-seed training driver (mean±std results)
├── utils.py                 # Shared utilities
├── ucf_x3d_train.txt        # UCF-Crime X3D feature list (train)
├── ucf_x3d_test.txt         # UCF-Crime X3D feature list (test)
└── xd_x3d_test.txt          # XD-Violence X3D feature list (test)
```

## Requirements

```bash
pip install -r requirements.txt
```

*(add a `requirements.txt` — torch, numpy, scikit-learn, etc.)*

## Usage

**Train:**
```bash
python main_enhanced.py --dataset ucf-crime --mode train
```

**Evaluate:**
```bash
python test_enhanced.py --dataset ucf-crime --checkpoint <path>
```

**Multi-seed training (mean ± std over 15 seeds):**
```bash
python train_15_seeds.py --dataset ucf-crime
```

### Key configs

- **DECOUPLED-LSTM (headline config)** — the single recommended setup; drives the main gain over the STEAD baseline.
  ```bash
  python main_enhanced.py --dataset ucf-crime --decoupled-lstm --no-se
  ```
- **Full model (+ SE blocks + BiLSTM bridge)** — adds Squeeze-and-Excitation and the BiLSTM bridge on top of DECOUPLED-LSTM. Bridge helps on UCF-Crime but is neutral/harmful on XD-Violence, so treat it as dataset-dependent rather than default.
  ```bash
  python main_enhanced.py --dataset ucf-crime --decoupled-lstm --se --bilstm-bridge
  ```

All flags are defined in `option_enhanced.py`; ablate any combination of `--decoupled-lstm`, `--se`, `--bilstm-bridge` from there.

### Output of `train_15_seeds.py`

Running it trains the model across 15 random seeds and writes:

```
saved_models/<config_name>/seed_<n>/model.pth     # per-seed checkpoint
runs/<config_name>/seed_<n>/                      # per-seed logs
aggregate_all.json                                # aggregated mean±std metrics across all 15 seeds
```

Use `aggregate.py` to (re)compute the mean±std summary from the per-seed results into `aggregate_all.json`.

## Datasets

Features are extracted with X3D (RGB appearance) and optical flow (RAFT/PTLFlow). Split lists are provided in `*_train.txt` / `*_test.txt` for UCF-Crime and XD-Violence.

## Citation

```bibtex
@article{mayya6628603stride,
  title={STRIDE: Spatial-Temporal Recurrent Intelligence for Detecting Events in Weakly-Supervised Video Anomaly Detection},
  author={Mayya, Ashwija and Khaire, Pushpajit A},
  journal={Available at SSRN 6628603}
}
```

## Contact

Ashwija Mayya (ASH) — M.Tech Research Scholar, Dept. of Mathematical and Computational Sciences, NITK Surathkal
Supervisor: Dr. Pushpajit Khaire
