# Data Preprocessing

Scripts and split lists for preparing X3D features used to train/evaluate STRIDE on UCF-Crime and XD-Violence.

## Contents

```
Data Preprocessing/
├── x3dextract.py         # Extracts X3D appearance features from raw video clips
├── AnomalyTrain.txt      # Paths to extracted X3D feature files (train split)
├── AnomalyTest.txt       # Paths to extracted X3D feature files (test split)
├── ucf_x3d_train.txt     # UCF-Crime X3D feature list (train split)
├── ucf_x3d_test.txt      # UCF-Crime X3D feature list (test split)
├── xd_x3d_train.txt      # XD-Violence X3D feature list (train split)
└── xd_x3d_test.txt       # XD-Violence X3D feature list (test split)
```

## Option 1: Use pre-extracted X3D features (recommended)

Pre-extracted X3D features for both datasets are hosted on Hugging Face:

- **UCF-Crime (X3D):** https://huggingface.co/datasets/CosmicAstro/UCF-Crime-X3D
- **XD-Violence (X3D):** https://huggingface.co/datasets/CosmicAstro/XD-Violence-X3D-Features

```bash
pip install huggingface_hub

huggingface-cli download CosmicAstro/UCF-Crime-X3D --repo-type dataset --local-dir ./data/ucf_crime_x3d
huggingface-cli download CosmicAstro/XD-Violence-X3D-Features --repo-type dataset --local-dir ./data/xd_violence_x3d
```

Point `option_enhanced.py`'s feature-path args at these local directories, and use the `*_x3d_train.txt` / `*_x3d_test.txt` lists in this folder as the corresponding split files.

## Option 2: Extract features yourself from raw videos

1. Download the raw datasets:
   - **UCF-Crime:** https://www.crcv.ucf.edu/projects/real-world/
   - **XD-Violence:** https://roc-ng.github.io/XD-Violence/

2. Run extraction:
   ```bash
   python x3dextract.py --input_dir /path/to/raw_videos --output_dir /path/to/x3d_features
   ```

3. Use the split lists in this folder (`ucf_x3d_train.txt`, `ucf_x3d_test.txt`, `xd_x3d_train.txt`, `xd_x3d_test.txt`) to map extracted features to train/test splits, matching the format STRIDE's `dataset.py` expects. `AnomalyTrain.txt` and `AnomalyTest.txt` hold the direct file paths to the extracted X3D feature files and are what `dataset.py` reads at load time — regenerate these to point at your own `x3dextract.py` output directory if paths differ from the originals.

## Notes

- X3D features (not optical flow) are what STRIDE's released configs use.
- Feature file naming/order in each list file must match the extracted output of `x3dextract.py`.
