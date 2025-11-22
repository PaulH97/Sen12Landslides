# Sen12Landslides: Spatio-Temporal Landslide & Anomaly Detection Dataset

A large-scale, multi-modal, multi-temporal collection of 128×128px Sentinel-1/2 + DEM patches with 10m spatial resolution and with 75k landslide annotations.

**Paper**: https://www.nature.com/articles/s41597-025-06167-2

**Dataset**: https://huggingface.co/datasets/paulhoehn/Sen12Landslides


## Quick Start
```bash
# Clone & setup
git clone https://github.com/PaulH97/Sen12Landslides.git
cd Sen12Landslides
pip install --upgrade huggingface_hub

# Download dataset
huggingface-cli login # paste your token from https://huggingface.co/settings/tokens (only once)
mkdir data
huggingface-cli download paulhoehn/Sen12Landslides --repo-type dataset --local-dir data

# Extract patches
for sensor in s1asc s1dsc s2; do
  for archive in data/$sensor/*.tar.gz; do
    tar -xzvf "$archive" -C "data/$sensor" && rm "$archive"
  done
done
```

## Dataset Overview

<table>
<tr>
<td>

Full Dataset
| Modality | Samples | Annotated | Ann. Rate |
|----------|:-------:|:---------:|:---------:|
| S1-asc   | 13,306  | 6,492     | 48.8%     |
| S1-dsc   | 12,622  | 6,347     | 50.3%     |
| S2       | 13,628  | 6,737     | 49.4%     |

</td>
<td>

Task Splits
| Modality | S12LS-LD      | S12LS-AD        |
|----------|:-------------:|:---------------:|
| S1-asc   | 4,793 (100%)  | 13,306 (48.8%)  |
| S1-dsc   | 4,666 (100%)  | 12,622 (50.3%)  |
| S2       | 4,988 (100%)  | 13,628 (49.4%)  |

</td>
</tr>
</table>

- S12LS-LD: Landslide detection with annotated patches 
- S12LS-AD: Anomaly detection with mixed annotated/non-annotated samples to learn normal vs. anomalous patterns
- See `Sen12Landslides/tasks/<task>/config.json` for split details

### Data Structure
```
Sen12Landslides/
├── data/
│   ├── inventories.shp.zip              # Ground-truth landslide polygons
│   ├── s1asc/                           # Sentinel-1 Ascending patches
│   │   └── <region>_s1asc_<id>.nc
│   ├── s1dsc/                           # Sentinel-1 Descending patches
│   └── s2/                              # Sentinel-2 patches
├── tasks/
│   ├── S12LS-LD/                        # Landslide detection task
│   │   ├── config.json
│   │   └── <modality>/
│   │       ├── splits.json              # Train/val/test file lists
│   │       ├── norm.json                # Per-band mean/std
│   │       └── patch_locations.geojson
│   └── S12LS-AD/                        # Anomaly detection task
│       └── ...
└── src/                                 # Data loaders, models, training
```

### Patch Format

Each `.nc` file contains 128×128 px across 15 time steps:

| Modality | Bands | Additional |
|----------|-------|------------|
| Sentinel-1 | VV, VH | DEM, MASK |
| Sentinel-2 | B02-B08, B8A, B11-B12 | SCL, DEM, MASK |
```python
>>> import xarray as xr
>>> ds = xr.open_dataset("Sen12Landslides/data/s2/italy_s2_6982.nc")
>>> ds
<xarray.Dataset> Size: 6MB
Dimensions:      (time: 15, x: 128, y: 128)
Coordinates:
  * x            (x) float64 1kB 7.552e+05 … 7.565e+05
  * y            (y) float64 1kB 4.882e+06 … 4.881e+06
  * time         (time) datetime64[ns] 2022-10-05 … 2023-09-10
Data variables: (12/14)
    B02          (time, x, y) int16 …
    B03          (time, x, y) int16 …
    …             
    B12          (time, x, y) int16 …
    SCL          (time, x, y) int16 …
    MASK         (time, x, y) uint8 …
    DEM          (time, x, y) int16 …
    spatial_ref  int64 8B  
Attributes:
    ann_id:           41125,41124,…  
    ann_bbox:         (755867.58,4880640.0,…)  
    event_date:       2023-05-16  
    date_confidence:  1.0  
    pre_post_dates:   {'pre': 7, 'post': 8}  
    annotated:        True  
    satellite:        s2  
    center_lat:       4881280.0  
    center_lon:       755840.0  
    crs:              EPSG:32632  
```

## Tasks

Some patches are challenging even for human experts (e.g., <10 annotated pixels, ambiguous temporal signatures, missing/noisy labels). We provide two task-specific configurations:

Creating custom splits:
```bash
python src/data/create_splits.py  # Configure in configs/splits/config.yaml
```

#### Always Generated (Root Level)
| File | Description |
|------|-------------|
| `config.json` | Filter criteria, split ratios, and stratification settings |

#### Per-Satellite Folders (`s1asc/`, `s1dsc/`, `s2/`)
| File | Description |
|------|-------------|
| `splits.json` | Train/val/test splits for this satellite modality |
| `norm.json` | Per-band normalization statistics (mean/std) for this satellite |
| `patch_locations.geojson` | Geographic patch locations with train/val/test assignments for this satellite |

#### Multi-Modal Files
| File | Description |
|------|-------------|
| `splits_aligned.json` | Train/val/test splits containing only patches available across all satellites |
| `norm_aligned.json` | Normalization statistics computed from aligned patches only |
| `patch_locations_aligned.geojson` | Geographic locations of patches available across all satellites |

**Usage:**
- Single-modal training: Load `<satellite>/splits.json` + `<satellite>/norm.json`
- Multi-modal training: Load `splits_aligned.json` + `norm_aligned.json` for cross-modal fusion
- Visualization: Open `patch_locations.geojson` in QGIS or mapping tools


## Training

This project uses [Hydra](https://hydra.cc/) for configuration management. See [Hydra documentation](https://hydra.cc/docs/intro/) for more details.

### Available Configurations

| Config | Options |
|--------|---------|
| `model` | `utae`, `convgru`, `unet3d`, `fpn_convlstm` |
| `dataset` | `sen12ls_s2`, `sen12ls_s1asc`, `sen12ls_s1dsc` |
| `trainer` | `cpu`, `gpu`, `ddp` |
| `lit_module` | `binary`, `multiclass` |


### Examples
```bash
# Train ConvGRU on Sentinel-2
python src/pipeline/train.py model=convgru dataset=sen12ls_s2

# Train UTAE on Sentinel-1 with DEM
python src/pipeline/train.py model=utae dataset=sen12ls_s1asc dataset.dem=true dataset.num_channels=3

# Multi-GPU training
python src/pipeline/train.py trainer.devices=4 trainer.strategy=ddp dataset=sen12ls_s2

# Multirun with three models
python src/pipeline/train.py --multirun model=utae,convlstm,convgru dataset=sen12ls_s2     
```

## Baselines

**Sen12Landslides** exhibits severe class imbalance, with landslide pixels constituting approximately **~3%** of the dataset. Consequently, standard metrics like Overall Accuracy (OA) are misleading (a model predicting "no landslide" everywhere would achieve ~97% OA). To ensure rigorous evaluation, we focus on metrics targeting the **positive class (Landslide)** rather than the background.

> **Note:** While our accompanying paper reports *macro-averaged* metrics to assess overall semantic consistency, we provide **Binary Metrics** (Class 1: Landslide) below. We strongly recommend using these for practical comparisons with other works on landlside detection.

### Benchmark Results (`S12LS-LD`)

**Coming soon**: An updated benchmark table utilizing the architectures described in the paper with **binary metrics**:

| Model | Precision | Recall | F1-Score | IoU | AP | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ConvGRU |  *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| U-ConvLSTM |  *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Unet3D | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| U-TAE | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Your Method** |  **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |  **0.00** |

Use provided `S12LS-LD` splits with hyperparameter from `configs/lit_module/binary` for reproducible comparisons.