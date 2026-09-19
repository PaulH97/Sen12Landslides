[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper: Nature](https://img.shields.io/badge/Paper-Nature%20Scientific%20Data-blue.svg)](https://www.nature.com/articles/s41597-025-06167-2)
[![Dataset: Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)

# Sen12Landslides: Spatio-Temporal Landslide & Anomaly Detection Dataset

A large-scale, multi-modal, multi-temporal collection of 128×128px Sentinel-1/2 + DEM patches with 10m spatial resolution and with 75k landslide annotations.

## Quick Start
```bash
# Clone & setup
git clone https://github.com/PaulH97/Sen12Landslides.git
cd Sen12Landslides
pip install -e .
pip install --upgrade huggingface_hub

# Authenticate (only once)
hf auth login

# Download harmonized dataset or raw dataset
mkdir -p data
hf download paulhoehn/Sen12Landslides \
  --repo-type dataset \
  --local-dir data \
  --include "data_harmonized/**" 

# Extract and clean up archives
for sensor in s1asc s1dsc s2; do
  for archive in data/data_harmonized/$sensor/*.tar.gz; do
    [ -f "$archive" ] && tar -xzf "$archive" -C "data/data_harmonized/$sensor" && rm "$archive"
  done
done
```

Update the path of `root_dir` in the global configuration file (`Sen12Landslides/configs/config.yaml`) to point to your Sen12Landslides folder, ensuring that it contains the aforementioned data. 

## Dataset Overview

<table>
<tr>
<td>

Full Dataset
| Modality | Positive | Negative | Total  |
|----------|:--------:|:--------:|:------:|
| S1-asc   | 6,492    | 6,814    | 13,306 |
| S1-dsc   | 6,347    | 6,275    | 12,622 |
| S2       | 6,737    | 6,891    | 13,628 |
| Aligned  | 6,026    | 5,693    | 11,719 |

</td>
<td>

Task Splits
| Modality | S12LS-LD | S12LS-AD (pos.) | S12LS-AD (neg.) | S12LS-AD (total) |
|----------|:--------:|:---------------:|:---------------:|:----------------:|
| S1-asc   | 4,793    | 4,793           | 8,344           | 13,137           |
| S1-dsc   | 4,666    | 4,666           | 7,787           | 12,453           |
| S2       | 4,988    | 4,988           | 8,479           | 13,467           |
| Aligned  | 4,392    | 4,392           | 7,172           | 11,564           |

</td>
</tr>
</table>

- S12LS-LD: landslide detection, only annotated (positive) patches
- S12LS-AD: anomaly detection, S12LS-LD's positive patches plus the negative (non-annotated) patches listed above, used to learn normal vs. anomalous patterns. Every S12LS-LD patch keeps its exact train/val/test assignment here, so the two tasks stay directly comparable.
- S12LS-AD-unaligned: superseded original AD release, split independently of S12LS-LD and still including the usa_alaska/nepal inventories. Reference only
- See `Sen12Landslides/tasks/<task>/config.json` for split details

### Reproducing the splits

- `src/data/create_splits.py` builds every task: pick one with `--config-name` (configs in `configs/splits/`)
- `s12ls_ad` must run after `s12ls_ld` for the same dataset version, since it pins shared patches to LD's `splits_aligned.json` via `fixed_splits_json`.

## Dataset Versions

### Harmonized (recommended)

The harmonized version contains radiometrically consistent data that has been pre-processed and bounded for stable model training:

- **Sentinel-1 (Backscatter):**
  - VH and VV bands converted from linear power to decibels (dB) via $10 \cdot \log_{10}(x)$
  - Values bounded to **[-50, 10] dB** to remove extreme noise and specular outliers
- **Sentinel-2 (Reflectance):**
  - Bands B02–B12 corrected for the +1000 DN radiometric offset introduced by ESA Baseline 04.00 (January 25, 2022 onward)
  - Values bounded to **[0, 10000] DN** to ensure physical reflectance consistency
- **DEM (Elevation):**
  - Values bounded to **[0, 8800] m** to maintain a global terrain baseline

### Raw (original)

The raw version preserves the data exactly as published in the original dataset paper, ensuring full reproducibility of reported results:

- **Sentinel-1:** Linear power scale (not converted to dB)
- **Sentinel-2:** No radiometric offset correction applied
- **DEM:** Unmodified

The conversion functions for both corrections are available in the `utils.py` file of the [GitHub repository](https://github.com/PaulH97/Sen12Landslides).

### Data Structure
```
Sen12Landslides/
├── data/
│   ├── data_harmonized/                    ← recommended for training
│   │   ├── inventories.shp.zip
│   │   ├── s1asc/                          Sentinel-1 Ascending (dB)
│   │   │   └── <region>_s1asc_<id>.nc
│   │   ├── s1dsc/                          Sentinel-1 Descending (dB)
│   │   │   └── <region>_s1dsc_<id>.nc
│   │   └── s2/                             Sentinel-2 (offset corrected)
│   │       └── <region>_s2_<id>.nc
│   └── data_raw/                           ← original paper version
│       ├── inventories.shp.zip
│       ├── s1asc/
│       ├── s1dsc/
│       └── s2/
├── tasks/
│   ├── S12LS-LD/                           Landslide detection
│   │   ├── config.json
│   │   ├── harmonized/
│   │   │   └── <modality>/
│   │   │       ├── splits.json
│   │   │       ├── norm.json
│   │   │       └── patch_locations.geojson
│   │   └── raw/
│   │       └── <modality>/
│   │           ├── splits.json
│   │           ├── norm.json
│   │           └── patch_locations.geojson
│   ├── S12LS-AD/                           Anomaly detection (splits pinned to S12LS-LD)
│   │   ├── config.json
│   │   ├── harmonized/
│   │   │   ├── <modality>/
│   │   │   │   ├── splits.json             per-satellite split
│   │   │   │   ├── norm.json
│   │   │   │   └── patch_locations.geojson
│   │   │   ├── splits_aligned.json         strict split (all satellites present)
│   │   │   ├── norm_aligned.json
│   │   │   └── patch_locations_aligned.geojson
│   │   └── raw/
│   │       └── ... (same layout)
│   └── S12LS-AD-unaligned/                 superseded AD release, reference only
│       ├── harmonized/
│       │   └── ... (same layout as S12LS-AD)
│       └── raw/
│           └── ... (same layout)
└── src/                                    Data loaders, models, training
```

### Patch Format

Each `.nc` file contains 128×128 px across 15 time steps:

| Modality | Bands | Additional |
|----------|-------|------------|
| Sentinel-1-NRB | VV, VH | DEM, MASK |
| Sentinel-2-L2A | B02-B08, B8A, B11-B12 | SCL, DEM, MASK |

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

We provide two task-specific configurations:

Creating custom splits:
```bash
python src/data/create_splits.py  # Configure in configs/splits/config.yaml
```

#### Always Generated (Root Level)
| File | Description |
|------|-------------|
| config.json | Filter criteria, split ratios, and stratification settings |

#### Per-Satellite Folders (`s1asc/`, `s1dsc/`, `s2/`)
| File | Description |
|------|-------------|
| splits.json | Train/val/test splits for this satellite modality |
| norm.json | Per-band normalization statistics (mean/std) for this satellite |
| patch_locations.geojson | Geographic patch locations with train/val/test assignments for this satellite |

#### Multi-Modal Files
| File | Description |
|------|-------------|
| splits_aligned.json | Train/val/test splits containing only patches available across all satellites |
| norm_aligned.json | Normalization statistics computed from aligned patches only |
| patch_locations_aligned.geojson | Geographic locations of patches available across all satellites |

**Usage:**
- Single-modal training: Load `<satellite>/splits.json` + `<satellite>/norm.json`
- Multi-modal training: Load `splits_aligned.json` + `norm_aligned.json` for cross-modal fusion
- Visualization: Open `patch_locations.geojson` in QGIS or mapping tools


## Training

This project uses [Hydra](https://hydra.cc/) for configuration management. See [Hydra documentation](https://hydra.cc/docs/intro/) for more details. Note that the standard parameters of some classes are overwritten by those in the configuration files. Therefore, ensure that you always update the config files under `configs/` accordingly for your hardware and requirements. 


### Available Configurations

| Config | Options |
|--------|---------|
| model | utae, convgru, unet3d, unet_convlstm |
| dataset | sen12ls_s2, sen12ls_s1asc, sen12ls_s1dsc |
| trainer | cpu, gpu, ddp |
| lit_module | binary, multiclass |


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

Due to class imbalance (~3% landslides), we provide, additionaly to our macro-avg metrics in the paper, **binary metrics** on the landslide class for benchmarking against other detection methods. 

> **Note:** To compare landslide detection performance, use the binary metrics below rather than the macro-averaged metrics from the paper.

### Benchmark Results (`S12LS-LD`)

Benchmark using paper architectures with **binary metrics** on **Sentinel-2 + DEM**:

| Model      | AP    | F1    | IoU   | Precision | Recall |
| ---------- | ----- | ----- | ----- | --------- | ------ |
| U-ConvLSTM | 65,13 | 61,95 | 44,88 | 60,59     | 63,92  |
| Unet3d     | 62,08 | 58,82 | 41,66 | 55,75     | 62,56  |
| ConvGRU    | 60,00 | 59,06 | 41,91 | 56,72     | 61,77  |
| U-TAE      | 67,75 | 61,80 | 44,74 | 53,19     | 74,90  |


Three training runs (`seed=42,123,777`) were performed for each model on the harmonized `S12LS-LD` split with `lit_module=binary` for 75 epochs. **Test metrics were averaged across seeds on the held-out test set.** See `configs/` for full settings.

> ⚠️ These results are a proof of concept to demonstrate the dataset's utility, not an upper bound on achievable performance. Both the model architecture and the random train/val/test splits leave substantial room for improvement.

### Reproducibility
```bash
# Train all baselines
python pipeline/train.py --multirun \
  model=unet3d,convgru,utae,unet_convlstm \
  seed=42,123,777 \
  dataset=sen12ls_s2
```

## Challenges

What makes this dataset demanding and a good resource for new methodological improvements to beat the baselines:

1. **Severe class imbalance** (~3% landslides)
2. **Small spatial extent** - landslides often span few pixels at 10m resolution
3. **Multi-temporal complexity** - effective temporal fusion remains challenging
4. **Geographic diversity** - varied terrain, vegetation, and climate
