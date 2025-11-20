# Sen12Landslides: Spatio-Temporal Landslide & Anomaly Dataset

A large-scale, multi-modal, multi-temporal collection of 128×128px Sentinel-1/2 + DEM patches with 10m spatial resolution and with 75k landslide annotations.

**Paper**:
https://www.nature.com/articles/s41597-025-06167-2

**Dataset**: 
🔗 [https://huggingface.co/datasets/paulhoehn/Sen12Landslides](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)

| Modailty | Samples | Annotated  | Non-Annotated |
|----------|:-------:|:----------:|:-------------:|
| Sentinel-1-asc | 13306 | 6494 | 6812 |
| Sentinel-1-dsc | 12622 |6349 | 6273 |
| Sentinel-2 | 13628 | 6739 | 6889 |


## Setup

```bash
# 1. Clone code repo
git clone https://github.com/PaulH97/Sen12Landslides.git
cd Sen12Landslides

# 2. Install HF CLI
pip install --upgrade huggingface_hub

# 3. Authenticate (only first time)
huggingface-cli login  # paste your token from https://huggingface.co/settings/tokens

# 4. Pull the dataset into `data/`
mkdir data
huggingface-cli download paulhoehn/Sen12Landslides --repo-type dataset --local-dir data
```

Unpack all `.nc` patches so the custom loader can read them directly:

```bash
# From repo root:
for sensor in s1asc s1dsc s2; do
  for archive in data/$sensor/*.tar.gz; do
    tar -xzvf "$archive" -C "data/$sensor" && rm "$archive" # remove compressed tar files after extraction 
  done
done
```

---

## Data Structure

```
Sen12Landslides/
├── data/
│   ├── ...
│   ├── inventories.shp.zip
│   ├── s1asc/
│   │   ├── italy_s1asc_6982.nc                # <region>_<sensor>_<patch_id>.nc
│   │   ├── chimanimani_s1asc_1024.nc
│   │   └── ...
│   ├── s1dsc/
│   │   └── ...
│   └── s2/
│       └── ...
├── tasks/
│   ├── S12LS-AD/                              # Anomaly detection task configuration
│   │   ├── config.json                        
│   │   ├── norm_aligned.json                  
│   │   ├── patch_locations_aligned.geojson    
│   │   ├── splits_aligned.json                       
│   │   ├── s1asc/
│   │   │   ├── norm_data.json                
│   │   │   ├── patch_locations.geojson        
│   │   │   └── splits.json                    
│   │   ├── s1dsc/
│   │   │   └── ...           
│   │   └── s2/
│   │       └── ...         
│   └── S12LS-LD/                              # Landslide detection task configuration
│       ├── config.json
│       ├── norm_aligned.json              
│       ├── patch_locations_aligned.geojson            
│       ├── splits_aligned.json            
│       ├── s1asc/
│       │   ├── norm.json
│       │   ├── patch_locations.geojson
│       │   └── splits.json              
│       ├── s1dsc/
│       │   └── ...          
│       └── s2/
│           └── ...            
├── src/                                       # data loaders, model definitions, ...
├── ...
└── README.md
```

### Core Data Files (`data/`)

**`inventories.shp.zip`**:
Zipped shapefile containing ground-truth landslide polygons. Each polygon represents one mapped landslide event, spatially aligned with image patches.

**Satellite Image Patches** (`s1asc/`, `s1dsc/`, `s2/`)
NetCDF files (`.nc`) with 128×128 pixel patches across 15 time steps:
- Sentinel-1: 2 polarizations (VV, VH), DEM, landslide mask (MASK), and metadata
- Sentinel-2: 10 spectral bands (B02–B08, B8A, B11–B12), Scene Classification Layer (SCL), DEM, landslide mask (MASK), and metadata


### Task Configurations (`tasks/`)

Some patches are challenging even for human experts (e.g., <10 annotated pixels, ambiguous temporal signatures, missing/noisy labels). We provide two task-specific configurations:

- **S12LS-LD**: Landslide detection with ~3,500-4,000 high-quality annotated patches (manual verification)
- **S12LS-AD**: Anomaly detection with mixed annotated/non-annotated samples to learn normal vs. anomalous patterns

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
- Single-modal: Load `<satellite>/data_paths.json` + `<satellite>/norm_data.json`
- Multi-modal: Load `splits.json` + `norm.json` for cross-modal fusion
- Visualization: Open `patch_locations.geojson` in QGIS or mapping tools

### Source Code (`src/`)

Processing, training, and evaluation codebase:
- Dataset loaders for NetCDF patches
- Model architectures (CNNs, transformers, fusion networks)
- Training pipelines with experiment tracking
- Evaluation scripts for metrics and visualization
- Preprocessing utilities for data augmentation and normalization

## Data Record 

Opening a patch with xarray reveals its structure:

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

Sentinel-1 patches are structured the same way but include SAR bands (`VV`, `VH`) instead of optical ones, and set `satellite="s1"` in the attributes.

## Creating Custom Splits

Create train/val/test stratified splits with multi-modal alignment:
```bash
python src/data/create_splits.py
```

**Key settings** in `configs/config.yaml`:
- `align_modalities: true` - Ensure all patches have all satellites (required for fusion)
- `test_size: 0.2` / `val_size: 0.2` - Split ratios
- `filter_criteria.annotated_only: true` - Only annotated patches

Output includes `data_paths.json` (file lists), `norm_data.json` (statistics), and `patch_locations.geojson` (visualization).


## Evaluation Metrics
Sen12Landslides is characterized by severe class imbalance (~3% landslides). For a meaningful evaluation, we strongly recommend using metrics that focus on the positive (landslide) class.

_Note: _Note: This guidance differs from the macro-averaged metrics used in our paper. While macro-averaging was suitable for our paper's broad technical validation, we want to clarify that for practical landslide detection applications, metrics focusing on the positive class provide more actionable insights. Users comparing their results to our published paper should be aware of this distinction.__


## Performance Baselines & Dataset Challenges

### Baseline Results
Please use the provided `S12LS-LD` splits for landslide detection tasks and the `S12LS-AD` for anomaly detection tasks.
Updated `S12LS-LD` with better geographical split and higher quality is in lrogress and uploaded tomorrow.
Coming soon: Updated benchmark table with the extended metrics on the new `S12LS-LD`.

### Why Landslide Detection is Challenging in Sen12Landslides

This dataset presents several characteristics that make it a **demanding benchmark** for landslide detection methods:

1. **Severe class imbalance** (~3% landslides) - requires methods robust to imbalanced learning
2. **Small spatial extent** - Many landslides span only a few pixels at 10m resolution
3. **Multi-temporal complexity** - Leveraging temporal information effectively remains an open challenge
4. **Multi-modal fusion** - Optimal integration of Sentinel-1 (SAR) and Sentinel-2 (optical) is non-trivial
5. **Geographic diversity** - Landslides occur across varied terrain, vegetation, and climate conditions

These **challenges** make Sen12Landslides an excellent benchmark for:
- Novel architectures for imbalanced, multi-modal learning
- Temporal fusion strategies
- Small object detection in remote sensing
- Transfer learning and domain adaptation approaches

A low baseline performance on the landslide class reflects the genuine difficulty of this real-world problem and presents **significant opportunity for methodological innovation**. We encourage researchers to tackle these challenges and contribute improved methods.
