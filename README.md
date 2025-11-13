# Sen12Landslides: Spatio-Temporal Landslide & Anomaly Dataset

A large-scale, multi-modal, multi-temporal collection of 128×128px Sentinel-1/2 + DEM patches with 10m spatial resolution and with 75k landslide annotations.

**Paper (coming soon) & dataset:**
🔗 [https://huggingface.co/datasets/paulhoehn/Sen12Landslides](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)

---

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
│   │   ├── italy_s1asc_6982.nc             # <region>_<sensor>_<patch_id>.nc
│   │   ├── chimanimani_s1asc_1024.nc
│   │   └── ...
│   ├── s1dsc/
│   │   ├── italy_s1dsc_6982.nc
│   │   ├── chimanimani_s1dsc_1024.nc
│   │   └── ...
│   └── s2/
│       ├── italy_s2_6982.nc
│       ├── chimanimani_s2_1024.nc
│       └── ...
├── tasks/
│   ├── S12LS-AD/                           # Anomaly detection task configuration
│   │   ├── config.json                     # Task-level metadata
│   │   ├── patch_locations_s1asc.geojson   # Independent per modality
│   │   ├── patch_locations_s1dsc.geojson   
│   │   ├── patch_locations_s2.geojson       
│   │   ├── s1asc/
│   │   │   ├── data_paths.json
│   │   │   └── norm_data.json
│   │   ├── s1dsc/
│   │   │   ├── data_paths.json
│   │   │   └── norm_data.json
│   │   └── s2/
│   │       ├── data_paths.json
│   │       └── norm_data.json
│   └── S12LS-LD/                           # Landslide detection task configuration
│       ├── config.json
│       ├── patch_locations.geojson         # Aligned across all modalities
│       ├── s1asc/
│       │   ├── data_paths.json
│       │   └── norm_data.json
│       ├── s1dsc/
│       │   ├── data_paths.json
│       │   └── norm_data.json
│       └── s2/
│           ├── data_paths.json
│           └── norm_data.json
├── src/                                    # Source code: data loaders, model definitions, training scripts
├── ...
└── README.md
```

### Folder Descriptions

* **`data/inventories.shp.zip`**
  A zipped shapefile containing all ground-truth landslide polygons. Each polygon corresponds to one mapped landslide and is spatially aligned with the image patches.

* **NetCDF patches (`.nc` files)**
  Contained in `s1asc/`, `s1dsc/`, and `s2/`. Each file represents a 128×128 patch with 15 time steps and includes:

  * Sentinel-2: 10 bands (B02–B12), SCL, DEM, MASK, metadata
  * Sentinel-1: 2 bands (VV, VH), DEM, MASK, metadata

* **`tasks/`**
  Contains task-specific configuration for anomaly detection (`S12LS-AD`) and landslide detection (`S12LS-LD`). Each task includes:
  
  * **`config.json`** - Task configuration (filters, split ratios, stratification method)
  * **`patch_locations.geojson`** - Patch locations with split assignments (train/val/test) for visualization
  * **Per-satellite folders** (`s1asc/`, `s1dsc/`, `s2/`) containing:
    * `data_paths.json` - File paths for train/val/test splits
    * `norm_data.json` - Mean/std statistics for normalization
* **`src/`**
  Contains the codebase used to process, train, and evaluate models on the dataset. This includes:

  * Dataset loaders
  * Model architectures
  * Training and evaluation pipelines
  * Utility functions for pre or post-processing

### Data Record 

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

---

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

---

## Evaluation Metrics
Sen12Landslides is characterized by severe class imbalance (~2% landslides). For a meaningful evaluation, we strongly recommend using metrics that focus on the positive (landslide) class:
- Average Precision (AP)
- Area Under the ROC Curve (AUROC)
- F1-Score, Precision, and Recall for the landslide class

_Note: This guidance differs from the macro-averaged metrics used in our paper. While macro-averaging was suitable for our paper's broad technical validation, the metrics recommended here provide a more direct and practical assessment of a model's ability to detect the rare landslide class._

---

**Coming soon:** Updated benchmark table with the recommended extensive metrics on the `S12LS-LD` for better benchmarking and comparison.
