# Sen12Landslides: Spatio-Temporal Landslide & Anomaly Dataset

A large-scale, multi-modal, multi-temporal collection of 128×128 Sentinel-1/2 + DEM patches with 75k landslide annotations.

**Paper (coming soon) & dataset:**
🔗 [https://huggingface.co/datasets/paulhoehn/Sen12Landslides](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)

---

## 1. Setup

```bash
# 1. Clone code repo
git clone https://github.com/your-org/Sen12Landslides.git
cd Sen12Landslides

# 2. Install HF CLI
pip install --upgrade huggingface_hub

# 3. Authenticate (only first time)
huggingface-cli login  # paste your token from https://huggingface.co/settings/tokens

# 4. Pull the dataset into `data/`
huggingface-cli repo clone paulhoehn/Sen12Landslides --repo-type dataset data
```

After cloning, you’ll have:

```
Sen12Landslides/
├── data/
│   ├── s1asc/             # s1asc_part01…s1asc_part13.tar.gz
│   ├── s1dsc/             # s1dsc_part01…s1dsc_part12.tar.gz
│   ├── s2/                # s2_part01…s2_part28.tar.gz
│   └── inventories.shp.zip
├── src/                   # code
├── tasks/
├── ...            
└── README.md
```

---

## 2. Extract

Unpack all `.nc` patches so your loader can read them directly:

```bash
# From repo root:
for sensor in s1asc s1dsc s2; do
  tar -xzvf data/$sensor/*.tar.gz -C data/$sensor
done
```

---

## 3. Data Layout

```
Sen12Landslides/
├── data/
│   ├── s1asc/
│   │   ├── italy_s1asc_6982.nc
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
│   ├── S12LS-AD/                      # Anomaly detection
│   │   ├── config.json
│   │   ├── s1asc/
│   │   ├── s1dsc/
│   │   └── s2/
│   │       ├── data_paths.json
│   │       └── norm_data.json
│   └── S12LS-LD/                      # Landslide detection
│       ├── config.json
│       ├── s1asc/
│       ├── s1dsc/
│       └── s2/
│           ├── data_paths.json
│           └── norm_data.json
├── src/                               # Data loaders, models, etc.
├── ...
└── README.md
```

* **`inventories.shp.zip`**
  A zipped shapefile containing all ground-truth landslide polygons. Each polygon corresponds to one mapped landslide and is spatially aligned with the image patches.

* **NetCDF patches** (`.nc` files)
  These files live in the `s1asc/`, `s1dsc/`, and `s2/` directories. Each patch is:

  * **Size:** 128 × 128 pixels
  * **Time series length:** 15 observations
  * **Contents:**
    * One or more spectral or SAR bands
    * A binary mask (`MASK`) marking landslide pixels
    * Event metadata (`event_date`, `pre_post_dates`, etc.)
    * **DEM** (digital elevation model)
    * **SCL** (Scene Classification Layer, Sentinel-2 only)

Filenames always follow:

```
<region>_<sensor>_<id>.nc ->  italy_s2_6982.nc
```

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

You’re now ready to build custom splits under `tasks/`, train models, and integrate into your pipeline.
