# Sen12Landslides: A spatio-temporal Dataset for satellite-based Landslide and Anomaly Detection
This is the official repository for **Sen12Landslides**, a large-scale, multi-modal, and multi-temporal dataset for satellite-based landslide detection and spatio-temporal anomaly analysis. It includes over 100,000 annotated landslides and 12,000+ Sentinel-1, Sentinel-2, and DEM image patches (128x128) with precise pre- and post-event timestamps.
(Paper-Link)

The dataset can be downloaded under: https://huggingface.co/datasets/paulhoehn/Sen12Landslides

## Data Structure

The dataset is organized into three main folders based on the satellite data source:

```
Sen12Landslides/
├── data/                           # Raw NetCDF patches (compressed)                
│   ├── s1asc/   
│   ├── s1dsc/  
│   └── s2/    
│       └── part*.tar.gz         
│
├── tasks/                       
│   ├── S12LS-LD/                   # Landslide detection
│   │   ├── config.json          
│   │   └── {s1asc,s1dsc,s2}/     
│   │       ├── data_paths.json  
│   │       └── norm_data.json   
│   │
│   └── S12LS-AD/                   # Anomaly detection
│       ├── config.json          
│       └── {s1asc,s1dsc,s2}/     
│           ├── data_paths.json  
│           └── norm_data.json   
│
└── inventories.shp.zip             # Landslide polygons
```

Each ```tar``` folder contains multiple `.nc` files (NetCDF format), where each file corresponds to a specific geographic region and patch. The filenames follow the structure:

```
<region>_<satellite>_<id>.nc
```

For example:

```
data/s1asc/italy_s1asc_6982.nc
data/s1dsc/italy_s1dsc_6982.nc
data/s2/italy_s2_6982.nc
```

Each file includes multi-temporal image patches along with pixel-level annotations and relevant metadata.
The output of such a file looks like the following after calling: ```xr.open_dataset("Sen12Landslides/data/s2/italy_s2_6982.nc")```
```
<xarray.Dataset> Size: 6MB
Dimensions:      (time: 15, x: 128, y: 128)
Coordinates:
  * x            (x) float64 1kB 7.552e+05 7.552e+05 ... 7.565e+05 7.565e+05
  * y            (y) float64 1kB 4.882e+06 4.882e+06 ... 4.881e+06 4.881e+06
  * time         (time) datetime64[ns] 120B 2022-10-05 2022-10-30 ... 2023-09-10
Data variables: (12/14)
    B02          (time, x, y) int16 492kB ...
    B03          (time, x, y) int16 492kB ...
    B04          (time, x, y) int16 492kB ...
    B05          (time, x, y) int16 492kB ...
    B06          (time, x, y) int16 492kB ...
    B07          (time, x, y) int16 492kB ...
    ...           ...
    B11          (time, x, y) int16 492kB ...
    B12          (time, x, y) int16 492kB ...
    SCL          (time, x, y) int16 492kB ...
    MASK         (time, x, y) uint8 246kB ...
    DEM          (time, x, y) int16 492kB ...
    spatial_ref  int64 8B ...
Attributes:
    ann_id:           41125,41124,37694,37696,41131,37693,37689,37695,37749,3...
    ann_bbox:         (755867.5791119931, 4880640.0, 755900.7341873142, 48806...
    event_date:       2023-05-16
    date_confidence:  1.0
    pre_post_dates:   {'pre': 7, 'post': 8}
    annotated:        True
    satellite:        s2
    center_lat:       4881280.0
    center_lon:       755840.0
    crs:              EPSG:32632
```
For the corresponding Sentinel-1 data, the overall structure remains the same, but the data variables are adapted to SAR input, containing `VV` and `VH` bands instead of optical bands. The metadata attributes are consistent across modalities, with the only change being the `satellite` attribute set to `"s1"` instead of `"s2"`.


