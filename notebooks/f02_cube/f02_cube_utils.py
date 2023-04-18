import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import rasterio as rio
import IPython.display
from typing import Any, Optional, Tuple

# load utils functions
from f02_cube_utils import *

# Sentinel Hub
from sentinelhub import (
    CRS,
    BBox,
    BBoxSplitter,
    ByocCollection,
    ByocCollectionAdditionalData,
    ByocCollectionBand,
    ByocTile,
    DataCollection,
    DownloadFailedException,
    SentinelHubDownloadClient,
    MimeType,
    SentinelHubBYOC,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
    os_utils,
)

def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

    
def bbox_optimal_subsize(bbox_size):
    if(bbox_size[0] < 2500 and bbox_size[1] < 2500):
        return 1
    else:
        max_size = np.max(bbox_size)
        size = np.max(bbox_size)//2500 + 1
        return size.item()
### reading raster
raster_2band = "./../../../s3/interim/LU001C/LU001C.tiff"
def combine_ua_table_from_city_subcube(raster_2band, output_file, bbox = None, bbox_size = None):
    # if raster_2band is a path (string), load the tiff with rasterio; otherwise raster_2band is a numpy array, and the bbox argument must be passed
    if(type(raster_2band) is str):
        xs, ys, b1, b2 = load_raster(raster_2band)
    else:
        # crs = "EPSG:4326"
        if((bbox == None) or (bbox_size == None)):
            print("Must provide bounding box for numpy arrays")
            exit()
        xs, ys, b1, b2 = load_numpyvector(raster_2band, bbox, bbox_size)
        
    data = {"X": pd.Series(xs.ravel()),
            "Y": pd.Series(ys.ravel()),
            "city_code": pd.Series(b1.ravel()),
            "urban_atlas_2018": pd.Series(b2.ravel())
           }

    df = pd.DataFrame(data=data)

    ### combine:
    combine_table_cube_urban=df.groupby(['city_code','urban_atlas_2018']).size().reset_index().rename(columns={0:'count'})
    combine_table_cube_urban.to_csv(index=False)

    # ouptut_table = "./../../../s3/data/c001_city_cube/tables/urban_cube_v1.csv"
    filepath = Path(output_file)  
    # filepath.parent.mkdir(parents=True, exist_ok=True)  
    combine_table_cube_urban.to_csv(filepath, index=False)  

def load_raster(raster_2band):
    with rio.Env():
        with rio.open(raster_2band) as src:
            crs = src.crs

            # create 1D coordinate arrays (coordinates of the pixel center)
            xmin, ymax = np.around(src.xy(0.00, 0.00), 9)  # src.xy(0, 0)
            xmax, ymin = np.around(src.xy(src.height-1, src.width-1), 9)  # src.xy(src.width-1, src.height-1)
            x = np.linspace(xmin, xmax, src.width)
            y = np.linspace(ymax, ymin, src.height)  # max -> min so coords are top -> bottom


            # create 2D arrays
            xs, ys = np.meshgrid(x, y)
            b1 = src.read(1)
            b2 = src.read(2)

            # Apply NoData mask
            mask = src.read_masks(1) > 0
            xs, ys, b1,b2 = xs[mask], ys[mask], b1[mask],b2[mask]
            return xs, ys, b1, b2

def load_numpyvector(raster_2band, bbox, bbox_size):
#     TODO change projection
    xmin, ymin, xmax, ymax = bbox
    width = bbox_size[0]
    height = bbox_size[1]
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymax, ymin, height)  # max -> min so coords are top -> bottom

    # create 2D arrays
    xs, ys = np.meshgrid(x, y)
    b1 = raster_2band[:,:,0]
    b2 = raster_2band[:,:,1]

    xs, ys, b1,b2 = xs, ys, b1,b2
    return xs, ys, b1, b2

def get_tiff_paths_from_code(URAU_CODE):
    lookuptable_folder = "./../../../s3/data/d000_lookuptables/"
    l_paths = pd.read_csv(lookuptable_folder+"city_UA_subcubes_path.csv")
    return l_paths[l_paths['URAU_CODE'] == URAU_CODE].path