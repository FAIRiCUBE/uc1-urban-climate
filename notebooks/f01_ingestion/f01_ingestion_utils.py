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

def list_byoc_collections(list_tiles=False):
    config = SHConfig()
    config.instance_id = os.environ.get("SH_INSTANCE_ID")
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    config.aws_access_key_id = os.environ.get("username")
    config.aws_secret_access_key = os.environ.get("password")

    # Initialize SentinelHubBYOC class
    byoc = SentinelHubBYOC(config=config)
    # list collections and tiles
    # from: https://sentinelhub-py.readthedocs.io/en/latest/examples/byoc_request.html
    collections_iterator = byoc.iter_collections()
    my_collections = list(collections_iterator)

    for collection in my_collections:
        print("Collection name:", collection["name"])
        print("Collection id: ", collection["id"])
        if(list_tiles):
            tiles = list(byoc.iter_tiles(collection))
            for tile in tiles:
                print("Tile status: ", tile['status'])
                print("Tile created: ", tile['created'])
                print("Tile path: ", tile['path'])
                if(tile['status'] == "FAILED"):
                    print("Ingestion failed error: ", tile['additionalData'])
        print("-------------------")
