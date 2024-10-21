# get water features from CLC 2018 within a buffer of 100m around core city
# uses SentinelHub statistical API

import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import IPython.display
from time import process_time
# load utils functions
from src import utils
from datetime import datetime
from src import measurer, utils, db_connect
from types import ModuleType
from loguru import logger
from scipy import ndimage
import rasterio
import rasterio.features
from sqlalchemy import text

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
    SentinelHubStatistical,
    Geometry
)


@logger.catch()
def buffer_bounds(row, buffer_multiplier, crs, resolution):
    bbox_coords = tuple(row.geometry.bounds)
    bbox = BBox(bbox=bbox_coords, crs=crs)
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    bbox_size
    # Compute the parameters of the georeference
    # pixel size in the x-direction in map units/pixel
    dirx = (bbox_coords[2] - bbox_coords[0]) / bbox_size[1]
    # pixel size in the y-direction in map units, almost always negative
    diry = -(bbox_coords[3] - bbox_coords[1]) / bbox_size[0]
    bbox_coords_b = [bbox_coords[0]-dirx,
                     bbox_coords[1]+buffer_multiplier*diry,
                     bbox_coords[2]+dirx,
                     bbox_coords[3]-buffer_multiplier*diry]
    bbox_b = BBox(bbox=bbox_coords_b, crs=crs)
    bbox_size_b = bbox_to_dimensions(bbox_b, resolution=resolution)
    bbox_size_b
    return bbox_b, bbox_size_b


@logger.catch()
def polygon_to_mask(coords, geometry, crs, resolution):
    bbox = BBox(bbox=coords, crs=crs)
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    # Compute the parameters of the georeference
    # pixel size in the x-direction in map units/pixel
    dirx = (coords[2] - coords[0]) / bbox_size[1]
    # pixel size in the y-direction in map units, almost always negative
    diry = -(coords[3] - coords[1]) / bbox_size[0]
    x0 = coords[0]  # x-coordinate of the center of the upper left pixel
    y0 = coords[3]  # y-coordinate of the center of the upper left pixel
    transform = rasterio.Affine(dirx, 0, x0,
                                0, diry, y0)
    rasterized = rasterio.features.rasterize(
        [geometry],
        out_shape=[bbox_size[1],bbox_size[0]],
        fill=0,
        default_value=1,
        transform=transform,
        all_touched=True,
        dtype='float64'
    )
    return rasterized


def avg_distance_to_water(data_array, mask_array):
    res = ndimage.distance_transform_edt(data_array)
    return res, np.mean(res, where=mask_array.astype(bool))


@logger.catch()
def sentinelhub_request(bbox, bbox_size, config):
    evalscript = """
    //VERSION=3
    function setup() {
    return {
        input: ["CLC", "dataMask"],
        output: {
        bands: 1,
        sampleType: "UINT16"
        }
    }
    }

    function evaluatePixel(sample) {
        if(sample.CLC < 45 && sample.CLC > 39) {
            return [sample.CLC*0];
        } else {
            return [sample.CLC/sample.CLC]
        }
    }

    """
    # copy CollectionId from FAIRiCube catalog https://catalog.fairicube.eu/
    collection_id_clc = "cbdba844-f86d-41dc-95ad-b3f7f12535e9"
    collection_name_clc="CLC"
    end_point = "https://creodias.sentinel-hub.com"
    # define collection
    data_collection_clc = DataCollection.define_byoc(collection_id_clc, name=collection_name_clc, service_url = end_point)
    input_data = [
        SentinelHubRequest.input_data(
            data_collection=DataCollection.CLC,
            time_interval=("2017-01-01", "2019-01-01")  # select only CLC 2018
        ),
    ]
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=input_data,
        responses=[SentinelHubRequest.output_response(
            "default", MimeType.PNG)],
        bbox=bbox,
        size=bbox_size,
        config=config,
    )
    return request

@logger.catch()
def process_row(row, resolution=100):
    config = SHConfig()
    config.instance_id = os.environ.get("SH_INSTANCE_ID")
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    config.aws_access_key_id = os.environ.get("username")
    config.aws_secret_access_key = os.environ.get("password")
    logger.info(f"Downloading {row.urau_code} {row.urau_name} data")

    # ------------------------------------------
    bbox, bbox_size = buffer_bounds(row, 2, CRS.WGS84, resolution)
    coords = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
    mask_array = polygon_to_mask(
        coords, row.geometry, CRS.WGS84, resolution)
    request = sentinelhub_request(bbox, bbox_size, config)

    bbox_subsize = utils.bbox_optimal_subsize(bbox_size)
    if (bbox_subsize == 1):
        request = sentinelhub_request(bbox, bbox_size, config)
        # try:
        data = request.get_data()[0]
        _, avg = avg_distance_to_water(data, mask_array)
        # df_data.append(                )
        print(avg)
        return avg
        # except:
        #     logger.info("an error occurred")
        #     # list_error.append(row.urau_code)
        #     print(row.urau_code)
        #     return row.urau_code
    else:
        logger.info(
            f"Splitting bounding box in {(bbox_subsize,bbox_subsize)} subgrid")
        bbox_split = BBoxSplitter([bbox], CRS.WGS84, bbox_subsize)
        # create a list of requests
        bbox_list = bbox_split.get_bbox_list()
        print(bbox_list)
        return row.urau_code

def process_row2(row, resolution=100):
    return resolution*row.geometry.bounds
@logger.catch
def subcube():

    # Path where the data are stored (the use of the disk in this path is measured).
    data_path = '/'
    logger.add(f"logfile_distance_to_water.log")
    measurer_instance = measurer.Measurer()
    tracker = measurer_instance.start(data_path=data_path)
    logger.info("Start")
    # load city polygons
    home_dir = os.environ.get('HOME')

    engine_postgresql = db_connect.create_engine(
        db_config=f"{home_dir}/uc1-urban-climate/database.ini")

    with engine_postgresql.begin() as conn:
        query = text("""
                SELECT urau_code, urau_name, geometry
                FROM lut.l_city_urau2021
                """)
        gdf = gpd.read_postgis(query, conn, geom_col='geometry')

    resolution = 100
    gdf['avg_distance'] = gdf.apply(lambda row: process_row(row, resolution), axis=1)
    measurer_instance.end(tracker=tracker,
                          shape=[],
                          libraries=[v.__name__ for k, v in globals().items() if type(
                              v) is ModuleType and not k.startswith('__')],
                          data_path=data_path,
                          program_path=__file__,
                          csv_file=f'{home_dir}/s3/data/l001_logs/benchmarks_stats_distance_to_water.csv')
    return gdf


if __name__ == "__main__":
    home_dir = os.environ.get('HOME')
    df = subcube()
    df[['urau_code', 'avg_distance']].to_csv(
       f"{home_dir}/s3/data/c001_city_cube/distance_to_water.csv", mode='a')
