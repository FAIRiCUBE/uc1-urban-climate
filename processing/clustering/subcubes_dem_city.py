# get elevation statistics (min, max, mean, std) within city from Copernicus DEM 30m
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
from src.measurer import Measurer
from types import ModuleType
from loguru import logger

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
def sentinelhub_stat_request(evalscript, geometry, bbox, bbox_size, config):
    calculations = {
        "default": {
            "histograms": {
                "default": {
                    "binWidth": "10",
                    # "lowEdge": "0",
                    # "highEdge": "101" #histogram interval is [lowEdge, highEdge) that is, highEdge value excluded
                }
            }
        }
    }
    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=("2018-01-01", "2019-05-01"),
            aggregation_interval="P1D",
            size=bbox_size
        ),
        # input_data=[SentinelHubStatistical.input_data(DataCollection.define_byoc('3947b646-383c-4e91-aade-2f039bd6ba4b', name=f'{dim_name}Density2018'))],
        input_data=[SentinelHubStatistical.input_data(DataCollection.DEM_COPERNICUS_30)],
        bbox=bbox,
        geometry = geometry,
        calculations=calculations,
        config=config,
    )
    return request

@logger.catch()
def sentinelhub_request(evalscript, geometry, bbox, bbox_size, config):
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.DEM_COPERNICUS_30,
            # time_interval=time_interval # DEM is static data, does not depend on date
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_size,
        geometry = geometry,
        config=config,
        )
    return request

@logger.catch
def subcube(dim_name):
    
    config = SHConfig()
    config.instance_id = os.environ.get("SH_INSTANCE_ID")
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    config.aws_access_key_id = os.environ.get("username")
    config.aws_secret_access_key = os.environ.get("password")

    # Path where the data are stored (the use of the disk in this path is measured).
    data_path = '/'
    logger.add(f"./../../s3/data/l001_logs/logfile_{dim_name}.log")
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    logger.info("Start")
    # load city polygons
    city_polygons = "./../../s3/data/d001_administration/urban_audit_city_2021/URAU_RG_100K_2021_3035_CITIES/URAU_RG_100K_2021_3035_CITIES.shp"
    geo_json_city = gpd.read_file(city_polygons)
    gdf_city = gpd.GeoDataFrame(geo_json_city, crs="EPSG:3035")
    # gdf_city = gdf_city[gdf_city.URAU_CODE.isin(['BG016C','HR007C','IT003C','BG017C','DE002C',
    #                                     'PT001C','ES088C','ES010C','ES069C','FR072C','FI004C',
    #                                     'SE008C','ES045C','ES039C','NL037C','PT004C'])]
    # define evalscript
    evalscript = """
    //VERSION=3

    function setup() {
    return {
        input: ["DEM"],
        output:[{ 
            id: "default",
            bands: 1, 
            sampleType: SampleType.INT16
        },
        {
            id: "dataMask",
            bands: 1
        }]
    }
    }

    function evaluatePixel(sample) {
    return {
        default: [sample.DEM+1.0], // shift by one to distinguish real 0 from nodata 0
        dataMask: [sample.dataMask]}
    }
    """
    
    # create temporary df
    df_all = pd.DataFrame(columns=['URAU_CODE', 'dem_min', 'dem_max', 'dem_mean', 'dem_std', 'dem_count'])
    for row in gdf_city.itertuples():

        logger.info(f"Downloading {row.URAU_CODE} {row.URAU_NAME}")
        
        #------------------------------------------
        geometry_gdf = row.geometry
        geometry_b, bbox_b, bbox_size_b = utils.buffer_geometry(geometry_gdf, buffer_size=0)

        bbox_subsize_b = utils.bbox_optimal_subsize(bbox_size_b)
        if(bbox_subsize_b == 1 ):
            request = sentinelhub_request(evalscript, geometry_b, bbox_b, bbox_size_b, config)
            try:
                data = request.get_data()[0]
                # set nodata to numpy nan
                data = data.astype("float")
                data[data == 0] = np.nan
                # shift back data to original value (see evalscript)
                data = data - 1
                if(np.nanstd(data) > 0):
                    df = pd.DataFrame(data = {
                        'URAU_CODE':    [row.URAU_CODE],
                        'dem_min':      [np.nanmin(data)],
                        'dem_max':      [np.nanmax(data)],
                        'dem_mean':     [np.nanmean(data)],
                        'dem_std':      [np.nanstd(data)],
                        'dem_count':    [len(data)]
                    })
            except:
                logger.info("an error occurred")
                print(row.URAU_CODE)
                break
            df_all = pd.concat([df_all, df])
            # break
        else:
            logger.info(f"Splitting bounding box in {(bbox_subsize_b,bbox_subsize_b)} subgrid")
            bbox_split = BBoxSplitter([geometry_b], CRS('3035').pyproj_crs(), bbox_subsize_b, reduce_bbox_sizes=True)
            # create a list of requests
            bbox_list = bbox_split.get_bbox_list()
            geometry_list = [Geometry(geometry=utils.split_geometry(geometry_b, bbox), crs=CRS('3035').pyproj_crs()) for bbox in bbox_list]
            sh_requests = [sentinelhub_request(evalscript, geometry, subbbox, bbox_to_dimensions(subbbox, resolution=10), config) for (geometry,subbbox) in list(zip(geometry_list,bbox_list))]
            error=False
            data_tmp = np.array([])
            for idx, req in enumerate(sh_requests):
                try:
                    data = req.get_data()[0]
                    # set nodata to numpy nan
                    data = data.astype("float")
                    data[data == 0] = np.nan
                    # shift back data to original value (see evalscript)
                    data = data - 1
                    if(np.nanstd(data) > 0):
                        data_tmp = np.concatenate((data_tmp,data.ravel()))
                        # print(data)
                        # print(data_tmp)
                        logger.info(f"Processing subbox no.{idx}")
                except:
                    logger.info("an error occurred")
                    print(row.URAU_CODE)
                    error=True
                    break
            if(~error and len(data_tmp)>0):
                logger.info("Concatenating results")
                df = pd.DataFrame(data = {
                        'URAU_CODE':    [row.URAU_CODE],
                        'dem_min':      [np.nanmin(data_tmp)],
                        'dem_max':      [np.nanmax(data_tmp)],
                        'dem_mean':     [np.nanmean(data_tmp)],
                        'dem_std':      [np.nanstd(data_tmp)],
                        'dem_count':    [len(data_tmp)]
                    })
                df_all = pd.concat([df_all, df])
        print(df_all[df_all.URAU_CODE == row.URAU_CODE])
    measurer.end(tracker=tracker,
                 shape=[],
                 libraries=[v.__name__ for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
                 data_path=data_path,
                 program_path=__file__,
                 csv_file=f'./../../s3/data/l001_logs/benchmarks_stats_{dim_name}_v3.csv')
    return df_all
    

if __name__ == "__main__":
    dim_name = 'DEM_COPERNICUS_30'
    df = subcube(dim_name)
    df.to_csv(f"./../../s3/data/c001_city_cube/{dim_name}_v3.csv", mode='a')