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
def sentinelhub_stat_request( evalscript, geometry, bbox, bbox_size, config):
    collection_id_clc = "cbdba844-f86d-41dc-95ad-b3f7f12535e9"
    collection_name_clc="CLC"
    end_point = "https://creodias.sentinel-hub.com"
    calculations = {
        "default": {
            "histograms": {
                "default": {
                    "binWidth": "1",
                    "lowEdge": "40",
                    "highEdge": "45" #histogram interval is [lowEdge, highEdge) that is, highEdge value excluded
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
        input_data=[SentinelHubStatistical.input_data(
            DataCollection.define_byoc(
                collection_id_clc, 
                name=collection_name_clc, 
                service_url = end_point))],
        bbox=bbox,
        geometry = geometry,
        calculations=calculations,
        config=config,
    )
    return request


@logger.catch
def subcube():
    
    config = SHConfig()
    config.instance_id = os.environ.get("SH_INSTANCE_ID")
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    config.aws_access_key_id = os.environ.get("username")
    config.aws_secret_access_key = os.environ.get("password")

    # Path where the data are stored (the use of the disk in this path is measured).
    data_path = '/'
    logger.add(f"logfile_CLC.log")
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    logger.info("Start")
    # load city polygons
    city_polygons = "./../../s3/data/d001_administration/urban_audit_city_2021/URAU_RG_100K_2021_3035_CITIES/URAU_RG_100K_2021_3035_CITIES.shp"
    geo_json_city = gpd.read_file(city_polygons)
    gdf_city = gpd.GeoDataFrame(geo_json_city, crs="EPSG:3035")
    # gdf_city = gdf_city[gdf_city.URAU_CODE.isin(['IT003C', 'FI003C', 'FI008C', 'BG017C', 'IS001C', 'DE002C', 'DE012C', 'NO004C', 'ES017C', 'PT001C', 'ES088C', 'ES091C', 'ES065C', 'ES010C', 'ES045C', 'ES069C', 'IT001C', 'FR072C', 
    #      'ES039C', 'IE001C', 'DK004C', 'ES062C', 'NL037C', 'IT067C', 'PT004C', 'SE011C', 'SE005C', 'SE004C', 'SE006C'])]
    # gdf_city = gdf_city[gdf_city.URAU_CODE.isin(["PT002C"])]
        # "AT001C", "IT003C","BG017C","DE002C","PT001C","ES088C","ES010C","ES069C","ES045C","ES039C","NL037C","PT004C"])]
    # define evalscript
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["CLC", "dataMask"],
        output: [{ 
        id: "data",
            bands: 1,
            sampleType: "UINT16" // raster format will be UINT16
            },
            {
            id: "dataMask",
            bands: 1
        }]
      }
    }

    function evaluatePixel(sample) {
        if(sample.CLC < 45 && sample.CLC > 39) {
            return {
            data: [sample.CLC],
            dataMask: [sample.dataMask]
            }
        } else {
            return {
            data: [sample.CLC*0],
            dataMask: [sample.dataMask]}
        }
    }

    """
    
    # create temporary df
    list_error = []
    df_all = pd.DataFrame(columns=['URAU_CODE', 'sampleCount', 'noDataCount', 
                                   'CLC_511', 'CLC_512', 'CLC_521', 'CLC_522', 'CLC_523'])
    for row in gdf_city.itertuples():
        logger.info(f"Downloading {row.URAU_CODE} {row.URAU_NAME} data")
        
        #------------------------------------------
        geometry_gdf = row.geometry
        geometry_b, bbox_b, bbox_size_b = utils.buffer_geometry(geometry_gdf, buffer_size=100)

        bbox_subsize_b = utils.bbox_optimal_subsize(bbox_size_b)
        if(bbox_subsize_b == 1 ):
            request = sentinelhub_stat_request(evalscript, geometry_b, bbox_b, bbox_size_b, config)
            try:
                data = request.get_data()[0]
                # print(data)
            except:
                logger.info("an error occurred")
                list_error.append(row.URAU_CODE)
                print(row.URAU_CODE)
            #     break
            # do something with the data
            if(len(data['data']) == 0):
                df = pd.DataFrame(data = {
                    'URAU_CODE': [row.URAU_CODE],
                    'sampleCount': [0],
                    'noDataCount': [0],
                    'CLC_511': [0],
                    'CLC_512': [0],
                    'CLC_521': [0],
                    'CLC_522': [0],
                    'CLC_523': [0]
                })
            else:
                df = pd.DataFrame(data = {
                    'URAU_CODE': [row.URAU_CODE],
                    'sampleCount': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['sampleCount']],
                    'noDataCount': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['noDataCount']],
                    'CLC_511': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][0]['count']],
                    'CLC_512': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][1]['count']],
                    'CLC_521': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][2]['count']],
                    'CLC_522': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][3]['count']],
                    'CLC_523': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][4]['count']],
                    })
            df_all = pd.concat([df_all, df])
            # print(df_all.tail(1))
            # break
        else:
            logger.info(f"Splitting bounding box in {(bbox_subsize_b,bbox_subsize_b)} subgrid")
            bbox_split = BBoxSplitter([geometry_b], CRS('3035').pyproj_crs(), bbox_subsize_b, reduce_bbox_sizes=True)
            # create a list of requests
            bbox_list = bbox_split.get_bbox_list()
            geometry_list = [Geometry(geometry=utils.split_geometry(geometry_b, bbox), crs=CRS('3035').pyproj_crs()) for bbox in bbox_list]
            print(bbox_list[0])
            print(geometry_list[0])
            sh_requests = [sentinelhub_stat_request(evalscript, geometry, subbbox, bbox_to_dimensions(subbbox, resolution=10), config) for (geometry,subbbox) in list(zip(geometry_list,bbox_list))]
            i = 1
            error=False
            df_tmp = pd.DataFrame(columns=['URAU_CODE', 'sampleCount', 'noDataCount',
                                   'CLC_511', 'CLC_512', 'CLC_521', 'CLC_522', 'CLC_523'])
            for req in sh_requests:
                # try:
                data = req.get_data()[0]
                print(data)
                # do something with the data
                if(len(data['data']) == 0):
                    df = pd.DataFrame(data = {
                        'URAU_CODE': [row.URAU_CODE],
                        'sampleCount': [0],
                        'noDataCount': [0],
                        'CLC_511': [0],
                        'CLC_512': [0],
                        'CLC_521': [0],
                        'CLC_522': [0],
                        'CLC_523': [0]
                    })
                else:
                    df = pd.DataFrame(data = {
                        'URAU_CODE': [row.URAU_CODE],
                        'sampleCount': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['sampleCount']],
                        'noDataCount': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['noDataCount']],
                        'CLC_511': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][0]['count']],
                        'CLC_512': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][1]['count']],
                        'CLC_521': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][2]['count']],
                        'CLC_522': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][3]['count']],
                        'CLC_523': [data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins'][4]['count']],
                        })

                df_tmp = pd.concat([df_tmp, df])
                i = i+1
                # except:
                    # logger.info("an error occurred")
                    # list_error.append(row.URAU_CODE)
                    # print(row.URAU_CODE)
                    # error=True
                    # break
            if(~error):
                df_tmp_gp = df_tmp.groupby('URAU_CODE').sum()
                df_tmp_gp.reset_index(inplace=True)
                print(df_tmp_gp)
                df_all = pd.concat([df_all, df_tmp_gp])
        print(df_all.tail(1))
    measurer.end(tracker=tracker,
                 shape=[],
                 libraries=[v.__name__ for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
                 data_path=data_path,
                 program_path=__file__,
                 csv_file='./../../s3/data/l001_logs/benchmarks_stats_CLC_v4.csv')
    print(list_error)
    return df_all, list_error


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df, errors = subcube()
    df.to_csv(f"./../../s3/data/c001_city_cube/CLC_v2.csv", mode='a')
    # errors.to_csv("failed_cities.csv")
