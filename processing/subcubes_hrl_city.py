# get treecover density or imperviousness density within core city in area (square meters)
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
def sentinelhub_stat_request(hrl, hrl_id, evalscript, geometry, bbox, bbox_size, config):
    calculations = {
        "default": {
            "histograms": {
                "default": {
                    "binWidth": "1",
                    "lowEdge": "0",
                    "highEdge": "101" #histogram interval is [lowEdge, highEdge) that is, highEdge value excluded
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
        # input_data=[SentinelHubStatistical.input_data(DataCollection.define_byoc('3947b646-383c-4e91-aade-2f039bd6ba4b', name=f'{hrl}Density2018'))],
        input_data=[SentinelHubStatistical.input_data(DataCollection.define_byoc(hrl_id, name=hrl))],
        bbox=bbox,
        geometry = geometry,
        calculations=calculations,
        config=config,
    )
    return request


@logger.catch
def subcube(hrl, hrl_id):
    
    config = SHConfig()
    config.instance_id = os.environ.get("SH_INSTANCE_ID")
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    config.aws_access_key_id = os.environ.get("username")
    config.aws_secret_access_key = os.environ.get("password")

    # Path where the data are stored (the use of the disk in this path is measured).
    data_path = '/'
    logger.add(f"logfile_{hrl}.log")
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    logger.info("Start")
    # load city polygons
    city_polygons = "./../../s3/data/d001_administration/urban_audit_city_2021/URAU_RG_100K_2021_3035_CITIES/URAU_RG_100K_2021_3035_CITIES.shp"
    geo_json_city = gpd.read_file(city_polygons)
    gdf_city = gpd.GeoDataFrame(geo_json_city, crs="EPSG:3035")
    gdf_city = gdf_city[~gdf_city.URAU_CODE.isin(['FI004C', 'BG016C', 'SE008C'])]
    # define evalscript
    evalscript = """

    //VERSION=3
    function setup() {
    return {
        input: ["B01", "dataMask"],
        output: [{ 
        id: "data",
            bands: 1,
            sampleType: "UINT16" // raster format will be UINT16
            },
            {
            id: "dataMask",
            bands: 1
        }]
        
    };
    }

    function evaluatePixel(sample) {
    return {
    data: [sample.B01],
    dataMask: [sample.dataMask]};
    }
    """
    
    # create temporary df
    df_all = pd.DataFrame(columns=['URAU_CODE', 'tot_areaSqm', 'noDataCount', f'{hrl}_areaSqm'])
    for row in gdf_city.itertuples():
        logger.info(f"Downloading {row.URAU_NAME} data")
        
        #------------------------------------------
        geometry_gdf = row.geometry
        geometry_b, bbox_b, bbox_size_b = utils.buffer_geometry(geometry_gdf, buffer_size=0)

        bbox_subsize_b = utils.bbox_optimal_subsize(bbox_size_b)
        if(bbox_subsize_b == 1 ):
            request = sentinelhub_stat_request(hrl, hrl_id,evalscript, geometry_b, bbox_b, bbox_size_b, config)
            try:
                data = request.get_data()[0]
                # print(data)
            except:
                logger.info("an error occurred")
                print(row.URAU_CODE)
                break
            # do something with the data
            if(len(data['data']) == 0):
                df = pd.DataFrame(data = {
                'URAU_CODE': [row.URAU_CODE],
                'tot_areaSqm': [0],
                'noDataCount': [0],
                f'{hrl}_areaSqm': [0]
                })
            else:
                df = pd.DataFrame(data = {
                    'URAU_CODE': [row.URAU_CODE],
                    'tot_areaSqm': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['sampleCount']*100],
                    'noDataCount': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['noDataCount']],
                    f'{hrl}_areaSqm': [sum([line['lowEdge']*line['count'] for line in data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins']])]
                    })
            df_all = pd.concat([df_all, df])
            # break
        else:
            logger.info(f"Splitting bounding box in {(bbox_subsize_b,bbox_subsize_b)} subgrid")
            bbox_split = BBoxSplitter([bbox_b], CRS('3035').pyproj_crs(), bbox_subsize_b)
            # create a list of requests
            bbox_list = bbox_split.get_bbox_list()
            geometry_list = [Geometry(geometry=utils.split_geometry(geometry_b, bbox), crs=CRS('3035').pyproj_crs()) for bbox in bbox_list]
            sh_requests = [sentinelhub_stat_request(hrl, hrl_id, evalscript, geometry, subbbox, bbox_to_dimensions(subbbox, resolution=10), config) for (geometry,subbbox) in list(zip(geometry_list,bbox_list))]
            i = 1
            error=False
            df_tmp = pd.DataFrame(columns=['URAU_CODE', 'tot_areaSqm', 'noDataCount', f'{hrl}_areaSqm'])
            for req in sh_requests:
                try:
                    data = req.get_data()[0]
                    # do something with the data
                    if(len(data['data']) == 0):
                        df = pd.DataFrame(data = {
                        'URAU_CODE': [row.URAU_CODE],
                        'tot_areaSqm': [0],
                        'noDataCount': [0],
                        f'{hrl}_areaSqm': [0]
                        })
                    else:
                        df = pd.DataFrame(data = {
                            'URAU_CODE': [row.URAU_CODE],
                            'tot_areaSqm': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['sampleCount']*100],
                            'noDataCount': [data['data'][0]['outputs']['data']['bands']['B0']['stats']['noDataCount']],
                            f'{hrl}_areaSqm': [sum([line['lowEdge']*line['count'] for line in data['data'][0]['outputs']['data']['bands']['B0']['histogram']['bins']])]
                            })

                    df_tmp = pd.concat([df_tmp, df])
                    i = i+1
                except:
                    logger.info("an error occurred")
                    print(row.URAU_CODE)
                    error=True
                    break
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
                 csv_file=f'./../../s3/data/l001_logs/benchmarks_stats_{hrl}.csv')
    return df_all
    

if __name__ == "__main__":
    hrl = 'imd'
    df = subcube(hrl, 'c57f7668-2717-4529-93cc-5372bc96ebbe')
    df.to_csv(f"./../../s3/data/c001_city_cube/{hrl}.csv", mode='a')