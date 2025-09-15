# get treecover density or imperviousness density within core city in area (square meters)
# uses SentinelHub statistical API

import geopandas as gpd
import pandas as pd
import os

# load utils functions
from src import utils
from src.measurer import Measurer
from types import ModuleType
from loguru import logger

# Sentinel Hub
from sentinelhub.constants import CRS
from sentinelhub.api.statistical import SentinelHubStatistical
from sentinelhub.data_collections import DataCollection
from sentinelhub.config import SHConfig
from sentinelhub.areas import BBoxSplitter
from sentinelhub.geometry import Geometry
from src.utils import *


@logger.catch()
def sentinelhub_stat_request(
    hrl, hrl_id, evalscript, geometry, bbox, bbox_size, config
):
    calculations = {
        "default": {
            "histograms": {
                "default": {
                    "binWidth": "1",
                    "lowEdge": "0",
                    "highEdge": "101",  # histogram interval is [lowEdge, highEdge) that is, highEdge value excluded
                }
            }
        }
    }
    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=("2018-01-01", "2019-05-01"),
            aggregation_interval="P1D",
            size=bbox_size,
        ),
        # input_data=[SentinelHubStatistical.input_data(DataCollection.define_byoc('3947b646-383c-4e91-aade-2f039bd6ba4b', name=f'{hrl}Density2018'))],
        input_data=[
            SentinelHubStatistical.input_data(
                DataCollection.define_byoc(hrl_id, name=hrl)
            )
        ],
        bbox=bbox,
        geometry=geometry,
        calculations=calculations,
        config=config,
    )
    return request


def sh_config():
    """Define here your Sentinel Hub configuration

    Returns:
        config (SHConfig): Sentinel Hub configuration object
    """
    config = SHConfig()
    config.instance_id = os.environ.get("SH_INSTANCE_ID")  # type: ignore
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")  # type: ignore
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")  # type: ignore
    config.aws_access_key_id = os.environ.get("username")  # type: ignore
    config.aws_secret_access_key = os.environ.get("password")  # type: ignore
    return config


@logger.catch
def sentinelhub_stat_request_wrapper(hrl, hrl_id, df_cities):

    config = sh_config()

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
    df_all = pd.DataFrame(
        columns=["URAU_CODE", "tot_areaSqm", "noDataCount", f"{hrl}_areaSqm"]
    )
    for row in df_cities.itertuples():
        logger.info(f"Downloading {row.URAU_NAME} data")

        # ------------------------------------------
        geometry_gdf = row.geometry
        geometry_b, bbox_b, bbox_size_b = utils.buffer_geometry(
            geometry_gdf, df_cities.crs, buffer_size=0
        )

        bbox_subsize_b = utils.bbox_optimal_subsize(bbox_size_b)
        if bbox_subsize_b == 1:
            request = sentinelhub_stat_request(
                hrl, hrl_id, evalscript, geometry_b, bbox_b, bbox_size_b, config
            )
            try:
                data = request.get_data()[0]
                # print(data)
            except:
                logger.info("an error occurred")
                print(row.URAU_CODE)
                break
            # do something with the data
            if len(data["data"]) == 0:
                df = pd.DataFrame(
                    data={
                        "URAU_CODE": [row.URAU_CODE],
                        "tot_areaSqm": [0],
                        "noDataCount": [0],
                        f"{hrl}_areaSqm": [0],
                    }
                )
            else:
                df = pd.DataFrame(
                    data={
                        "URAU_CODE": [row.URAU_CODE],
                        "tot_areaSqm": [
                            data["data"][0]["outputs"]["data"]["bands"]["B0"]["stats"][
                                "sampleCount"
                            ]
                            * 100
                        ],
                        "noDataCount": [
                            data["data"][0]["outputs"]["data"]["bands"]["B0"]["stats"][
                                "noDataCount"
                            ]
                        ],
                        f"{hrl}_areaSqm": [
                            sum(
                                [
                                    line["lowEdge"] * line["count"]
                                    for line in data["data"][0]["outputs"]["data"][
                                        "bands"
                                    ]["B0"]["histogram"]["bins"]
                                ]
                            )
                        ],
                    }
                )
            df_all = pd.concat([df_all, df])
            # break
        else:
            logger.info(
                f"Splitting bounding box in {(bbox_subsize_b,bbox_subsize_b)} subgrid"
            )
            bbox_split = BBoxSplitter([bbox_b], CRS("3035"), bbox_subsize_b)
            # create a list of requests
            bbox_list = bbox_split.get_bbox_list()
            geometry_list = [
                Geometry(
                    geometry=utils.split_geometry(geometry_b, bbox),
                    crs=CRS("3035"),
                )
                for bbox in bbox_list
            ]
            sh_requests = [
                sentinelhub_stat_request(
                    hrl,
                    hrl_id,
                    evalscript,
                    geometry,
                    subbbox,
                    bbox_to_dimensions(subbbox, resolution=10),
                    config,
                )
                for (geometry, subbbox) in list(zip(geometry_list, bbox_list))
            ]
            i = 1
            error = False
            df_tmp = pd.DataFrame(
                columns=["URAU_CODE", "tot_areaSqm", "noDataCount", f"{hrl}_areaSqm"]
            )
            for req in sh_requests:
                try:
                    data = req.get_data()[0]
                    # do something with the data
                    if len(data["data"]) == 0:
                        df = pd.DataFrame(
                            data={
                                "URAU_CODE": [row.URAU_CODE],
                                "tot_areaSqm": [0],
                                "noDataCount": [0],
                                f"{hrl}_areaSqm": [0],
                            }
                        )
                    else:
                        df = pd.DataFrame(
                            data={
                                "URAU_CODE": [row.URAU_CODE],
                                "tot_areaSqm": [
                                    data["data"][0]["outputs"]["data"]["bands"]["B0"][
                                        "stats"
                                    ]["sampleCount"]
                                    * 100
                                ],
                                "noDataCount": [
                                    data["data"][0]["outputs"]["data"]["bands"]["B0"][
                                        "stats"
                                    ]["noDataCount"]
                                ],
                                f"{hrl}_areaSqm": [
                                    sum(
                                        [
                                            line["lowEdge"] * line["count"]
                                            for line in data["data"][0]["outputs"][
                                                "data"
                                            ]["bands"]["B0"]["histogram"]["bins"]
                                        ]
                                    )
                                ],
                            }
                        )

                    df_tmp = pd.concat([df_tmp, df])
                    i = i + 1
                except:
                    logger.info("an error occurred")
                    print(row.URAU_CODE)
                    error = True
                    break
            if ~error:
                df_tmp_gp = df_tmp.groupby("URAU_CODE").sum()
                df_tmp_gp.reset_index(inplace=True)
                print(df_tmp_gp)
                df_all = pd.concat([df_all, df_tmp_gp])
        print(df_all.tail(1))
    return df_all


if __name__ == "__main__":
    layer_name = "imd"
    layer_id = "c57f7668-2717-4529-93cc-5372bc96ebbe"
    data_path = "/"
    logger.add(f"logfile_{layer_name}.log")
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)

    # read city polygons
    # Path where the data are stored (the use of the disk in this path is measured).
    logger.info("Start")
    # load city polygons
    city_polygons = (
        "./data/city_features_collection/URAU_RG_100K_2021_3035_CITIES.geojson"
    )
    geo_json_city = gpd.read_file(city_polygons)
    gdf_city = gpd.GeoDataFrame(geo_json_city, crs="EPSG:3035")

    # use subset for testing
    gdf_city = gdf_city[~gdf_city.URAU_CODE.isin(["FI004C", "BG016C", "SE008C"])]

    df = sentinelhub_stat_request_wrapper(layer_name, layer_id, gdf_city)

    # save results
    df.to_csv(f"./../../s3/data/c001_city_cube/{layer_name}.csv", mode="a")
    measurer.end(
        tracker=tracker,
        shape=[],
        libraries=[
            v.__name__
            for k, v in globals().items()
            if type(v) is ModuleType and not k.startswith("__")
        ],
        data_path=data_path,
        program_path=__file__,
        variables=[],
        csv_file=f"./../../s3/data/l001_logs/benchmarks_stats_{layer_name}.csv",
    )
