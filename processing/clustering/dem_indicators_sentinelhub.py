# get elevation statistics (min, max, mean, std) within city from Copernicus DEM 30m
# uses SentinelHub statistical API

import numpy as np
import geopandas as gpd
import pandas as pd
import os

# load utils functions
from src import utils
from src.measurer import Measurer
from types import ModuleType
from loguru import logger

# Sentinel Hub
# Sentinel Hub
from sentinelhub.constants import CRS
from sentinelhub.api.statistical import SentinelHubStatistical
from sentinelhub.data_collections import DataCollection
from sentinelhub.config import SHConfig
from sentinelhub.areas import BBoxSplitter
from sentinelhub.geometry import Geometry
from src.utils import *


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


@logger.catch()
def sentinelhub_request(evalscript, geometry, bbox, bbox_size, config):
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.DEM_COPERNICUS_30,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_size,
        geometry=geometry,
        config=config,
    )
    return request


@logger.catch
def sentinelhub_request_wrapper(dim_name: str, gdf_city) -> pd.DataFrame:

    config = sh_config()

    # Path where the data are stored (the use of the disk in this path is measured).
    data_path = "/"
    logger.add(f"./../../s3/data/l001_logs/logfile_{dim_name}.log")
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    logger.info("Start")

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
    df_all = pd.DataFrame(
        columns=["URAU_CODE", "dem_min", "dem_max", "dem_mean", "dem_std", "dem_count"]
    )
    for row in gdf_city.itertuples():

        logger.info(f"Downloading {row.URAU_CODE} {row.URAU_NAME}")

        # ------------------------------------------
        geometry_gdf = row.geometry
        geometry_b, bbox_b, bbox_size_b = utils.buffer_geometry(
            geometry_gdf, crs=CRS("3035"), buffer_size=0
        )

        bbox_subsize_b = utils.bbox_optimal_subsize(bbox_size_b)
        if bbox_subsize_b == 1:
            request = sentinelhub_request(
                evalscript, geometry_b, bbox_b, bbox_size_b, config
            )
            try:
                data = request.get_data()[0]
                # set nodata to numpy nan
                data = data.astype("float")
                data[data == 0] = np.nan
                # shift back data to original value (see evalscript)
                data = data - 1
                if np.nanstd(data) > 0:
                    df = pd.DataFrame(
                        data={
                            "URAU_CODE": [row.URAU_CODE],
                            "dem_min": [np.nanmin(data)],
                            "dem_max": [np.nanmax(data)],
                            "dem_mean": [np.nanmean(data)],
                            "dem_std": [np.nanstd(data)],
                            "dem_count": [len(data)],
                        }
                    )
            except:
                logger.info("an error occurred")
                print(row.URAU_CODE)
                break
            df_all = pd.concat([df_all, df])
            # break
        else:
            logger.info(
                f"Splitting bounding box in {(bbox_subsize_b,bbox_subsize_b)} subgrid"
            )
            bbox_split = BBoxSplitter(
                [geometry_b], CRS("3035"), bbox_subsize_b, reduce_bbox_sizes=True
            )
            # create a list of requests
            bbox_list = bbox_split.get_bbox_list()
            geometry_list = [
                Geometry(
                    geometry=utils.split_geometry(geometry_b, bbox), crs=CRS("3035")
                )
                for bbox in bbox_list
            ]
            sh_requests = [
                sentinelhub_request(
                    evalscript,
                    geometry,
                    subbbox,
                    bbox_to_dimensions(subbbox, resolution=10),
                    config,
                )
                for (geometry, subbbox) in list(zip(geometry_list, bbox_list))
            ]
            error = False
            data_tmp = np.array([])
            for idx, req in enumerate(sh_requests):
                try:
                    data = req.get_data()[0]
                    # set nodata to numpy nan
                    data = data.astype("float")
                    data[data == 0] = np.nan
                    # shift back data to original value (see evalscript)
                    data = data - 1
                    if np.nanstd(data) > 0:
                        data_tmp = np.concatenate((data_tmp, data.ravel()))
                        # print(data)
                        # print(data_tmp)
                        logger.info(f"Processing subbox no.{idx}")
                except:
                    logger.info("an error occurred")
                    print(row.URAU_CODE)
                    error = True
                    break
            if ~error and len(data_tmp) > 0:
                logger.info("Concatenating results")
                df = pd.DataFrame(
                    data={
                        "URAU_CODE": [row.URAU_CODE],
                        "dem_min": [np.nanmin(data_tmp)],
                        "dem_max": [np.nanmax(data_tmp)],
                        "dem_mean": [np.nanmean(data_tmp)],
                        "dem_std": [np.nanstd(data_tmp)],
                        "dem_count": [len(data_tmp)],
                    }
                )
                df_all = pd.concat([df_all, df])
        print(df_all[df_all.URAU_CODE == row.URAU_CODE])
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
        csv_file=f"./../../s3/data/l001_logs/benchmarks_stats_{dim_name}_v3.csv",
    )
    return df_all


if __name__ == "__main__":
    layer_name = "DEM_COPERNICUS_30"
    # load city polygons
    # download from Eurostat GISCO
    city_polygons = "path/to/URAU_RG_100K_2021_3035_CITIES.shp"
    gdf_city = gpd.read_file(city_polygons, crs="EPSG:3035")
    df = sentinelhub_request_wrapper(layer_name, gdf_city)
    df.to_csv(f"./../../s3/data/c001_city_cube/{layer_name}.csv", mode="a")
