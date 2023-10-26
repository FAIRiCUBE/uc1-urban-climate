#!/usr/bin/env python
# coding: utf-8
# Author: Maria Ricci

# # Get city statistics from climate data downloaded from CDS
# Workflow to get day- and nighttime utci statistics

import sqlite3
import pandas as pd
import xarray as xr
from src import utils
from src.measurer import Measurer
import time
from dask.distributed import Client
from dask.distributed import LocalCluster
import geopandas as gpd
from loguru import logger

########################
## !! important !!
# LocalCluster can start only within if __name__ == "__main__":
## Github issue: https://github.com/dask/distributed/issues/4751
    
@logger.catch
def run_process():
    # parameters
    input_folder = "./../../../s3/data/"
    climate_folder = input_folder+"d003_climate/cl_01_utci/raw_data/"
    year = '1994'

    logger.info(f"Processing data for year {year}")

    logger.info("Start dask local cluster")
    # cluster = LocalCluster()
    # client = Client(cluster)
    # print(cluster.dashboard_link)
    
    # record computational costs
    measurer = Measurer()
    tracker = measurer.start(data_path=climate_folder, logger=logger)


    # ## Read the downloaded .nc file with xarray
    logger.info("Start reading data as xarray")
    climate_path = climate_folder+"utci_hourly_1994/ECMWF_utci_1994010*.nc"
    # climate_folder+f"utci_hourly_1994/ECMWF_utci_1994*_v1.1_con.nc" 
    data = xr.open_mfdataset(climate_path, engine="netcdf4", parallel=True, chunks={"latitude": 100, "longitude": 100})
    logger.info("Finished reading data as xarray")

#     # ## Compute statistics
#     data_cities = data["utci"].sel(lon=target_lon, lat=target_lat, method="ffill")
#     data_cities_daytime = data_cities.resample(time="12H", base = 7)
#     utci_mean = data_cities_daytime.mean()
#     utci_min = data_cities_daytime.min()
#     utci_max = data_cities_daytime.max()

#     # client = Client()
#     stats = xr.merge([utci_mean.rename("utci_mean"), utci_min.rename("utci_min"), utci_max.rename("utci_max")])
#     # ## Convert to GeoDataFrame
#     logger.info("Start computing statistics")
#     stats_df = stats.to_dataframe()
#     logger.info("Finished computing statistics")
#     logger.info("Save to shapefile")
#     stats_df = stats_df.reset_index()
#     gdf = gpd.GeoDataFrame(
#         stats_df[["city", "time", "utci_mean", "utci_min", "utci_max"]], geometry=gpd.points_from_xy(stats_df.lon,stats_df.lat), crs="EPSG:4326")
#     gdf.to_file(input_folder+f"d003_climate/cl_01_utci/stats_{year}_v2.shp", driver="GeoJSON")



#     ## Close connection to dask cluster

#     logger.info("Close dask cluster")
    # client.close()
    # cluster.close()
    logger.info("Finished")

    # stop measurer
    from types import ModuleType
    measurer.end(tracker=tracker,
                 shape=[],
                 libraries=[k for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
                 data_path=climate_folder+"utci_hourly_1994/",
                 csv_file=input_folder+f"l001_logs/utci_benchmarks_{year}_test_open_mfdataset2.csv",
                logger=logger)

if __name__ == "__main__":
    # ## Set up logger
    # add log file
    year = '1994'
    logger.add(f"./../../../s3/data/l001_logs/utci_logger_{year}_test_open_mfdataset2.log", enqueue=True)
    run_process()