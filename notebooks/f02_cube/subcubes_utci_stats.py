#!/usr/bin/env python
# coding: utf-8

# # Get city statistics from climate data downloaded from CDS
# Workflow to get day- and nighttime utci statistics

# In[ ]:


import sqlite3
import pandas as pd
import xarray as xr
from src import utils
import time
input_folder = "./../../../s3/data/"
climate_folder = input_folder+"d003_climate/cl_01_utci/raw_data/"
year = '1996'


# ## Set up logger

# In[ ]:


from loguru import logger
# add log file
logger.add(input_folder+f"l001_logs/utci_logger_{year}.log")
logger.info(f"Processing data for year {year}")


# ## Download hourly data

# In[ ]:


logger.info("Start downloading data from CDS")
import cdsapi
c = cdsapi.Client()
args = {
    "months": ['01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',],
    "days":   ['01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31'],
    }
print(year)
c.retrieve(
        'derived-utci-historical', 
    {
        'version': '1_1',
        'format': 'zip',
        'day': args["days"],
        'month': args["months"],
        'year': year,
        'product_type': 'consolidated_dataset',
        'variable': 'universal_thermal_climate_index',
    },
    climate_folder+f'utci_hourly_{year}.zip')
logger.info("Finished downloading data from CDS")


# In[ ]:


logger.info("Start unzipping data")
output_folder = utils.unzip_to_folder(climate_folder, f'utci_hourly_{year}')
logger.info("Finished unzipping data")


# ## Get city coordinates

# In[ ]:


# path to databases
city_geom  = input_folder+'d000_lookuptables/city_pts_urban_audit2021.sqlite'
con = sqlite3.connect(city_geom)
# read full table
city_all = pd.read_sql_query("SELECT _wgs84x, _wgs84y, city_code FROM urau_lb_2021_3035_cities_center_points_4", con)
con.close()
# get city coordinates
# lonlat_list =[["NL005C", 4.640960, 52.113299], ["NL006C", 5.384670, 52.173656], ["NL007C", 5.921886, 52.189884]]
lon_list = city_all["_wgs84x"].values.tolist()
lat_list = city_all["_wgs84y"].values.tolist()
city_list = city_all["city_code"].values.tolist()
target_lon = xr.DataArray(lon_list, dims="city", coords={"city": city_list})
target_lat = xr.DataArray(lat_list, dims="city", coords={"city": city_list})


# ## Read the downloaded .nc file with xarray

# In[ ]:


logger.info("Start reading data as xarray")
climate_path = "/home/mari-s4e/s3/data/d003_climate/cl_01_utci/raw_data/utci_hourly_1996/*.nc"
# climate_folder+f"utci_hourly_1994/ECMWF_utci_1994*_v1.1_con.nc" 
data = xr.open_mfdataset(climate_path, engine="netcdf4", parallel=True, autoclose=True)
logger.info("Finished reading data as xarray")


# In[ ]:


# data


# ## Compute statistics

# In[ ]:


data_cities = data["utci"].sel(lon=target_lon, lat=target_lat, method="ffill")
data_cities_daytime = data_cities.resample(time="12H", base = 7)
utci_mean = data_cities_daytime.mean()
utci_min = data_cities_daytime.min()
utci_max = data_cities_daytime.max()


# In[ ]:


data_cities


# In[ ]:


from dask.distributed import Client
from dask.distributed import LocalCluster


# In[ ]:


logger.info("Start dask local cluster")
cluster = LocalCluster()
client = Client()


# In[ ]:


stats = xr.merge([utci_mean.rename("utci_mean"), utci_min.rename("utci_min"), utci_max.rename("utci_max")])
# stats


# ## Convert to GeoDataFrame

# In[ ]:


logger.info("Start computing statistics")
stats_df = stats.to_dataframe()
# stats_df
logger.info("Finished computing statistics")


# In[ ]:


logger.info("Save to shapefile")
import geopandas as gpd
stats_df = stats_df.reset_index()
gdf = gpd.GeoDataFrame(
    stats_df[["city", "time", "utci_mean", "utci_min", "utci_max"]], geometry=gpd.points_from_xy(stats_df.lon,stats_df.lat), crs="EPSG:4326")


# In[ ]:


gdf.to_file(input_folder+f"d003_climate/cl_01_utci/stats_{year}.shp", driver="GeoJSON")


# ## Delete original data to save space

# In[ ]:


# Print out bucket names
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)


# In[ ]:


# logger.info("Delete raw data")
# shutil.rmtree(input_folder+f"utci_hourly_{year}", ignore_errors=False, onerror=None)
import boto3

# Create an S3 client
s3 = boto3.client('s3')
bucket = "hub-fairicube0"
# List all objects in the folder
objects = s3.list_objects(Bucket=bucket, Prefix=f'data/d003_climate/cl_01_utci/raw_data/utci_hourly_{year}')
# Delete all objects in the folder
for obj in objects['Contents']:
    print(obj['Key'])
    s3.delete_object(Bucket=bucket, Key=obj['Key'])
# Delete the folder
s3.delete_object(Bucket=bucket, Key=f'data/d003_climate/cl_01_utci/raw_data/utci_hourly_{year}')
s3.delete_object(Bucket=bucket, Key=f'data/d003_climate/cl_01_utci/raw_data/utci_hourly_{year}.zip')


# ## Close connection to dask cluster

# In[ ]:


logger.info("Close dask cluster")
client.close()
cluster.close()
logger.info("Finished")

