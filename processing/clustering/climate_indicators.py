'''
Compute climate indicators for selected cities

Datasource are ERA-5 Data on: https://cloud.google.com/storage/docs/public-datasets/era5?hl=de

We use this data source because the data is available there in an analysis ready cloud-optimised format.

The selected climate indices are:
- Number of summer days
- Number of tropical nights
- 2m temperature statistics (mean, std, min, max)
- total precipitation (mean, std, min, max)
'''

import sqlalchemy as sa # conection to the database
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
# from src.utils import db_connect
from configparser import ConfigParser
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr      
import fsspec
from dask.distributed import Client, performance_report
from src.measurer import Measurer
from types import ModuleType
from loguru import logger

# @logger.catch()
def config(filename, section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))

    return db

# @logger.catch()
def lon_to_360(dlon: float) -> float:
    return ((360 + (dlon % 360)) % 360)

# @logger.catch()
def daytime(x):
    time_local = x.time+timedelta(hours=x.time_zone_offset)
    if(time_local.hour >= 6 and time_local.hour < 18):
        return 1
    else:
        return 0

# @logger.catch()
def night_date(x):
    # assign previous day date to nighttime hours [0..end_night]
    # first adjust to local time
    time_local = x.time+timedelta(hours=x.time_zone_offset)
    if(time_local.hour < 6):
        # date = x.time - timedelta(days=1)
        return x.time - timedelta(days=1)
    else:
        return x.time

@logger.catch()
def summer_days_tropical_nights():
    # start meter
    # Path where the data are stored (the use of the disk in this path is measured).
    data_path = '/'
    logger.add(f"./../../s3/data/l001_logs/logfile_2m_temperature_summer_days_tropical_nights_2018.log")
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    logger.info("Start")
    
    # start dask client
    client = Client()
    year = '2018'
    # connect to data
    fs = fsspec.filesystem('gs')
    fs.ls('gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/')
    arco_era5 = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2', 
        # chunks={'time': 24*30},
        consolidated=True,
        drop_variables=["10m_u_component_of_wind",  
        "10m_v_component_of_wind",
        # "2m_temperature",
        "angle_of_sub_gridscale_orography",
        "anisotropy_of_sub_gridscale_orography",
        "geopotential",
        "geopotential_at_surface",
        "high_vegetation_cover",
        "lake_cover",
        "lake_depth",
        "land_sea_mask",
        "low_vegetation_cover",
        "mean_sea_level_pressure",
        "sea_ice_cover",
        "sea_surface_temperature",
        "slope_of_sub_gridscale_orography",
        "soil_type",
        "specific_humidity",
        "standard_deviation_of_filtered_subgrid_orography",
        "standard_deviation_of_orography",
        "surface_pressure",
        "temperature",
        "toa_incident_solar_radiation",
        "total_cloud_cover",
        "total_column_water_vapour",
        # "total_precipitation",
        "type_of_high_vegetation",
        "type_of_low_vegetation",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"]
    )
    
    logger.info("Connected to data")
    # get cities
    db_config = './../database.ini'
    keys = config(filename=db_config)
    engine_postgresql = sa.create_engine('postgresql://'+keys['user']+':'+keys['password']+ '@'+keys['host']+':'+str(keys['port'])+ '/' + keys['database'])
    # read list of cities
    with engine_postgresql.begin() as conn:
        query = text("""SELECT city_code, _wgs84y, _wgs84x, time_zone_offset
    FROM lut.l_city_urau2021;""")
        city_center_df = pd.read_sql_query(query, conn)
    
    city_center_df_r = city_center_df.reset_index()
    lon_list = [lon_to_360(val) for val in city_center_df_r["_wgs84x"].values.tolist()]
    lat_list = city_center_df_r["_wgs84y"].values.tolist()
    city_list = city_center_df_r["city_code"].values.tolist()
    target_lon = xr.DataArray(lon_list, dims="city", coords={"city": city_list})
    target_lat = xr.DataArray(lat_list, dims="city", coords={"city": city_list})
    time_zone_offset = xr.DataArray(city_center_df_r['time_zone_offset'], dims="city", coords={"city": city_list})

    start_date = "2018-01-01"
    end_date = "2018-12-31"
    
    # Filter datacube by location (city centre point coordinates, time)
    data = arco_era5.sel(time=slice(start_date, end_date)).sel(
    longitude=target_lon, 
    latitude=target_lat, method="ffill")

    data = xr.merge([data,time_zone_offset])
    
    # transform into Dask DataFrame to speed up computations
    data_df = data.to_dask_dataframe()
    data_df = data_df.reset_index()
    # select only relevant variables
    data_df = data_df[['city', 'time', 'time_zone_offset', 'latitude', 'longitude', '2m_temperature', 'total_precipitation']]
    logger.info("Filtered data")
    # compute daytime/nighttime max of 2m temperature (12H window) by city
    # take care of different timezones
    data_df['daytime'] = data_df.apply(daytime, axis=1, meta=(None, 'int'))
    # change night hours date to count night hours together
    # example 01-01-2018 18H-23H + 02-01-2018 00H-06H = 1 night therefore
    # change date 02-01-2018 to 01-01-2018 for 00H-06H hours
    data_df['time_shifted'] =data_df.apply(night_date, axis=1, meta=(None, 'datetime64[ns]'))
    data_df['date'] = data_df.time_shifted.dt.date
    
    with performance_report(filename="temp_max_2018_v2_test.html"):
        temp_max = data_df.groupby(['city', 'daytime', 'date']).max()
        # -------------------------------------------------------------------------------------------------------------
        temp_max_c = temp_max.compute()
    client.close()
    logger.info("Finished computing max")
    # compute count of tropical nights and summer days per year per city
    # 2m temperature is in Kelvin
    tropical_threshold = 30.0 + 273.15
    summer_day_threshold = 25.0 + 273.15
    temp_max_c = temp_max_c.reset_index()
    summer_days = temp_max_c.loc[(temp_max_c['2m_temperature'] > summer_day_threshold) & (temp_max_c['daytime'] == 1)].reset_index()
    tropical_count = temp_max_c.loc[(temp_max_c['2m_temperature'] > tropical_threshold) & (temp_max_c['daytime'] == 0)].reset_index()
    tropical_count = tropical_count.groupby(['city', 'daytime']).count()
    summer_days_count = summer_days_count.groupby(['city', 'daytime']).count()
    temp_max_c.to_csv("test_missing_cities_summer_days.csv")
    logger.info("Finished computing count")
    # prepare dataframes to be saved into database
    # summer days
#     table_name = 'cu_city_era5_summer_days'
#     schema_name = 'cube'
#     summer = summer_days_count
#     summer.rename( columns= {'city': 'city_code',
#                              'index': 'parameter_value'}, inplace=True)
#     summer['year'] = year
#     summer['parameter'] = 'Count of summer days (>25 degrees) per year per city, based on 5th gen. ECMWF Atmospheric Reanalysis model'
#     summer['parameter_id'] = 'city_era5_summer_days_count'
#     summer['lineage'] = 'https://github.com/FAIRiCUBE/uc1-urban-climate/blob/master/notebooks/dev/f04_climate_data/climate_indicators.ipynb'
#     summer['datasource'] = 'https://cloud.google.com/storage/docs/public-datasets/era5'
#     summer.to_sql(table_name, engine_postgresql, schema=schema_name, if_exists='replace')
#     logger.info("Saved summer days to database")
    
#     # tropical nights
#     table_name = 'cu_city_era5_tropical_nights'
#     schema_name = 'cube'

#     tropical_count.rename( columns= {'city': 'city_code', 'index': 'parameter_value'}, inplace=True)
#     tropical_count['year'] = year
#     tropical_count['parameter'] = 'Count of tropical nights (>305 degrees) per year per city, based on 5th gen. ECMWF Atmospheric Reanalysis model'
#     tropical_count['parameter_id'] = 'city_era5_tropical_nights_count'
#     tropical_count['lineage'] = 'https://github.com/FAIRiCUBE/uc1-urban-climate/blob/master/notebooks/dev/f04_climate_data/climate_indicators.ipynb'
#     tropical_count['datasource'] = 'https://cloud.google.com/storage/docs/public-datasets/era5'
#     tropical_count.to_sql(table_name, engine_postgresql, schema=schema_name, if_exists='replace')
#     logger.info("Saved tropical nights to database")
    
    
    measurer.end(tracker=tracker,
             shape=[],
             libraries=[v.__name__ for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
             data_path=data_path,
             program_path=__file__,
             csv_file=f'./../../s3/data/l001_logs/benchmarks_2m_temperature_summer_days_tropical_nights_2018_test.csv')
        
if __name__ == "__main__":
    
    summer_days_tropical_nights()