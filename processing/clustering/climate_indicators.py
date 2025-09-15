"""
Compute climate indicators for selected cities

Analysis ready cloud-optimised format data source for ERA-5 on Google Cloud: https://cloud.google.com/storage/docs/public-datasets/era5?hl=de

The selected climate indices are:
- Number of summer days
- Number of tropical nights
- 2m temperature statistics (mean, std, min, max)
- total precipitation (mean, std, min, max)
"""

from datetime import timedelta
from math import log

# from src.utils import db_connect
import pandas as pd
import xarray as xr
import fsspec
from dask.distributed import Client, performance_report
from src.measurer import Measurer
from types import ModuleType
from loguru import logger
import geopandas as gpd


def daytime(x):
    time_local = x.time + timedelta(hours=x.time_zone_offset)
    if time_local.hour >= 6 and time_local.hour < 18:
        return 1
    else:
        return 0


def night_date(x):
    # assign previous day date to nighttime hours [0..end_night]
    # first adjust to local time
    time_local = x.time + timedelta(hours=x.time_zone_offset)
    if time_local.hour < 6:
        # date = x.time - timedelta(days=1)
        return x.time - timedelta(days=1)
    else:
        return x.time


def lon_to_360(dlon: float) -> float:
    return (360 + (dlon % 360)) % 360


# @logger.catch()
def get_ERA5_max_2m_temperature(
    data_url: str,
    df_cities: pd.DataFrame = pd.DataFrame(
        {
            "city_code": ["DE0001", "DE0002", "DE0003"],
            "_wgs84y": [52.5200, 48.1351, 50.1109],
            "_wgs84x": [13.4050, 11.5820, 8.6821],
            "time_zone_offset": [1, 1, 1],
        }
    ),
    start_date: str = "2018-01-01",
    end_date: str = "2018-01-11",
    performance_report_path="",
) -> pd.DataFrame:
    # connect to data
    fs = fsspec.filesystem("gs")
    fs.ls(data_url)
    arco_era5 = xr.open_zarr(
        data_url,
        chunks={"time": 24 * 30},  # type: ignore
        consolidated=True,
        drop_variables=[  # all except 2m temperature
            "10m_u_component_of_wind",
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
            "total_precipitation",
            "type_of_high_vegetation",
            "type_of_low_vegetation",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        ],
    )

    logger.info("Connected to data")
    # get cities
    # check that df_cities has the required columns
    required_columns = ["city_code", "_wgs84y", "_wgs84x"]
    for col in required_columns:
        if col not in df_cities.columns:
            raise ValueError(f"df_cities must contain the column '{col}'")
    if "time_zone_offset" not in df_cities.columns:
        df_cities["time_zone_offset"] = 1  # default to 1 if not provided

    lon_list = [lon_to_360(val) for val in df_cities["_wgs84x"].values.tolist()]
    lat_list = df_cities["_wgs84y"].values.tolist()
    city_list = df_cities["city_code"].values.tolist()
    target_lon = xr.DataArray(lon_list, dims="city", coords={"city": city_list})
    target_lat = xr.DataArray(lat_list, dims="city", coords={"city": city_list})
    time_zone_offset = xr.DataArray(
        df_cities["time_zone_offset"], dims="city", coords={"city": city_list}
    )

    # Filter datacube by location (city centre point coordinates, time)
    data = arco_era5.sel(time=slice(start_date, end_date)).sel(
        longitude=target_lon, latitude=target_lat, method="ffill"
    )

    data = xr.merge([data, time_zone_offset])

    # transform into Dask DataFrame to speed up computations
    data_df = data.to_dask_dataframe()
    data_df = data_df.reset_index()
    # select only relevant variables
    data_df = data_df[
        [
            "city",
            "time",
            "time_zone_offset",
            "latitude",
            "longitude",
            "2m_temperature",
        ]
    ]
    logger.info("Filtered data")
    # compute daytime/nighttime max of 2m temperature (12H window) by city
    # take care of different timezones
    data_df["daytime"] = data_df.apply(daytime, axis=1, meta=(None, "int"))
    # change night hours date to count night hours together
    # example 01-01-2018 18H-23H + 02-01-2018 00H-06H = 1 night therefore
    # change date 02-01-2018 to 01-01-2018 for 00H-06H hours
    data_df["time_shifted"] = data_df.apply(
        night_date, axis=1, meta=(None, "datetime64[ns]")
    )
    data_df["date"] = data_df.time_shifted.dt.date

    # start dask client
    client = Client()
    print(client.dashboard_link)
    if performance_report_path != "":
        logger.info("Starting performance report")
        with performance_report(filename=performance_report_path):
            temp_max = data_df.groupby(["city", "daytime", "date"]).max()
            temp_max_c = temp_max.compute()
    else:
        logger.info("Starting computation without performance report")
        temp_max = data_df.groupby(["city", "daytime", "date"]).max()
        temp_max_c = temp_max.compute()
    client.close()
    logger.info("Finished computing max")
    return temp_max_c


if __name__ == "__main__":
    # start meter
    # Path where the data are stored (the use of the disk in this path is measured).
    data_path = "/"
    logger.add(
        f"./../../s3/data/l001_logs/logfile_2m_temperature_summer_days_tropical_nights_2018.log"
    )
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    logger.info("Start")

    data_url = (
        "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
    )

    # get df cities
    cities_path = "./data/city_features_collection/URAU_LB_2021_4326.geojson"
    load_cities_df = gpd.read_file(cities_path)
    load_cities_df.rename(columns={"URAU_CODE": "city_code"}, inplace=True)
    # extract x,y coordinates from geometry
    load_cities_df["_wgs84x"] = load_cities_df.geometry.x
    load_cities_df["_wgs84y"] = load_cities_df.geometry.y

    logger.info(f"Loaded {len(load_cities_df)} cities")
    df_temp = get_ERA5_max_2m_temperature(
        data_url=data_url,
        df_cities=load_cities_df[["city_code", "_wgs84x", "_wgs84y"]].iloc[
            0:1
        ],  # test with 1 city
        start_date="2018-01-01",
        end_date="2018-01-10",
        performance_report_path="performance_report_2m_temperature.html",
    )

    # compute count of tropical nights and summer days per year per city
    # 2m temperature is in Kelvin
    tropical_threshold = 30.0 + 273.15
    summer_day_threshold = 25.0 + 273.15
    temp_max_c = df_temp.reset_index()
    summer_days = temp_max_c.loc[
        (temp_max_c["2m_temperature"] > summer_day_threshold)
        & (temp_max_c["daytime"] == 1)
    ].reset_index()
    tropical_count = temp_max_c.loc[
        (temp_max_c["2m_temperature"] > tropical_threshold)
        & (temp_max_c["daytime"] == 0)
    ].reset_index()
    tropical_count = tropical_count.groupby(["city", "daytime"]).count()
    summer_days_count = summer_days.groupby(["city", "daytime"]).count()
    temp_max_c.to_csv("test_missing_cities_summer_days.csv")
    logger.info("Finished computing count")
    tropical_count.to_csv("test_missing_cities_tropical_nights.csv")
    summer_days_count.to_csv("test_missing_cities_summer_days.csv")
    logger.info("Finished saving results")

    measurer.end(
        tracker=tracker,
        shape=[],
        libraries=[
            v.__name__
            for k, v in globals().items()
            if type(v) is ModuleType and not k.startswith("__")
        ],
        variables=[],
        data_path=data_path,
        program_path=__file__,
        csv_file=f"./../../s3/data/l001_logs/benchmarks_2m_temperature_summer_days_tropical_nights_2018_test.csv",
    )
