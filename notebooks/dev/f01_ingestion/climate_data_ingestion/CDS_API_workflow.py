import cdstoolbox as ct

# specify how to get data -> download
@ct.output.download()

# processing routine
def get_daily_utci_stats(lonlat=[[-1,60]], day_or_night="night"):
    """get daily (daytime or nighttime) mean, min and max of utci (universal thermal climate index) 
    at given locations (lon, lat coordinates). 
    Source dataset: Thermal comfort indices derived from ERA5 reanalysis
    More info about dataset: https://cds.climate.copernicus.eu/cdsapp#!/dataset/derived-utci-historical?tab=overview

    Args:
        lonlat (list, optional): List of [lon,lat] coordinates (WGS84) at which statistics are computed. Defaults to [[-1,60]].
        day_or_night (str, optional): whether to compute statistics for daytime or nighttime. Defaults to "night".
    """
    if(day_or_night == "night"):
        hours = ['18:00', '19:00', '20:00', '21:00', '22:00', '23:00', 
                 '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00']
    else:
        hours = ['07:00', '08:00', '09:00', '10:00', '11:00', '12:00',
                '13:00', '14:00', '15:00', '16:00', '17:00', '18:00']
        
    data = ct.catalogue.retrieve(
        'derived-utci-historical', 
    {
        'version': '1_1',
        'day': [
            '01', '02',
        ],
        'month': [
            '01',
        ],
        'year': '2023',
        'time': hours,
        'product_type': 'consolidated_dataset',
        'variable': 'universal_thermal_climate_index',
    })
    data_list = []
    city_list = []
    for code,lon,lat in lonlat:
        data_location = ct.cube.interpolate(data, lon=lon, lat=lat)
        daily_mean = ct.climate.daily_mean(data_location)
        daily_min = ct.climate.daily_min(data_location)
        daily_max = ct.climate.daily_max(data_location)
        daily_data = ct.cube.concat([daily_mean, daily_min, daily_max], dim={'statistic': ["mean", "min", "max"]})
        data_list.append(daily_data)
        city_list.append(code)
    return_data = ct.cube.concat(data_list, dim={"city_lonlat": city_list})
    return return_data