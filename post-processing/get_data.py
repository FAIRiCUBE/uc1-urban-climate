# extract city features collection from database and save to geojson
import pkg_resources
import os
from sqlalchemy import text
import pandas as pd
import geopandas as gpd

if __name__ == "__main__":

    required = {'src'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    if missing:
        print('src package is not installed')
    else:
        from src import db_connect
        home_dir = os.environ.get('HOME') # works only in EOX Hub
        engine_postgresql = db_connect.create_engine(
            db_config=f"{home_dir}/uc1-urban-climate/database.ini")

        with engine_postgresql.begin() as conn:
            query = text("""SELECT 
                        public.city_2018_demo_view.urau_code, public.city_2018_demo_view.urau_name, 
                        public.city_2018_demo_view._wgs84x, public.city_2018_demo_view._wgs84y, 
                        ez_code, city_area_ha, dem_mean,
                        imd_percent_2018, treecover_percent_2018,
                        class_11100, class_11210, class_11220, class_11230,
                        class_11240, class_11300, class_12100, class_12210,
                        class_12220, class_12230, class_12300, class_12400,
                        class_13100, class_13300, class_13400, class_14100,
                        class_14200, class_21000, class_22000, class_23000,
                        class_24000, class_25000, class_31000, class_32000,
                        class_33000, class_40000, class_50000, urban_blue_percent,
                        urban_green_percent, avg_2m_temp_kelvin_2018,
                        number_of_summer_days_2018, number_of_tropical_nights_2018,
                        utci_heat_nights_2018, coastal_city, de1001v_2018, de1028v_2018,
                        de1055v_2018, ec1174v_2018, ec1010v_2018, ec1020i_2018,
                        ec3040v_2018, sa2013v_2018, de1028i_2018, de1055i_2018, lutua.geometry as geometry
                        FROM public.city_2018_demo_view
                        LEFT JOIN lut.l_city_urau2021 lutua ON public.city_2018_demo_view.urau_code = lutua.urau_code;""")
            df = gpd.GeoDataFrame.from_postgis(query, conn, geom_col='geometry')
        df.to_file(
            f"{home_dir}/uc1-urban-climate/data/city_features_collection/city_features_collection_v0.1.geojson", driver='GeoJSON')
