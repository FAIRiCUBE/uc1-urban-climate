from src import db_connect, measurer
import s3fs
import xarray as xr
import rioxarray as rxr
import pandas as pd
import numpy as np
import os
from sqlalchemy import text
from types import ModuleType
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

# read data cube from s3
def read_data_cube(base_path):

    s3_path = f"s3:///{os.environ.get('S3_USER_STORAGE_BUCKET')}{base_path}"
    s3fs_FS = s3fs.S3FileSystem(
        key=os.environ.get("S3_USER_STORAGE_KEY"),
        secret=os.environ.get("S3_USER_STORAGE_SECRET"),
    )
    ## Dataset 01 Lichtzahl:-------------------------------------------
    cube_01_L = rxr.open_rasterio(s3fs_FS.open(f"{s3_path}/shadow_2019_10m_b1.tif"))
    cube_01_L = cube_01_L.to_dataset(name="d01_L_light")

    ### Dataset 02 Feuchtezahl :---------------------------------------
    cube_02_F = rxr.open_rasterio(s3fs_FS.open(f"{s3_path}/twi_2019_10m_b1.tif"))
    cube_02_F = cube_02_F.to_dataset(name="d02_F_wetness")

    ### Dataset 03 Temperatur:-----------------------------------------
    ### monthly temperature for 2017
    cube_03_T = rxr.open_rasterio(
        s3fs_FS.open(f"{s3_path}/air_temperature_2017_month_mean_10m_b12.tif")
    )
    cube_03_T = cube_03_T.to_dataset(name="d03_T_temperatur")
    cube_03_T_stacked = xr.concat(
        [cube_03_T.sel(band=i) for i in range(1, 13)], dim="month"
    )
    ###Celsius = Kelvin - 273.15
    ## Calculate the maximum across all bands
    max_across_month = cube_03_T_stacked.max(dim="month") - 273.15
    max_across_month = max_across_month.rename(
        {"d03_T_temperatur": "d03_T_max_temperatur_2017_celsius"}
    )
    ## Calculate the min across all bands
    min_across_month = cube_03_T_stacked.min(dim="month") - 273.15
    min_across_month = min_across_month.rename(
        {"d03_T_temperatur": "d03_T_min_temperatur_2017_celsius"}
    )
    ## Calculate the avg across all bands
    avg_across_month = cube_03_T_stacked.mean(dim="month") - 273.15
    avg_across_month = avg_across_month.rename(
        {"d03_T_temperatur": "d03_T_avg_temperatur_2017_celsius"}
    )
    ### Merge three temperature min-max-avg datasets:
    cube_03_temperature_2017 = xr.merge(
        [max_across_month, min_across_month, avg_across_month]
    )

    ### Dataset 04 Kontinentaliätzahl:---------------------------------
    # TODO
    # cube_04_K = rxr.open_rasterio(s3fs_FS.open(f'{base_path}/.tif'))
    # cube_04_K = cube_04_K.to_dataset(name='d04_K_continentality')

    ### Dataset 05 Reaktionszahl (ph):---------------------------------
    cube_05_R = rxr.open_rasterio(s3fs_FS.open(f"{s3_path}/pH_CaCl_10m_b1.tif"))
    cube_05_R = cube_05_R.to_dataset(name="d05_R_ph")

    ## ### Dataset 06 Stickstoff:--------------------------------------
    cube_06_N = rxr.open_rasterio(s3fs_FS.open(f"{s3_path}/soil_nitrat_10m_b1.tif"))
    cube_06_N = cube_06_N.to_dataset(name="d06_N_nitrogen")  ##

    ### Dataset 09 Lebensform:-----------------------------------------
    cube_09_temp_LF = rxr.open_rasterio(
        s3fs_FS.open(f"{s3_path}/land_cover_2021_10m_b1.tif")
    )
    cube_09_temp_LF = cube_09_temp_LF.to_dataset(name="d09_LV_landcover")

    # -- landcover_code	landcover_name
    # -- 10	buildings
    # -- 20	other constructed areas
    # -- 30	bare soil
    # -- 60	water
    # -- 70	trees
    # -- 71	dead trees
    # -- 80	bushes
    # -- 91	permanent herbaceous vegetation
    # -- 92	seasonal herbaceous vegetation
    # -- 93	vineyards

    d09_LF_parameter_temp_water_area = xr.where(
        cube_09_temp_LF["d09_LV_landcover"] == 60, 1, 0
    )  # Else set to 0
    # Adding the result back to the dataset (optional)
    cube_09_temp_LF["ellenberg_water_area"] = d09_LF_parameter_temp_water_area

    cube_09_LF_x = cube_09_temp_LF["ellenberg_water_area"]
    cube_09_1_LF_water = cube_09_LF_x.to_dataset(name="ellenberg_water_area")

    d09_LF_parameter_temp_not_sealed = xr.where(
        cube_09_temp_LF["d09_LV_landcover"].isin([30, 70, 71, 80, 91, 92, 93]), 1, 0
    )  # Else set to 0
    # Adding the result back to the dataset (optional)
    cube_09_temp_LF["ellenberg_not_sealed_area"] = d09_LF_parameter_temp_not_sealed
    cube_09_LF_x_non_sealed = cube_09_temp_LF["ellenberg_not_sealed_area"]
    cube_09_2_LF_non_sealed = cube_09_LF_x_non_sealed.to_dataset(
        name="ellenberg_not_sealed_area"
    )

    habitat_parameter_cube = xr.merge(
        [
            cube_01_L,
            cube_02_F,
            cube_03_temperature_2017,
            cube_05_R,
            cube_06_N,
            cube_09_temp_LF,
            cube_09_1_LF_water,
            cube_09_2_LF_non_sealed,
        ]
    )

    return habitat_parameter_cube


def get_occurrence_data(species_name):
    config_path = os.environ.get("HOME") + "/uc1-urban-climate/database.ini"
    engine_postgresql = db_connect.create_engine(config_path)
    with engine_postgresql.begin() as conn:
        query = text(
            """    
                  SELECT gbif_key, species_name, sample_date, x_epsg2169, y_epsg2169 
                  FROM luxembourg_species.neophytes_geometry 
                  WHERE LOWER(REPLACE(species_name, ' ', '_')) = :species_name
                AND sample_date > '2010-01-01'
        """
        )
        df = pd.read_sql_query(query, conn, params={"species_name": species_name})
    # query data cube -- list of points
    x_coords = xr.DataArray(df.x_epsg2169)
    y_coords = xr.DataArray(df.y_epsg2169)
    return x_coords, y_coords


if __name__ == "__main__":

    species_name = "robinia_pseudoacacia"
    base_path = "/data/d012_luxembourg"
    test_size = 0.2

    # # initialize measurer
    m_instance = measurer.Measurer()
    tracker = m_instance.start(data_path=os.environ.get("HOME") + "/s3" + base_path)
    print("started measurer")
    x_occurrence, y_occurrence = get_occurrence_data(species_name)
    print("loaded occurrence data")

    habitat_cube = read_data_cube(base_path)
    habitat_cube = habitat_cube.squeeze()
    print("loaded cube")
    # Create the mask: where 'ellenberg_not_sealed_area' == 1
    mask = (habitat_cube["ellenberg_not_sealed_area"] != 1) | (
        habitat_cube["ellenberg_water_area"] == 1
    )

    habitat_cube = habitat_cube.where(~mask, np.nan)

    # Use gbif occurence data and extract nearest habitat value with sel
    habitat_occurrence = habitat_cube.sel(
        y=y_occurrence, x=x_occurrence, method="nearest"
    )

    habitat_occurrence_df = habitat_occurrence.to_dataframe()
    habitat_occurrence_df["occurrence"] = True
    habitat_occurrence_df.dropna(inplace=True)
    n_occurrence = habitat_occurrence_df.shape[0]
    print(habitat_occurrence_df.shape)
    # Select no occurrence data
    habitat_background = habitat_cube.drop_sel(
        y=habitat_occurrence.y, x=habitat_occurrence.x
    )

    habitat_background_sample = habitat_background.drop_indexes(["x", "y"]).stack(
        sample=("x", "y")
    )
    habitat_background_sample = habitat_background_sample.isel(
        sample=sorted(
            np.random.randint(
                0, habitat_background_sample.sample.shape, 10 * n_occurrence
            )
        )
    )

    habitat_background_sample_df = habitat_background_sample.to_dataframe()
    habitat_background_sample_df["occurrence"] = False
    habitat_background_sample_df.dropna(inplace=True)
    print(habitat_background_sample_df.shape)
    # concatenate samples
    df_sampled = pd.concat([habitat_occurrence_df, habitat_background_sample_df])
    print("loaded training data")
    print(df_sampled.shape)
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    # Sample for classification model from equally distributed pre sampled data
    train, test = train_test_split(df_sampled, test_size=test_size, random_state=3)

    feature_columns = [
        "d01_L_light",
        "d02_F_wetness",
        "d03_T_max_temperatur_2017_celsius",
        #'d03_T_min_temperatur_2017_celsius',
        #'d03_T_avg_temperatur_2017_celsius',
        # "d04_K_continentality",
        "d05_R_ph",
        "d06_N_nitrogen",
        "d09_LV_landcover",
        # "ellenberg_water_area",
        # "ellenberg_not_sealed_area",
    ]
    # Initialize the model
    model = RandomForestClassifier()
    # model = xgb.XGBRegressor(objective="reg:squarederror")

    # Fit the model
    model.fit(train[feature_columns], train["occurrence"])

    # Make predictions
    pred = model.predict(test[feature_columns])

    # Evaluate the model
    class_report = classification_report(
        test["occurrence"], pred, zero_division=0, output_dict=True
    )
    print(classification_report(test["occurrence"], pred, zero_division=0))

    # use the model on the complete dataset
    df_habitat_cube = habitat_cube.to_dataframe().dropna()
    pred = model.predict(df_habitat_cube[feature_columns])
    df_habitat_cube["prediction"] = pred
    # convert to xarray DataArray
    # use sortby to make continuous x,y axis
    habitat_prediction = df_habitat_cube["prediction"].to_xarray().sortby("x")

    ## save to file
    # create folder for each species
    if not os.path.exists(
        os.environ.get("HOME") + f"/s3{base_path}/habitat_potential_map/{species_name}"
    ):
        os.mkdir(
            os.environ.get("HOME")
            + f"/s3{base_path}/habitat_potential_map/{species_name}"
        )

    habitat_prediction.attrs = habitat_cube.spatial_ref.attrs
    habitat_prediction.rio.write_crs(2169, inplace=True)
    habitat_prediction = habitat_prediction.where(habitat_prediction == 1, np.nan)
    habitat_prediction.rio.to_raster(
        os.environ.get("HOME")
        + f"/s3{base_path}/habitat_potential_map/{species_name}/{species_name}_RandomForest_model.tif"
    )

    # also save classification report
    df_report = pd.DataFrame(class_report).transpose()
    df_report.to_csv(
        os.environ.get("HOME")
        + f"/s3{base_path}/habitat_potential_map/{species_name}/{species_name}_RandomForest_model_report.csv"
    )

    m_instance.end(
        tracker=tracker,
        shape=[],
        libraries=[
            v.__name__
            for k, v in globals().items()
            if type(v) is ModuleType and not k.startswith("__")
        ],
        data_path=os.environ.get("HOME") + "/s3" + base_path,
        program_path=__file__,
        variables=locals(),
        csv_file=os.environ.get("HOME")
        + "/uc1-urban-climate/processing/sdm/logs/RandomForest_model.csv",
    )
