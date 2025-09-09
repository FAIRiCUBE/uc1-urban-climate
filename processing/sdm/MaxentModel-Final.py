import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import text  # conection to the database
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import rioxarray as rxr
import elapid  # For Maxent
import shap  # For explainibility
import logging
import warnings
from types import ModuleType
from src.measurer import Measurer
from src import db_connect
from src import measurer
from src.utils import (
    plot_shap_detailed,
    plot_shap_global,
    presence_data_extraction,
    background_data_extraction,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("shap").setLevel(logging.ERROR)

print("✅ Libraries successfully loaded!")


def maxent_testing(features, labels):
    ## MAXENT (3.1.2)- Maxent that trains on: Part of the positives + all background data and tests on: the second part of positives + all background data
    # Step 1: Split presence data into training and testing sets
    presence_indices = np.where(labels == 1)[0]  # Indices for presence points

    # Split presence data
    presence_train_idx, presence_test_idx = train_test_split(
        presence_indices, test_size=0.2, random_state=42
    )

    # Combine presence training data with all background data for training
    train_indices = np.concatenate([presence_train_idx, np.where(labels == 0)[0]])
    X_train = features.iloc[train_indices]
    y_train = labels[train_indices]

    # Testing data: Combine test presence points with all background points
    test_indices = np.concatenate([presence_test_idx, np.where(labels == 0)[0]])
    X_test = features.iloc[test_indices]
    y_test = np.concatenate(
        [np.ones(len(presence_test_idx)), np.zeros(np.where(labels == 0)[0].shape[0])]
    )

    # Train the Maxent model
    maxent = elapid.MaxentModel()
    maxent.fit(X_train, y_train)

    # Predict suitability scores for the test set
    y_pred_prob = maxent.predict(X_test)

    return roc_auc_score(y_test, y_pred_prob)


############################################################################################## Main ###########################################################################

if __name__ == "__main__":
    measurer = Measurer()
    tracker = measurer.start(data_path="./s3/data/d012_luxembourg/")
    shape = []
    ######## Get data from database
    # TODO change to your own connection method
    engine_postgresql = db_connect.create_engine("./database_cinfig.ini")

    with engine_postgresql.connect() as connection:
        query = text(
            """
            SELECT *
            FROM luxembourg_species.neophytes_geometry
        """
        )
        occurrence_cube = pd.read_sql_query(query, connection)
    logging.info(f"🔌 Downloaded occurrence cube from database")

    logging.info("🌍 Starting data extraction...")
    ######################## Datacube extraction ######################

    base_path = "./s3/data/d012_luxembourg/"

    ## Datasets 01 SHADOW:------------------------------------------------------
    cube_01_L = rxr.open_rasterio(
        f"{base_path}/shadow_2019_10m_b1.tif", default_name="d01_L_light"
    ).squeeze()

    ### Dataset 02 WETNESS :----------------------------------------------------
    cube_02_F = rxr.open_rasterio(
        f"{base_path}/twi_2019_10m_b1.tif", default_name="d02_F_wetness"
    ).squeeze()

    ### Dataset 03 TEMPERATURE: monthly temp for 2017---------------------------
    cube_03_temperature_2017 = rxr.open_rasterio(
        os.path.join(base_path, "air_temperature_2017_month_mean_10m_b12.tif"),
        default_name="d03_T_parameter_2017",
    ).mean(dim="band")

    ### Dataset 05 Reaktionszahl (ph):-------------------------------------------
    cube_05_R = rxr.open_rasterio(
        f"{base_path}/pH_CaCl_10m_b1.tif", default_name="d05_R_ph"
    ).squeeze()

    ## ### Dataset 06 N:-------------------------------------------
    cube_06_N = rxr.open_rasterio(
        f"{base_path}/soil_nitrat_10m_b1.tif", default_name="d06_N_nitrogen"
    ).squeeze()

    ### Dataset 09 : Land cover :-----------------------------------
    cube_09_landcover = rxr.open_rasterio(
        f"{base_path}/land_cover_2021_10m_b1.tif", default_name="d09_LV_landcover"
    ).squeeze()

    # -- landcover_code	landcover_name LEGEND:
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

    ### Binary masks for Landcover
    # 1. Water areas
    d09_water_area = xr.where(cube_09_landcover == 60, 1, 0)  # Else set to 0
    d09_water_area = d09_water_area.rename("water_mask")

    # 2. Non sealed areas:
    d09_not_sealed = xr.where(
        cube_09_landcover.isin([30, 70, 71, 80, 91, 92, 93]), 1, 0
    )  # Else set to 0
    d09_not_sealed = d09_not_sealed.rename("not_sealed_mask")

    ##  Management buffer: area on in around roads, railways and water -buffered by 10m
    cube_10_not_management_buffer = rxr.open_rasterio(
        f"{base_path}/hip_b1_v2.tif", default_name="d10_not_management_buffer"
    ).squeeze()

    logging.info("🌍 Raster data extracted!")
    habitat_parameter_cube = xr.merge(
        [
            cube_01_L,
            cube_02_F,
            cube_03_temperature_2017,
            cube_05_R,
            cube_06_N,
            cube_09_landcover,
            d09_water_area,
            d09_not_sealed,
            cube_10_not_management_buffer,
        ]
    )  ### FULL CUBE with all paramter - resolution 10m

    ################################################################## Model #########################################################
    # Select species
    species_list = [
        "Robinia Pseudoacacia",
        "Fallopia Japonica",
        "Impatiens Glandulifera",
        "Heracleum Mantegazzianum",
    ]

    for species in species_list:
        logging.info(f"🌿 Processing {species}...")

        species_name = species.replace(" ", "_")
        species_occ_ds = occurrence_cube[occurrence_cube["species_name"] == species]

        x_coords = species_occ_ds["gridnum2169_10m_x"].values
        y_coords = species_occ_ds["gridnum2169_10m_y"].values
        # convert coordinates in xarray coordinates
        x_coords_da = xr.DataArray(x_coords)
        y_coords_da = xr.DataArray(y_coords)

        nearest_habitat_values = habitat_parameter_cube.sel(
            x=x_coords_da, y=y_coords_da, method="nearest"
        )

        # Presence data
        presence_data = presence_data_extraction(nearest_habitat_values)

        background_data = background_data_extraction(
            habitat_parameter_cube,
            nearest_habitat_values,
            len(presence_data),
            background_ratio=20,
        )

        # Combine presence and background data
        presence_labels = np.ones(len(presence_data))  # 1 for presence
        background_labels = np.zeros(len(background_data))  # 0 for background

        # Combine the data into one dataset
        combined_data = pd.concat([presence_data, background_data], ignore_index=True)
        labels = np.concatenate([presence_labels, background_labels])

        print(f"[Data Preparation] Combined dataset: {len(combined_data)} rows")
        print(f"  ├── Presence data: {len(presence_data)} rows")
        print(f"  ├── Background data: {len(background_data)} rows")

        # Select environmental variables (excluding species and coordinates)
        features = combined_data.drop(
            columns=[
                "longitude",
                "latitude",
                "band",
                "dim_0",
                "spatial_ref",
                "d10_not_management_buffer",
                "water_mask",
                "not_sealed_mask",
            ]
        )
        print(
            f"📊 Features ({len(features.columns)}):\n  - "
            + "\n  - ".join(features.columns)
        )

        print(
            f"\n[Training Maxent Model]: Training and evaluating suitability model for {species}..."
        )
        auc_score = maxent_testing(features, labels)
        logging.info(
            f"✔️ Maxent Model Evaluation Completed - AUC Score: {auc_score:.4f}"
        )

        ##################### Train on all for deployement ######################
        maxent = elapid.MaxentModel()
        maxent.fit(features, labels)

        ##################### Save sustainability map ###########################
        habitat_parameter_df = habitat_parameter_cube.where(
            (habitat_parameter_cube["water_mask"] == 0)
            & (habitat_parameter_cube["not_sealed_mask"] == 1)
        ).to_dataframe()
        habitat_parameter_df["sustainability"] = maxent.predict(
            habitat_parameter_df[
                [
                    "d01_L_light",
                    "d02_F_wetness",
                    "d03_T_parameter_2017",
                    "d05_R_ph",
                    "d06_N_nitrogen",
                    "d09_LV_landcover",
                ]
            ]
        )
        habitat_parameter_df = habitat_parameter_df[
            ~habitat_parameter_df.index.duplicated()
        ]
        sustainability_map = habitat_parameter_df.to_xarray()
        da_to_save = sustainability_map.sustainability
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        out_file = f"{base_path}habitat_potential_map/{species_name.lower()}/{timestamp}_{species_name.lower()}_maxent.tif"
        da_to_save.rio.to_raster(out_file)
        logging.info(f"Saved sustainability map to {out_file}")

        ##################### SHAP for explaning ################################
        # Predict function for Maxent suitability maps
        def predict_suitability(X):
            return maxent.predict(X).flatten()

        # Sampling
        sample_size = 100
        sample_features = shap.sample(features, sample_size, random_state=42)

        print(
            f"[SHAP Analysis] Running on {sample_size} sample points (Total dataset: {len(features)} rows)"
        )

        # Create explained
        explainer = shap.KernelExplainer(predict_suitability, features)

        # Generate SHAP values
        shap_values = explainer.shap_values(sample_features)

        # SHAP plots
        plot_shap_detailed(species, shap_values, sample_features)
        print(f"\n✅ SHAP detailed plot saved!")

        plot_shap_global(species, shap_values, sample_features)
        print(f"\n✅ Global SHAP plot saved!")

        logging.info(f"🌿 Processing {species} completed \n")

    print("\n✅ Process Complete - All species processed successfully!")
    measurer.end(
        tracker=tracker,
        shape=shape,
        libraries=[
            v.__name__
            for k, v in globals().items()
            if type(v) is ModuleType and not k.startswith("__")
        ],
        data_path="./s3/data/d012_luxembourg/",
        program_path=__file__,
        variables=locals(),
        csv_file="benchmarks_maxent.csv",
    )
