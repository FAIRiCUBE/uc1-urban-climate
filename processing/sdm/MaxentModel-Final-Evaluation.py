import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import text  # conection to the database
import os
import glob
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sys
import elapid  # For Maxent
from elapid import GeographicKFold
import geopandas as gpd
from scipy.stats import spearmanr

project_root = os.path.abspath("../..")
sys.path.append(project_root)
from src import db_connect
from src import measurer
import shap  # For explainibility
import logging
import warnings
from types import ModuleType
from src import db_connect
from src.measurer import Measurer
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


def continuous_boyce_index(pred_presence, pred_background, bins=10):
    # Bin edges
    bin_edges = np.linspace(np.min(pred_background), np.max(pred_background), bins + 1)

    # Mid-points of bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Count presence and background frequencies in bins
    pres_hist, _ = np.histogram(pred_presence, bins=bin_edges)
    back_hist, _ = np.histogram(pred_background, bins=bin_edges)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        pe_ratio = pres_hist / back_hist
        pe_ratio = np.where(back_hist == 0, np.nan, pe_ratio)

    # Remove NaNs for correlation
    valid = ~np.isnan(pe_ratio)
    if np.sum(valid) < 2:
        return np.nan  # Not enough valid bins

    # Spearman correlation
    cbi, _ = spearmanr(bin_centers[valid], pe_ratio[valid])

    return cbi


def maxent_testing(features, labels, presence_data, background_data):
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
    print(f"Train size= {len(X_train)}")
    print(f"Test size = {len(X_test)}")

    # Train the Maxent model
    maxent = elapid.MaxentModel()
    maxent.fit(X_train, y_train)

    # Predict suitability scores for the test set
    y_pred_prob = maxent.predict(X_test)

    pred_presence = maxent.predict(presence_data)
    pred_background = maxent.predict(background_data)

    auc = roc_auc_score(y_test, y_pred_prob)
    cbi = continuous_boyce_index(pred_presence, pred_background, bins=10)

    return (auc, cbi)


def evaluate_maxent_checkerboard(
    presence_data, background_data, features, grid_size=5000
):
    """
    Evaluates MaxEnt model using checkerboard spatial split.
    """

    merged = elapid.stack_geodataframes(
        presence_data, background_data, add_class_label=True
    )
    train, test = elapid.checkerboard_split(merged, grid_size=grid_size)

    xtrain = train[features.columns]
    ytrain = train["class"]
    xtest = test[features.columns]
    ytest = test["class"]
    print(f"Training Size= {len(xtrain)}")
    print(f"Testing Size= {len(xtest)}")
    model = elapid.MaxentModel()
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    auc = roc_auc_score(ytest, ypred)

    pred_presence = model.predict(presence_data)
    pred_background = model.predict(background_data)

    cbi = continuous_boyce_index(pred_presence, pred_background, bins=10)

    return (auc, cbi)


def evaluate_maxent_geographic_kfold(presence, background, features, n_splits=3):
    """
    Evaluates MaxEnt model using geographic k-fold cross-validation.
    """

    auc_scores = []
    cbi_scores = []
    gfolds = GeographicKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(gfolds.split(presence)):
        p_train = presence.iloc[train_idx]
        p_test = presence.iloc[test_idx]

        b_train = background.sample(n=len(background) // 2, random_state=fold)
        b_test = background.drop(b_train.index)

        train = elapid.stack_geodataframes(p_train, b_train)
        test = elapid.stack_geodataframes(p_test, b_test)

        xtrain = train[features.columns]
        # xtrain = train.drop(columns=['class'])
        ytrain = train["class"]
        # xtest = test.drop(columns=['class'])
        xtest = test[features.columns]
        ytest = test["class"]
        model = elapid.MaxentModel()
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)

        auc = roc_auc_score(ytest, ypred)
        auc_scores.append(auc)
        print(f"Fold {fold+1} AUC: {auc:.4f}")
        pred_presence = model.predict(presence_data)
        pred_background = model.predict(background_data)

        cbi = continuous_boyce_index(pred_presence, pred_background, bins=10)
        cbi_scores.append(cbi)
        print(f"Fold {fold+1} CBI: {cbi:.4f}")

    mean_auc = np.mean(auc_scores)
    mean_cbi = np.mean(cbi_scores)

    return (mean_auc, mean_cbi)


def evaluate_maxent_buffered_loocv(presence, background, features, distance=5000):
    """
    Evaluates MaxEnt model using buffered leave-one-out cross-validation.
    """

    merged = elapid.stack_geodataframes(presence, background, add_class_label=True)
    # merged = ela.annotate(merged, features.columns.tolist(), drop_na=True, quiet=True)

    bloo = elapid.BufferedLeaveOneOut(distance=distance)
    yobs_scores = []
    ypred_scores = []

    model = elapid.MaxentModel()

    for train_idx, test_idx in bloo.split(merged, class_label="class"):
        train = merged.iloc[train_idx]
        test = merged.iloc[test_idx]

        xtrain = train[features.columns]
        ytrain = train["class"]
        xtest = test[features.columns]
        ytest = test["class"]

        if len(np.unique(ytrain)) < 2:
            continue  # Skip this fold if only one class in training set

        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)

        ypred_scores.append(ypred[0])
        yobs_scores.append(ytest.values[0])

    # Predict background for full AUC calc
    bg = merged[merged["class"] == 0]
    xbg = bg[features.columns]
    ybg = bg["class"].to_numpy()
    bg_pred = model.predict(xbg)

    ypred_all = np.concatenate([bg_pred, ypred_scores])
    yobs_all = np.concatenate([ybg, yobs_scores])

    auc = roc_auc_score(yobs_all, ypred_all)
    pred_presence = model.predict(presence)
    pred_background = model.predict(background)

    cbi = continuous_boyce_index(pred_presence, pred_background, bins=10)

    return (auc, cbi)


# ############################################################################################# Main ###########################################################################


def to_gdf(df):
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["latitude"], df["longitude"]),
        crs="EPSG:4326",
    )


if __name__ == "__main__":
    measurer = Measurer()
    tracker = measurer.start(data_path="./s3/data/d012_luxembourg/")
    shape = []
    results_path = "maxent_model_evaluation_results.csv"
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

    ## For masking of the final data we need an extra cube:
    c_8 = cube_09_1_LF_water
    c_9 = cube_09_2_LF_non_sealed

    sealed_water_cube = xr.merge([c_8, c_9])
    # Create the list of variables and names

    parameters = [
        habitat_parameter_cube.d01_L_light.sel(band=1),
        habitat_parameter_cube.d02_F_wetness.sel(band=1),
        habitat_parameter_cube.d03_T_parameter_2017,
        habitat_parameter_cube.d05_R_ph.sel(band=1),
        habitat_parameter_cube.d06_N_nitrogen.sel(band=1),
        habitat_parameter_cube.d09_LV_landcover.sel(band=1),
        habitat_parameter_cube.ellenberg_water_area.sel(band=1),
        habitat_parameter_cube.ellenberg_not_sealed_area.sel(band=1),
    ]

    plot_variables(parameters)

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
                species,
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
        print(
            f"\n 1. [Trains on: (80% of positives + all background). Tests on: 20% of positives + all background])"
        )
        (auc1, cbi1) = maxent_testing(features, labels, presence_data, background_data)
        print(f"Precence only split AUC Score: {auc1:.4f}")
        print(f"Precence only split CBI = {cbi1}")

        presence_data = to_gdf(presence_data)
        background_data = to_gdf(background_data)

        grid_size = 5000
        print(
            f"\n 2. [Training and testing using checkerboard spatial split]: alternating grid cells are used for training and testing to ensure spatial separation (Grid_size = {grid_size})."
        )

        (auc2, cbi2) = evaluate_maxent_checkerboard(
            presence_data, background_data, features, grid_size=grid_size
        )
        print(f"Checkerboard split (Grid_size = {grid_size}) AUC = {auc1}")
        print(f"Checkerboard split (Grid_size = {grid_size}) CBI = {cbi2}")

        n_splits = 3
        print(
            f"\n 3. [Training and testing using geographic k-fold cross-validation]: presence data is split into spatial clusters to evaluate model generalization across regions (K = {n_splits})."
        )
        (auc3, cbi3) = evaluate_maxent_geographic_kfold(
            presence_data, background_data, features, n_splits=n_splits
        )
        print(f"GeographicKFold mean (K = {n_splits}) AUC = {auc3}")
        print(f"GeographicKFold mean (K = {n_splits}) CBI = {cbi3}")

        distance = 5000
        print(
            f"\n 4. [Training and testing using buffered leave-one-out cross-validation]: presence points are iteratively held out, excluding nearby training points (buffered), to evaluate model generalization far from training areas (Distance = {distance})."
        )
        (auc4, cbi4) = evaluate_maxent_buffered_loocv(
            presence_data, background_data, features, distance=distance
        )
        print(f"Buffered Leave-One-Out (Distance = {distance}) AUC: {auc4:.4f}")
        print(f"Buffered Leave-One-Out (Distance = {distance}) CBI: {cbi4:.4f}")

        # Collect results in a list
        if "evaluation_results" not in locals():
            evaluation_results = []

        evaluation_results.append(
            {
                "Species": species,
                "Split": "Presence-only",
                "Config": "80/20",
                "AUC": round(auc1, 4),
                "CBI": round(cbi1, 4),
            }
        )
        evaluation_results.append(
            {
                "Species": species,
                "Split": "Checkerboard",
                "Config": f"Grid: {grid_size//1000} km",
                "AUC": round(auc2, 4),
                "CBI": round(cbi2, 4),
            }
        )
        evaluation_results.append(
            {
                "Species": species,
                "Split": "GeoKFold",
                "Config": f"k={n_splits}",
                "AUC": round(auc3, 4),
                "CBI": round(cbi3, 4),
            }
        )
        evaluation_results.append(
            {
                "Species": species,
                "Split": "Buffered LOO",
                "Config": f"Buffer: {distance//1000} km",
                "AUC": round(auc4, 4),
                "CBI": round(cbi4, 4),
            }
        )

        print(f"✔️ Maxent Model Evaluation Completed")

        ##################### Train on all for deployement ######################
        maxent = elapid.MaxentModel()
        maxent.fit(features, labels)

        ############################### SHAP for explaning #####################
        # Predict function for Maxent suitability maps
        def predict_suitability(X):
            return maxent.predict(X).flatten()

        # Sampling
        sample_size = 1
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

        if os.path.exists(results_path):
            existing_df = pd.read_csv(results_path)
            results_df = pd.concat(
                [existing_df, pd.DataFrame(evaluation_results)], ignore_index=True
            )
        else:
            results_df = pd.DataFrame(evaluation_results)

    results_df.to_csv(results_path, index=False)

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
