import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import create_engine, text # conection to the database
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import rioxarray as rxr
import matplotlib.patches as mpatches
import sys

project_root = os.path.abspath("../..")
sys.path.append(project_root)
from src import db_connect
from src import measurer
import elapid # For Maxent
import shap # For explainibility 
import logging
import warnings
from src.measurer import Measurer  # Correct import
from types import ModuleType

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("shap").setLevel(logging.ERROR)


print("✅ Libraries successfully loaded!")



########################################### FUNCTIONS ######################################
def load_raster(base_path, file_name, name):
    var = os.path.join(base_path, file_name)
    cube = rxr.open_rasterio(var)
    os.path.join(base_path, file_name)
    return cube.to_dataset(name=name)

def plot_variables(parameters):
    # Plot all variable
    parameter_names = [
        "occurence_data", 
        'L_light', 
        'F_wetness',
        'T_temperature', 
        'R_ph',
        'N_nitrogen', 
        'water_area', 
        'not_sealed_area'
    ]
    plt.figure(figsize=(15, 20))
    for i, param in enumerate(parameters):
        plt.subplot(4, 3, i + 1)  # Create a grid of subplots
        param.plot() 
        plt.title(parameter_names[i])
    plt.tight_layout
    plt.savefig(f"../../images/variable_maxent_plot.png")
    plt.close


def presence_data_extraction(nearest_habitat_values):
    presence_habitat_df = nearest_habitat_values.to_dataframe().reset_index()
    presence_habitat_df[species] = True
    presence_habitat_df = presence_habitat_df.rename(columns={'x': 'longitude', 'y': 'latitude'})
    
    presence_habitat_df = presence_habitat_df.replace(-9999, np.nan)

    presence_habitat_df =  presence_habitat_df.dropna()
    
    return presence_habitat_df

    
def background_data_extraction(habitat_parameter_cube, nearest_habitat_values, presence_size, background_ratio = 20):
    ## reading the FULL CUBE for Luxembourg, set to na points where no plant growth is possible (water & sealed areas)
    background_cube =  habitat_parameter_cube.sel(band=1).where((habitat_parameter_cube['ellenberg_water_area'] ==0) & (habitat_parameter_cube['ellenberg_not_sealed_area'] == 1))

    ## set to na points where species has occurred
    # get coordinates from occurrence cube
    x_coords_grid = list(nearest_habitat_values.x.values)
    y_coords_grid = list(nearest_habitat_values.y.values)

    # Create a boolean mask for species occurrences
    mask = background_cube.assign(mask=lambda x: (x.d01_L_light * 0 + 1).astype(bool)).drop_vars(background_cube.keys())
    mask.mask.loc[dict(x=x_coords_grid, y=y_coords_grid)] = False
    # set locations of species occurrence to na
    background_habitat_values = background_cube.where(mask.mask)
    # Step 4: Convert the non-occurrence habitat data to a DataFrame and remove masked values (all na)
    background_habitat_df = background_habitat_values.to_dataframe().reset_index().dropna()
 
    feature_columns = background_habitat_df.columns.drop(['y', 'x', 'band'])

    # Randomly sample background points after filtering
    target_bg_size = min(len(background_habitat_df), background_ratio * presence_size)

    background_habitat_df = background_habitat_df.sample(n=target_bg_size, random_state=42)
    
    # Step 5: Mark these samples as "False" for species presence
    background_habitat_df[species] = False 

    
    ## MAXENT: data preparation
    background_data = background_habitat_df.rename(columns={'x': 'longitude', 'y': 'latitude'})
    
    
    # Replace -9999 by NaN
    background_data = background_data.replace(-9999, np.nan)
    
    # Drop rows with NaN values
    background_data = background_data.dropna()
    
    return background_data
    

    
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
    y_test = np.concatenate([np.ones(len(presence_test_idx)), np.zeros(np.where(labels == 0)[0].shape[0])])
    
    # Train the Maxent model
    maxent = elapid.MaxentModel()
    maxent.fit(X_train, y_train)

    # Predict suitability scores for the test set
    y_pred_prob = maxent.predict(X_test)
    
    return roc_auc_score(y_test, y_pred_prob)


def plot_shap_detailed(species, shap_values, sample_features):
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample_features, show=False)
    plt.title("SHAP Summary Plot for Maxent ("+species +")")
    plt.tight_layout()
    plt.savefig(f"../../images/shapMaxent/Shap_Detailed_{species}_{len(sample_features)}.png", dpi=300)
    plt.close()


    
    
def plot_shap_global(species, shap_values, sample_features):
    
    # Compute global SHAP values using the sampled dataset
    global_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Compute correlation between sampled feature values and SHAP values
    feature_shap_correlation = []
    for i, feature in enumerate(sample_features.columns):
        correlation = np.corrcoef(sample_features[feature], shap_values[:, i])[0, 1]  # Pearson correlation
        feature_shap_correlation.append(correlation)

    # Create a DataFrame with feature importance and correlation
    feature_importance_df = pd.DataFrame({
        'Feature': sample_features.columns,  # Use sampled features
        'Mean_Abs_SHAP': global_shap_values,
        'Correlation': feature_shap_correlation
    })

    feature_importance_df = feature_importance_df.sort_values(by='Mean_Abs_SHAP', ascending=False)

    sorted_features = feature_importance_df['Feature'].values
    sorted_shap_values = feature_importance_df['Mean_Abs_SHAP'].values
    sorted_correlation_values = feature_importance_df['Correlation'].values

    colors = ['green' if corr > 0 else 'orange' for corr in sorted_correlation_values]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_features, sorted_shap_values, color=colors)  # Bars now sorted correctly
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title(f'Global Feature Importance for Maxent Suitability ({species})')

    for bar, corr in zip(bars, sorted_correlation_values):
        plt.text(
            bar.get_width() / 2,  # Position in the middle of the bar
            bar.get_y() + bar.get_height()/2,  # Centered in the bar
            f"{corr:.2f}",  # Format to 2 decimal places
            va='center', ha='center',  # Center text inside bar
            fontsize=10, 
            color='white' if abs(corr) > 0 else 'black',  # Improve readability
            fontweight='bold'
        )

    legend_patches = [
        mpatches.Patch(color='green', label='Higher values increase suitability'),
        mpatches.Patch(color='orange', label='Lower values increase suitability')
    ]
    plt.legend(handles=legend_patches, loc='lower right')

    plt.gca().invert_yaxis()  # Ensure most important feature remains on top
    plt.tight_layout()

    plt.savefig(f"../../images/shapMaxent/Shap_Global_{species}_{sample_size}.png", dpi=300)
    plt.close()
    
############################################################################################## Main ###########################################################################





if __name__ == "__main__":
    measurer = Measurer()
    tracker = measurer.start(data_path=os.environ.get("HOME") +"/s3/data/d012_luxembourg/")
    shape = []
    ######## connect to DATABASE server: 
    database_config_path = glob.glob(os.environ.get("HOME")+'/database*.ini')[0]
    keys = db_connect.config(filename=database_config_path)
    POSTGRESQL_SERVER_NAME=keys['host']
    PORT=                  keys['port']
    Database_name =        keys['database']
    USER =                 keys['user']
    PSW =                  keys['password']

    engine_postgresql = create_engine('postgresql://'+USER+':'+PSW+ '@'+POSTGRESQL_SERVER_NAME+':'+str(PORT)+ '/' + Database_name)
    connection = engine_postgresql.raw_connection()
    cursor = connection.cursor()
    connection.commit()
    print(f"🔌 Connecting to database...")

    logging.info("🌍 Starting data extraction...")
    ######################## Datacube extraction ######################

    base_path = os.environ.get("HOME") +"/s3/data/d012_luxembourg/"

    ## Datasets 01 SHADOW:------------------------------------------------------
    cube_01_L = load_raster(base_path, 'shadow_2019_10m_b1.tif', 'd01_L_light') 

    ### Dataset 02 WETNESS :----------------------------------------------------
    cube_02_F = load_raster(base_path, 'twi_2019_10m_b1.tif', 'd02_F_wetness')

    ### Dataset 03 TEMPERATURE: monthly temp for 2017---------------------------
    cube_03_temperature_2017 = load_raster(base_path, 'air_temperature_2017_month_mean_10m_b12.tif', 'd03_T_parameter_2017').mean(dim='band')

    ### Dataset 05 Reaktionszahl (ph):-------------------------------------------
    cube_05_R = load_raster(base_path, 'pH_CaCl_10m_b1.tif', 'd05_R_ph')

    ## ### Dataset 06 N:-------------------------------------------
    cube_06_N = load_raster(base_path, 'soil_nitrat_10m_b1.tif', 'd06_N_nitrogen')

    ### Dataset 09 : Land cover -water surface:-----------------------------------
    cube_09__temp_LF = load_raster(base_path, 'land_cover_2021_10m_b1.tif', 'd09_LV_landcover')

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
    #1. Water areas
    ds = cube_09__temp_LF
    d09_LF_parameter_temp_water_area =    xr.where(ds['d09_LV_landcover'] == 60, 1, 0) # Else set to 0
    cube_09__temp_LF['ellenberg_water_area'] = d09_LF_parameter_temp_water_area
    cube_09_LF_x = cube_09__temp_LF['ellenberg_water_area'] 
    cube_09_1_LF_water = cube_09_LF_x.to_dataset(name='ellenberg_water_area')

    #2. Non sealed areas:
    d09_LF_parameter_temp_not_sealed =    xr.where(ds['d09_LV_landcover'].isin ([30,70,71,80,91,92,93]), 1, 0) # Else set to 0
    cube_09__temp_LF['ellenberg_not_sealed_area'] = d09_LF_parameter_temp_not_sealed
    cube_09_LF_x_non_sealed = cube_09__temp_LF['ellenberg_not_sealed_area'] 
    cube_09_2_LF_non_sealed = cube_09_LF_x_non_sealed.to_dataset(name='ellenberg_not_sealed_area')

    ##  Management buffer: area on in around roads, railways and water -buffered by 10m
    cube_10_not_maagement_buffer = load_raster(base_path, 'hip_b1_v2.tif', 'd10_not_management_buffer')
    logging.info("🌍\ Raster data extracted!")

    # the following code set up ONE datacube with the raw data - (only to be used for model -tuning in the moment)
    c_1 = cube_01_L 
    c_2 = cube_02_F 
    c_3 = cube_03_temperature_2017 
    c_5 = cube_05_R
    c_6 = cube_06_N 
    c_7 = cube_09__temp_LF 
    c_8 = cube_09_1_LF_water 
    c_9 = cube_09_2_LF_non_sealed 
    c_10=cube_10_not_maagement_buffer

    habitat_parameter_cube = xr.merge([c_1, c_2, c_3, c_5, c_6,c_7, c_8, c_9,c_10])  ### FULL CUBE with all paramter - resolution 10m 

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
        habitat_parameter_cube.ellenberg_not_sealed_area.sel(band=1)
    ]



    plot_variables(parameters)


    ################################################################## Model #########################################################    
    # Select species
    species_list = [
    'Robinia Pseudoacacia', 
    'Fallopia Japonica', 
    'Impatiens Glandulifera', 
    'Heracleum Mantegazzianum'       
    ]   


    for species in species_list:
        logging.info(f"🌿 Processing {species}...")

        species_name = species.replace(" ", "_")

        with engine_postgresql.connect() as connection:
            query = text("""
                SELECT *
                FROM luxembourg_species.neophytes_geometry
            """)
            species_occ_df = pd.read_sql_query(query, connection)

        species_occ_df = species_occ_df[species_occ_df['species_name']==species]

        x_coords = species_occ_df["gridnum2169_10m_x"].values
        y_coords = species_occ_df["gridnum2169_10m_y"].values
        # convert coordinates in xarray coordinates
        x_coords_da = xr.DataArray(x_coords)
        y_coords_da = xr.DataArray(y_coords)

        nearest_habitat_values = habitat_parameter_cube.sel(
            x=x_coords_da,
            y=y_coords_da,
            method="nearest"
        )


        # Presence data
        presence_data = presence_data_extraction(nearest_habitat_values)

        background_data = background_data_extraction(habitat_parameter_cube, nearest_habitat_values, len(presence_data), background_ratio = 20)

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
        features = combined_data.drop(columns=[species, 'longitude', 'latitude', 'band', 'dim_0', 'spatial_ref', 'd10_not_management_buffer', 'ellenberg_water_area', 'ellenberg_not_sealed_area'])
        print(f"📊 Features ({len(features.columns)}):\n  - " + "\n  - ".join(features.columns))


        print(f"\n[Training Maxent Model] Training and evaluating suitability model for {species}...")
        auc_score = maxent_testing(features, labels)
        print(f"✔️ Maxent Model Evaluation Completed - AUC Score: {auc_score:.4f}")


        ##################### Train on all for deployement ######################
        maxent = elapid.MaxentModel()
        maxent.fit(features, labels)

        ############################### SHAP for explaning #####################
        # Predict function for Maxent suitability maps 
        def predict_suitability(X):
            return maxent.predict(X).flatten()

        # Sampling
        sample_size = 100
        sample_features = shap.sample(features, sample_size, random_state=42)
        
        print(f"[SHAP Analysis] Running on {sample_size} sample points (Total dataset: {len(features)} rows)")

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
    measurer.end(tracker=tracker,
                 shape=shape,
                 libraries=[v.__name__ for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
                 data_path=os.environ.get("HOME") +"/s3/data/d012_luxembourg/",
                 program_path=__file__,
                 variables=locals(),
                 csv_file='benchmarks_maxent.csv')


