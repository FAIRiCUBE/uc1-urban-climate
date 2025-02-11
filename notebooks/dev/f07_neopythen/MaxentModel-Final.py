import numpy as np
from glob import glob
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path, WindowsPath
import pandas as pd
import xarray as xr
from configparser import ConfigParser
import sqlalchemy as sa # conection to the database
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.ndimage import gaussian_filter
import elapid
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import rioxarray as rxr
import shap  
import matplotlib.patches as mpatches



print ("Libraries loaded")

# connect to DATABASE server: 

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
keys = config(filename='database_nilu.ini')
POSTGRESQL_SERVER_NAME=keys['host']
PORT=                  keys['port']
Database_name =        keys['database']
USER =                 keys['user']
PSW =                  keys['password']
##################################################

engine_postgresql = sa.create_engine('postgresql://'+USER+':'+PSW+ '@'+POSTGRESQL_SERVER_NAME+':'+str(PORT)+ '/' + Database_name)
print (engine_postgresql)
connection = engine_postgresql.raw_connection()
cursor = connection.cursor()
connection.commit()
print ("Connected to SQL")


species_list = [
'Robinia Pseudoacacia', 
'Fallopia Japonica', 
'Impatiens Glandulifera', 
'Heracleum Mantegazzianum'       
]


# # reading raster from CWS:

##  base folder on CWS:
base_path = os.environ.get("HOME") +"/s3/data/d012_luxembourg/"


## Datasets 01 Lichtzahl:-------------------------------------------
d01_L_parameter = os.path.join(base_path, 'shadow_2019_10m_b1.tif')
print(d01_L_parameter)
# Open the file:
cube_01_L = rxr.open_rasterio(d01_L_parameter)
cube_01_L = cube_01_L.to_dataset(name='d01_L_light')

### Dataset 02 Feuchtezahl :-------------------------------------------
d02_F_parameter = os.path.join(base_path, 'twi_2019_10m_b1.tif')
print(d02_F_parameter)
# Open the file:
cube_02_F = rxr.open_rasterio(d02_F_parameter)
cube_02_F = cube_02_F.to_dataset(name='d02_F_wetness')

### Dataset 03 Temperatur:-----------------------------------------------------------------------------------------------------------S
### monthly temp for 2017
d03_T_parameter_2017 = os.path.join(base_path, 'air_temperature_2017_month_mean_10m_b12.tif')
cube_03_temperature_2017 = rxr.open_rasterio(d03_T_parameter_2017)
cube_03_temperature_2017 = cube_03_temperature_2017.to_dataset(name='d03_T_parameter_2017')
cube_03_temperature_2017 = cube_03_temperature_2017.mean(dim='band')


# ## Dataset 04 Kontinentaliätzahl:-------------------------------------------
# d04_K_parameter = os.path.join(base_path, 'air_temperature_2017_month_mean_10m_b12.tif')
# print(d04_K_parameter)
# # Open the file:
# cube_04_K = rxr.open_rasterio(d04_K_parameter)
# cube_04_K = cube_04_K.to_dataset(name='d04_K_continentality')
# cube_04_K= cube_04_K.mean(dim='band')  # mean


### Dataset 05 Reaktionszahl (ph):-------------------------------------------
d05_R_parameter = os.path.join(base_path, 'pH_CaCl_10m_b1.tif')
print(d05_R_parameter)
# Open the file:
cube_05_R = rxr.open_rasterio(d05_R_parameter)
cube_05_R = cube_05_R.to_dataset(name='d05_R_ph')

## ### Dataset 06 Stickstoff:-------------------------------------------
d06_N_parameter = os.path.join(base_path, 'soil_nitrat_10m_b1.tif')
print(d06_N_parameter)
## # Open the file:
cube_06_N = rxr.open_rasterio(d06_N_parameter)
cube_06_N = cube_06_N.to_dataset(name='d06_N_nitrogen')## 

# # ### Dataset 07 Salz:------------------------------------------- NO DATA ()
# # d07_S_parameter = os.path.join(base_path, 'xxx.tif')
# # print(d07_S_parameter)
# # # Open the file:
# # cube_07_S = rxr.open_rasterio(d07_S_parameter)
# # cube_07_S = cube_07_S.to_dataset(name='d07_S_salt')## 

# # ### Dataset 08 Schwermetall:-------------------------------------------  NO DATA ()
# # d08_HM_parameter = os.path.join(base_path, 'xxx.tif')
# # print(d08_HM_parameter)
# # # Open the file:
# # cube_08_HM = rxr.open_rasterio(d08_HM_parameter)
# # cube_08_HM = cube_08_HM.to_dataset(name='d08_HM_heavy_metal')

### Dataset 09 Lebensform:-------------------------------------------
d09_watersurface_raster = os.path.join(base_path, 'land_cover_2021_10m_b1.tif')
cube_09__temp_LF = rxr.open_rasterio(d09_watersurface_raster)
#print(cube_09__temp_LF)
cube_09__temp_LF = cube_09__temp_LF.to_dataset(name='d09_LV_landcover')

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


ds = cube_09__temp_LF
d09_LF_parameter_temp_water_area =    xr.where(ds['d09_LV_landcover'] == 60, 1, 0) # Else set to 0
# Adding the result back to the dataset (optional)
cube_09__temp_LF['ellenberg_water_area'] = d09_LF_parameter_temp_water_area

cube_09_LF_x = cube_09__temp_LF['ellenberg_water_area'] 
cube_09_1_LF_water = cube_09_LF_x.to_dataset(name='ellenberg_water_area')
#cube_09_LF_water

d09_LF_parameter_temp_not_sealed =    xr.where(ds['d09_LV_landcover'].isin ([30,70,71,80,91,92,93]), 1, 0) # Else set to 0
# Adding the result back to the dataset (optional)
cube_09__temp_LF['ellenberg_not_sealed_area'] = d09_LF_parameter_temp_not_sealed
cube_09_LF_x_non_sealed = cube_09__temp_LF['ellenberg_not_sealed_area'] 
cube_09_2_LF_non_sealed = cube_09_LF_x_non_sealed.to_dataset(name='ellenberg_not_sealed_area')
#cube_09_LF_non_sealed



print ("----------------Data Uploaded----------------")


# the following code set up ONE datacube with the raw data - (only to be used for model -tuning in the moment)

c_1 = cube_01_L 
c_2 = cube_02_F 
c_3 = cube_03_temperature_2017 
#c_4 = cube_04_K 
c_5 = cube_05_R
c_6 = cube_06_N 
c_7 = cube_09__temp_LF 
c_8 = cube_09_1_LF_water 
c_9 = cube_09_2_LF_non_sealed 

habitat_parameter_cube = xr.merge([c_1, c_2, c_3, c_5, c_6,c_7, c_8, c_9])
#habitat_parameter_cube=habitat_parameter_cube.squeeze()



# the following code set up ONE datacube with the raw data - (only to be used for model -tuning in the moment)

## for masking of the final data we need an extra cube:
c_8 = cube_09_1_LF_water 
c_9 = cube_09_2_LF_non_sealed 

sealed_water_cube = xr.merge([ c_8, c_9])
#habitat_parameter_cube=habitat_parameter_cube.squeeze()
#habitat_parameter_cube
# Showing cube data

# Create the list of variables and names

parameters = [
    #habitat_parameter_cube_wit_occurence.occurence_data,
    habitat_parameter_cube.d01_L_light.sel(band=1),
    habitat_parameter_cube.d02_F_wetness.sel(band=1),
    habitat_parameter_cube.d03_T_parameter_2017,
    #habitat_parameter_cube.d04_K_continentality,
    habitat_parameter_cube.d05_R_ph.sel(band=1),
    habitat_parameter_cube.d06_N_nitrogen.sel(band=1),
    habitat_parameter_cube.d09_LV_landcover.sel(band=1),
    habitat_parameter_cube.ellenberg_water_area.sel(band=1),
    habitat_parameter_cube.ellenberg_not_sealed_area.sel(band=1)
]

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

# Plot each variable
plt.figure(figsize=(15, 20))

for i, param in enumerate(parameters):
    plt.subplot(4, 3, i + 1)  # Create a grid of subplots
    param.plot()  # No need to specify 'cmap' here, it will default
    plt.title(parameter_names[i])

plt.tight_layout

for species in species_list:
    print(f"---------{species} Selected------------")
    species_name = species.replace(" ", "_")
    #print (species_name)
    print (species_name)

    #Fallopia_Japonica
    query = """
    SELECT *
    FROM luxembourg_species.neophytes_geometry
    """
    species_occ_df = pd.read_sql(query, engine_postgresql)

    species_occ_df = species_occ_df[species_occ_df['species_name']==species]

    x_coords = species_occ_df["gridnum2169_10m_x"].values
    y_coords = species_occ_df["gridnum2169_10m_y"].values

    species_occ_df[species_occ_df['species_name']==species]




    # pseudo absence data:
    # SQL query to select points where species does NOT occur but share the same grid coordinates
    query_non_occ = f"""
    SELECT *
    FROM luxembourg_species.neophytes_geometry
    WHERE species_name != '{species}'
    AND (gridnum2169_10m_x, gridnum2169_10m_y) NOT IN (
        SELECT gridnum2169_10m_x, gridnum2169_10m_y
        FROM luxembourg_species.neophytes_geometry
        WHERE species_name = '{species}'
    );
    """
    # Fetch the non-occurrence data into a Pandas DataFrame
    non_occ_df_all = pd.read_sql(query_non_occ, engine_postgresql)

    x_coords_da = xr.DataArray(x_coords)
    y_coords_da = xr.DataArray(y_coords)

    # merge
    xds_merged = habitat_parameter_cube

    nearest_habitat_values = xds_merged.sel(
        x=x_coords_da,
        y=y_coords_da,
        method="nearest"
    )

    # Convert to DataFrame and merge with occurrence data
    nearest_habitat_df = nearest_habitat_values.to_dataframe().reset_index()
    nearest_habitat_df[species] = True

    print(f"Size of {species} prsence points = {len(nearest_habitat_df)}")


    #Select data from the SQL table (with species different than the selected one + different from any location in the species data)
    #Specify number of background points 
    nb_background = 10*len(nearest_habitat_df)  ### why 3
    non_occ_df = non_occ_df_all#.sample(n=nb_background) 
    print(f"Background data size= {len(non_occ_df)}")
    
    x_non_occ_coords = non_occ_df['gridnum2169_10m_x'].values
    y_non_occ_coords = non_occ_df['gridnum2169_10m_y'].values   


    x_selected = x_non_occ_coords
    y_selected = y_non_occ_coords   

    x_selected_da = xr.DataArray(x_selected)
    y_selected_da = xr.DataArray(y_selected)    

    # Step 3: Extract habitat values for the selected non-occurrence coordinates
    non_occ_habitat_values = xds_merged.sel(
        x=x_selected_da,
        y=y_selected_da,
        method="nearest"
    )   



    # Step 4: Convert the non-occurrence habitat data to a DataFrame
    non_occ_habitat_df = non_occ_habitat_values.to_dataframe().reset_index()    

    # Step 5: Mark these samples as "False" for species presence
    non_occ_habitat_df[species] = False 

    #non_occ_habitat_df.to_csv('background_' + species + '.csv', index=False)

    ## (3) MAXENT (Elapid)Machine Learning for Modeling species distribution 
    ## MAXENT: data preparation
    # Rename columns for consistency
    background_data = non_occ_habitat_df.rename(columns={'x': 'longitude', 'y': 'latitude'})
    presence_data = nearest_habitat_df.rename(columns={'x': 'longitude', 'y': 'latitude'})
    background_data = background_data.replace(-9999, np.nan)
    presence_data = presence_data.replace(-9999, np.nan)
    # Drop rows with NaN values
    background_data = background_data.dropna()
    presence_data = presence_data.dropna()

    # Combine presence and background data
    presence_labels = np.ones(len(presence_data))  # 1 for presence
    background_labels = np.zeros(len(background_data))  # 0 for background

    # Combine the data into one dataset
    combined_data = pd.concat([presence_data, background_data], ignore_index=True)
    labels = np.concatenate([presence_labels, background_labels])

    print(f"All data size: {len(combined_data)}")
    # Select environmental variables (excluding species and coordinates)
    features = combined_data.drop(columns=[species, 'longitude', 'latitude', 'band', 'dim_0', 'spatial_ref'])


    print(f"Number of features = {len(features.columns)}")

    print('################### Maxent 1: 80-20 split on all the data #########################')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)    # 80 % Training, 20% QC data


    # Step 3: Train the Maxent model
    maxent = elapid.MaxentModel()
    maxent.fit(X_train, y_train)

    # Step 4: Make predictions
    y_pred_MX1 = maxent.predict(X_test)

    # Step 5: Evaluate the model
    auc_score = roc_auc_score(y_test, y_pred_MX1)
    print(f"Maxent AUC Score: {auc_score}")
    X_test['prediction'] = y_pred_MX1

    ##MAXENT: #### (3.1.1)- Train on all data
    maxent = elapid.MaxentModel()
    maxent.fit(features, labels)

    # Step 4: Make predictions
    pred_prob = maxent.predict(features)
    # Step 5: Evaluate the model
    auc_score = roc_auc_score(labels, pred_prob)
    print(f"Maxent 1 AUC Score: {auc_score}")

    print('################### Maxent2: that trains on: Part of the positives + all background data and tests on: the second part of positives + all background data  #########################')
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

    # Evaluate using AUC
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"Maxent 2 AUC Score: {auc_score}")

    
    ##################################Train on all#####################################
    # Train the Maxent model
    maxent = elapid.MaxentModel()
    maxent.fit(features, labels)

    # Predict suitability scores for the test set
    y_pred_prob = maxent.predict(features)

    # Evaluate using AUC--NOT REPRESENTATIVE
    auc_score = roc_auc_score(labels, y_pred_prob )
    print(f"Maxent ALL AUC Score: {auc_score}")

    ######################################################################## SHAP for explaning ################################################################################
    shap_sample = True  # Run only on a selected sample of size sample_size, otherwise run on all
    sample_size = 400

    shap_detailed = True # Run explanation for all each data point, otherwise a global explanation with correlation

    # Define predict function for Maxent suitability maps 
    def predict_suitability(X):
        # Ensure X is in the correct format
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=features.columns)
        #print("Input to predict_suitability (shape):", X.shape) 
        # Predict suitability values (continuous outputs)
        return maxent.predict(X).flatten()


    print("Features/points Shape:", features.shape)

    if shap_sample:
        sample_features = shap.sample(features, sample_size, random_state=42)
        print("Sample Features/points Shape:", sample_features.shape)
    else:
        sample_features = features

    # Create explained
    explainer = shap.KernelExplainer(predict_suitability, sample_features)

    # Generate SHAP values
    shap_values = explainer.shap_values(sample_features)


    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample_features, show=False)
    plt.title("SHAP Summary Plot for Maxent ("+species +")")
    plt.tight_layout()

    # Save the SHAP summary plot
    if shap_sample:
        plt.savefig(f"Shap_summary_plot_{species} _Detailed_Sample_{sample_size}.png", dpi=300)
        plt.close()
        print(f"SHAP summary plot saved as 'Shap_summary_plot_{species} _Detailed_Sample_{sample_size}.png")
    else:
        plt.savefig("Shap_summary_plot_" +species + "_Detailed_All.png", dpi=300)
        plt.close()
        print(f"SHAP summary plot saved as 'Shap_feature_importance_{species}_Detailed_all.png'")


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

    #**Sort data by SHAP values in descending order (most important feature at the top)**
    feature_importance_df = feature_importance_df.sort_values(by='Mean_Abs_SHAP', ascending=False)

    # Extract sorted values
    sorted_features = feature_importance_df['Feature'].values
    sorted_shap_values = feature_importance_df['Mean_Abs_SHAP'].values
    sorted_correlation_values = feature_importance_df['Correlation'].values

    # 🔹 **Assign colors AFTER sorting**
    colors = ['blue' if corr > 0 else 'red' for corr in sorted_correlation_values]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_features, sorted_shap_values, color=colors)  # Bars now sorted correctly
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title(f'Global Feature Importance for Maxent Suitability ({species})')

    # 🔹 **Ensure correlation values appear correctly inside each corresponding bar**
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

    # Add legend for color interpretation
    legend_patches = [
        mpatches.Patch(color='blue', label='Higher values increase suitability'),
        mpatches.Patch(color='red', label='Lower values increase suitability')
    ]
    plt.legend(handles=legend_patches, loc='lower right')

    plt.gca().invert_yaxis()  # Ensure most important feature remains on top
    plt.tight_layout()

    # Save the plot
    if shap_sample:
        filename = f"Global_shap_feature_importance_{species}_Sample_{sample_size}.png"
    else:
        filename = f"Global_shap_feature_importance_{species}_All.png"

    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Global SHAP feature importance plot saved as '{filename}'")



    ############## Save in database#################    

    ##MAXENT: now we use the model (maxent) to calculate all pixel of the full city dataset.:
    # Convert to DataFrame
#     df_full_cube = habitat_parameter_cube.to_dataframe().reset_index()
#     # remove not needed columns for model use:
#     df_features_all = df_full_cube.drop(columns=['x', 'y', 'band', 'spatial_ref'])
#     # Step 4: Make predictions
#     y_pred_full = maxent.predict(df_features_all)
#     df_full_cube['prediction'] = y_pred_full
#     #using pandas to mask out:
#     # Update 'prediction' column based on the condition
#     df_full_cube.loc[(df_full_cube['ellenberg_not_sealed_area'] == 0) | (df_full_cube['ellenberg_water_area'] == 1), 'prediction'] = 0




#     df_full_cube2 = df_full_cube[['x', 'y', 'prediction']].dropna()

#     # Create a GeoDataFrame from results with points:
#     geometry = [Point(xy) for xy in zip(df_full_cube2['x'], df_full_cube2['y'])]
#     gdf = gpd.GeoDataFrame(df_full_cube2, geometry=geometry)
#     # Set a coordinate reference system (CRS) (LUREF - 2169)
#     gdf.set_crs(epsg=2169, inplace=True)

#     ### data storing on database

#     # Write DataFrame to PostgreSQL table
#     table_name =  species_name + '_maxent_distribution_v2'
#     schema_name = 'luxembourg_species'

#     # gdf.to_postgis(table_name, engine_postgresql, if_exists='replace', index=False)
#     gdf.to_postgis(table_name, engine_postgresql, schema=schema_name, if_exists='replace', index=False)

#     print(f"Table with geometry written to table '{schema_name}.{table_name}' in PostGIS.")


    #  break


print ("loop done")

