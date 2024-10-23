import numpy as np
import pandas as pd
import xgboost as xgb
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

# Load the datasets
background_data = pd.read_csv('background_Heracleum Mantegazzianum.csv')
presence_data = pd.read_csv('presence_Heracleum Mantegazzianum.csv')

# Rename columns for consistency
background_data = background_data.rename(columns={'x': 'longitude', 'y': 'latitude'})
presence_data = presence_data.rename(columns={'x': 'longitude', 'y': 'latitude'})

# Replace -999999 with NaN
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

# Select environmental variables (excluding species and coordinates)
features = combined_data.drop(columns=['Heracleum Mantegazzianum', 'longitude', 'latitude'])

# Drop rows with NaN values in features and update labels accordingly
features_clean = features.dropna()

# Check if features_clean still contains NaNs
if features_clean.isna().sum().sum() > 0:
    print("There are still NaNs in features_clean")

clean_indices = features_clean.index
labels_clean = labels[clean_indices]

# Step 1: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features_clean, labels_clean, test_size=0.2, random_state=42)

# Step 2: Initialize and train the XGBoost regression model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# Step 3: Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Step 4: Evaluate the model using Mean Squared Error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared (R2): {r2}")




##########################PLOTTING########################################

# Now we define test_data_with_coords correctly from combined_data using the test indices
test_data_with_coords = combined_data.iloc[X_test.index][['longitude', 'latitude']]

# Coordinate transformation
# Assuming your data is in UTM Zone 33N (EPSG:32633), you should replace 'EPSG:32633' if you're using a different CRS
transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

# Apply the transformation to the longitude and latitude columns
test_data_with_coords['longitude'], test_data_with_coords['latitude'] = transformer.transform(
    test_data_with_coords['longitude'].values,
    test_data_with_coords['latitude'].values
)

# Debugging: Check the transformed coordinates
print("\nTransformed Longitude and Latitude:")
print(test_data_with_coords[['longitude', 'latitude']].head(10))

# Proceed with the rest of the process
valid_coords = test_data_with_coords.dropna(subset=['longitude', 'latitude'])

# Filter out extreme values (longitudes beyond [-180, 180] and latitudes beyond [-90, 90])
valid_coords = valid_coords[(valid_coords['longitude'].between(-180, 180)) &
                            (valid_coords['latitude'].between(-90, 90))]

# Create GeoDataFrame from valid test data with coordinates
gdf_test = gpd.GeoDataFrame(valid_coords,
                            geometry=gpd.points_from_xy(valid_coords['longitude'], valid_coords['latitude']))

# Ensure the GeoDataFrame has the correct CRS
gdf_test = gdf_test.set_crs('EPSG:4326')  # Coordinates are now in EPSG:4326 after transformation

# Convert GeoDataFrame to EPSG:3857 for basemap display
gdf_test = gdf_test.to_crs(epsg=3857)

# Generate Heatmap Data
# Extract X and Y values for the heatmap
x_values = gdf_test.geometry.x
y_values = gdf_test.geometry.y

# Create a 2D histogram (heatmap) from the coordinates
heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=300)

# Apply Gaussian filter to smooth the heatmap
heatmap = gaussian_filter(heatmap, sigma=5)

# Create a figure
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the heatmap
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', alpha=0.6)

# Add the basemap
ctx.add_basemap(ax, crs=gdf_test.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Add labels and title
ax.set_title('Predicted Heatmap Distribution of Heracleum Mantegazzianum')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()
