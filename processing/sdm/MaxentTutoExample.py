import numpy as np
import pandas as pd
import elapid
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.metrics import roc_auc_score

"""
Testing elapid Maxent model with test data from the Maxent software.
"""
if __name__ == "__main__":
    # Load the datasets
    background_data = pd.read_csv(".data/test/SDM_test_data/background.csv")
    presence_data = pd.read_csv(".data/test/SDM_test_data/bradypus_swd.csv")

    # Rename columns for consistency
    background_data = background_data.rename(
        columns={"x": "longitude", "y": "latitude"}
    )
    presence_data = presence_data.rename(
        columns={"dd long": "longitude", "dd lat": "latitude"}
    )

    # Combine presence and background data
    presence_labels = np.ones(len(presence_data))  # 1 for presence
    background_labels = np.zeros(len(background_data))  # 0 for background

    # Combine the data into one dataset
    combined_data = pd.concat([presence_data, background_data], ignore_index=True)
    labels = np.concatenate([presence_labels, background_labels])

    # Select environmental variables (excluding species and coordinates)
    features = combined_data.drop(columns=["species", "longitude", "latitude"])

    # Step 3: Initialize and train MaxEnt model
    model = elapid.MaxentModel()
    model.fit(features, labels)

    # Step 4: Make predictions for the entire dataset
    predicted_probabilities = model.predict(features)

    # Evaluate the model using AUC
    auc_score = roc_auc_score(labels, predicted_probabilities)
    print(f"Model AUC: {auc_score}")

    # Plot Predicted probabilities for the test data
    test_data_with_coords = combined_data.iloc[features.index][
        ["longitude", "latitude"]
    ]
    gdf_test = gpd.GeoDataFrame(
        test_data_with_coords,
        geometry=gpd.points_from_xy(
            test_data_with_coords["longitude"], test_data_with_coords["latitude"]
        ),
    )

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    gdf_test.plot(
        ax=axes,
        column=predicted_probabilities,
        cmap="RdYlBu_r",
        legend=True,
        markersize=10,
    )
    axes.set_title("Maxent Predictions for Bradypus variegatus with Background Map")
    axes.set_xlabel("Longitude")
    axes.set_ylabel("Latitude")
    ctx.add_basemap(axes, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
    plt.savefig(".data/test/SDM_test_data/maxent_all_data_result.pdf", format="pdf")
    plt.tight_layout()
    plt.show()
