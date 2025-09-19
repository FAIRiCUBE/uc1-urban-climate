import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is enabled
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from src.models.Weighted_Clustering import WeightedKMeans
"""
Example script for weighted k-means clustering of cities based on their features.
Uses the WeightedKMeans class from src.models.Weighted_Clustering.py and the European cities atlas from data/eu_cities_atlas/eu_cities_atlas_v0.1.geojson.
"""

def weighted_kmeans_clustering(
    data, min_cities=100, threshold=0.9, optimal_k=17, out_dir="", weight_dict={}
):
    """Weighted clustering of cities based on their features. 

    Args:
        data (DataFrame): pandas DataFrame with features as columns and cities as rows
        min_cities (int, optional): minimunm number of non-zero rows. Features that have #nan > min_cities will be discarded. Defaults to 100.
        threshold (float, optional): Correlation threshold. One of the correlated feature in a correlated feature pair is discarded. Defaults to 0.9.
        optimal_k (int, optional): Number of clusters. Defaults to 17.
        out_dir (str, optional): Output directory to save plots. If empty, plots are not saved. Defaults to "".
        weight_dict (dict, optional): Weights assigned to features to increase their influence on the clustering. Defaults to {}. Pass an empty weight_dict to set all weights to 1.

    Returns:
        DataFrame: DataFrame with city index and assigned cluster
    """

    df = data.copy()
    ##### Handle missing values and drop columns with too many NaNs
    for i in df.columns:
        nan_count = df[i].isna().sum()
        if nan_count > min_cities:
            df = df.drop(columns=[i])
        else:
            df = df.dropna(subset=[i])

    ##### Check correlation between features and drop highly correlated ones
    correlation_matrix = df.corr()

    # plt.figure(figsize=(50, 50))  # Set the figure size
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Matrix")  # Set the title of the heatmap
    if out_dir != "":
        plt.savefig("correlation_matrix_all.png", format="png")
    plt.show()

    correlated_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            # Check if the absolute correlation value exceeds the threshold
            correlation_value = correlation_matrix.iloc[i, j]
            if abs(correlation_value) > threshold:
                # Add the pair of features and their correlation value to the list
                correlated_pairs.append(
                    (
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_value,
                    )
                )

    # Print the pairs of features along with their corresponding correlation values
    print(f"{len(correlated_pairs)} pairs of features with correlation > {threshold}")
    for pair in correlated_pairs:
        print(pair[0], "and", pair[1], "have correlation", pair[2])

    for pair in correlated_pairs:
        if pair[0] in df.columns:
            df = df.drop(columns=[pair[0]])

    ##### Data normalization and clustering
    # This only if needed (impute values if some exist)
    imputer = SimpleImputer(strategy="mean")
    imputed_features = imputer.fit_transform(df)
    # MinMaxScaler alternatively, we can use StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_features = scaler.fit_transform(imputed_features)
    # set all weights to 1 if no weight dict is provided
    if weight_dict == {}:
        weight_dict = {i: 1 for i in df.columns}
    feature_weights = []
    for i in df.columns:
        feature_weights.append(weight_dict[i])
    feature_weights /= np.sum(feature_weights)

    weighted_kmeans = WeightedKMeans(
        n_clusters=optimal_k, max_iter=100, random_state=42
    )
    weighted_kmeans.fit(normalized_features, feature_weights)
    df["cluster"] = weighted_kmeans.labels_
    clustered_cities = df[["cluster"]]

    #### 2D and 3D Visualization of clusters
    tsne = TSNE(
        n_components=2, random_state=42
    )  # Set n_components=2 for 2D visualization
    tsne_features = tsne.fit_transform(normalized_features)
    jet_cmap = plt.get_cmap("jet")
    cluster_colors = [jet_cmap(i / float(optimal_k)) for i in range(optimal_k)]

    # Create a 2D scatter plot for t-SNE visualization
    plt.figure(figsize=(10, 8))
    for cluster_label, color in zip(range(optimal_k), cluster_colors):
        # Filter data points belonging to the current cluster
        cluster_indices = np.where(weighted_kmeans.labels_ == cluster_label)[0]
        plt.scatter(
            tsne_features[cluster_indices, 0],
            tsne_features[cluster_indices, 1],
            label=f"Cluster {cluster_label}",
            color=color,
            alpha=0.5,
        )

    plt.title("t-SNE Visualization of Clusters")
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.legend()
    if out_dir != "":
        plt.savefig(f"{out_dir}/Clustering_TSNE_2D.png", format="png")
    plt.show()
    plt.close()

    tsne = TSNE(n_components=3, random_state=42)
    tsne_features = tsne.fit_transform(normalized_features)
    cluster_colors = [jet_cmap(i / float(optimal_k)) for i in range(optimal_k)]

    # Create a 3D scatter plot for t-SNE visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for cluster_label, color in zip(range(optimal_k), cluster_colors):
        # Filter data points belonging to the current cluster
        cluster_indices = np.where(weighted_kmeans.labels_ == cluster_label)[0]
        ax.scatter(
            tsne_features[cluster_indices, 0],
            tsne_features[cluster_indices, 1],
            tsne_features[cluster_indices, 2],
            label=f"Cluster {cluster_label}",
            color=color,
            alpha=0.5,
        )

    ax.set_title("t-SNE Visualization of Clusters")
    ax.set_xlabel("t-SNE Feature 1")
    ax.set_ylabel("t-SNE Feature 2")
    ax.set_zlabel("t-SNE Feature 3")  # type: ignore # This works because ax is a 3D axes object
    ax.legend()
    if out_dir != "":
        plt.savefig(f"{out_dir}/Clustering_TSNE_3D.png", format="png")
    plt.show()
    plt.close()

    # for i in features.columns[-14:]:
    #     df[i] = df[i].astype(int)
    # Box plot of feature values within each cluster
    for i in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="cluster", y=i, data=df)
        plt.title(f"Boxplot of {i} by Cluster")
        if out_dir != "":
            plt.savefig(f"{out_dir}/boxplot_{i}.png", format="png")

    # Get all combinations of feature pairs
    feature_combinations = list(itertools.combinations(df.columns, 2))

    # Define a list of colors for each cluster
    cluster_colors = [jet_cmap(i / float(optimal_k)) for i in range(optimal_k)]

    # Create scatter plots for each pair of features
    for feature_pair in feature_combinations:
        feature1_index = df.columns.get_loc(feature_pair[0])
        feature2_index = df.columns.get_loc(feature_pair[1])

        plt.figure(figsize=(8, 6))
        for cluster_label, color in zip(range(optimal_k), cluster_colors):
            # Filter data points belonging to the current cluster
            cluster_indices = np.where(weighted_kmeans.labels_ == cluster_label)[0]
            plt.scatter(
                normalized_features[cluster_indices, feature1_index],
                normalized_features[cluster_indices, feature2_index],
                label=f"Cluster {cluster_label}",
                color=color,
                alpha=0.5,
            )

        plt.title(
            f"Scatter Plot of Clusters for {feature_pair[0]} vs {feature_pair[1]}"
        )
        plt.xlabel(feature_pair[0])
        plt.ylabel(feature_pair[1])
        plt.legend()
        if out_dir != "":
            plt.savefig(
                f"{out_dir}/{feature_pair[0]}_vs_{feature_pair[1]}_scatter_plot.png",
                format="png",
            )
        plt.show()
        plt.close()
        break  # Remove this line if you want to plot for all pairs

    print(f"Figures saved in {out_dir}")
    return clustered_cities


if __name__ == "__main__":
    ###### define source of data and output directory
    data_path = "./data/eu_cities_atlas/eu_cities_atlas_v0.1.geojson"
    out_dir = ""

    ##### Load data
    df = gpd.read_file(data_path)
    df = df.drop(columns=["geometry"])
    # set index to urau_code
    df = df.set_index("urau_code")
    print(df.columns)

    df2 = df[["urau_name", "ez_code"]]

    df = pd.get_dummies(df, columns=["ez_code"])
    feature_columns = df.columns[4:]

    optimal_k = 17  # Set the optimal number of clusters based on prior analysis
    clustered = weighted_kmeans_clustering(
        df[feature_columns], min_cities=100, threshold=0.9, optimal_k=optimal_k
    )
    df["cluster"] = clustered["cluster"]
    ##### Visualizations
    cluster_counts = clustered.value_counts()
    print(cluster_counts)

    df = df.merge(df2[["urau_name", "ez_code"]], on="urau_name", how="inner")

    ez_counts = df["ez_code"].value_counts()
    print(ez_counts)
