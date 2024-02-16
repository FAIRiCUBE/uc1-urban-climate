import pandas as pd
from configparser import ConfigParser
import sqlalchemy as sa
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, text
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import os
import itertools
import numpy as np
import warnings

from sklearn.manifold import TSNE

class WeightedKMeans:
    def __init__(self, n_clusters=8, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X, weights):
        np.random.seed(self.random_state)
        # Initialize cluster centers randomly
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]

        for _ in range(self.max_iter):
            # Assign labels based on closest cluster center
            self.labels_ = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1)

            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i]
                if len(cluster_points) == 0:
                    new_center_idx = np.random.randint(X.shape[0])
                    self.labels_[new_center_idx] = i
            # Update cluster centers based on the mean of the points in each cluster

            Y = X * weights
            new_centers = np.array([Y[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])




            distances = np.linalg.norm(X - self.cluster_centers_[self.labels_], axis=1)
            self.inertia_ = np.sum(distances ** 2)
            # Check for convergence
            if np.allclose(self.cluster_centers_, new_centers):
                break

            self.cluster_centers_ = np.array(new_centers)

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1)

# Example usage:
# Initialize and fit the Weighted K-means model





from sklearn.preprocessing import StandardScaler

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

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    ################################################################
    keys = config(filename='database_nilu.ini')  #
    ################################################################

    POSTGRESQL_SERVER_NAME = keys['host']
    PORT = keys['port']
    Database_name = keys['database']
    USER = keys['user']
    PSW = keys['password']
    ##################################################

    engine_postgresql = sa.create_engine(
        'postgresql://' + USER + ':' + PSW + '@' + POSTGRESQL_SERVER_NAME + ':' + str(PORT) + '/' + Database_name)
    print(engine_postgresql)
    connection = engine_postgresql.raw_connection()
    cursor = connection.cursor()
    connection.commit()
    print("done")

    ## The following script is reading our lasted city CUBE dataset from PostgreSQL Server and imported the table (VIEW) into a data frame:

    connection = engine_postgresql.raw_connection()
    cursor = connection.cursor()
    connection.commit()

    ## testing reading tables from database:

    with engine_postgresql.begin() as conn:
        query = text("""

                  SELECT * FROM public.city_2018_demo_view;


        """)
        df = pd.read_sql_query(query, conn)

    cursor.close()
    connection.commit()

    print("View (1) import to df - done")

    df['ez_code'] = df['ez_code'].replace('None', np.nan)



    df = df.drop(columns=['info_columns'])
    df2 = df[['urau_name', 'ez_code']]

    df = pd.get_dummies(df, columns=['ez_code'])
    # le = LabelEncoder()
    # df['ez_code'] = le.fit_transform(df['ez_code'])

    min_cities = 100
    for i in df.columns[13:]:
        nan_count = df[i].isna().sum()
        if nan_count > min_cities:
            df = df.drop(columns=[i])
        else:
            df = df.dropna(subset=[i])

    feature_columns = df.columns[13:]
    correlation_matrix = df[feature_columns].corr()
    plt.figure(figsize=(50, 50))  # Set the figure size
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Correlation Matrix')  # Set the title of the heatmap
    plt.savefig('Correlation Matrix_all.pdf', format='pdf')
    plt.close()

    threshold = 0.9  # Adjust as needed
    correlated_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            # Check if the absolute correlation value exceeds the threshold
            correlation_value = correlation_matrix.iloc[i, j]
            if abs(correlation_value) > threshold:
                # Add the pair of features and their correlation value to the list
                correlated_pairs.append(
                    (correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_value))

    # Print the pairs of features along with their corresponding correlation values
    print("Pairs of features with correlation > ", threshold)
    print('Pairs size: ', len(correlated_pairs))
    for pair in correlated_pairs:
        print(pair[0], "and", pair[1], "have correlation", pair[2])

    for pair in correlated_pairs:
        if pair[0] in df.columns:
            df = df.drop(columns=[pair[0]])

    # Selecting features (From 0 to 12 are cities infos, e.g.s city code)
    features = df.iloc[:, 13:]
    # This only if needed (impute values if some exist)
    imputer = SimpleImputer(strategy='mean')
    imputed_features = imputer.fit_transform(features)
    # MinMaxScaler alternatively, we can use StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_features = scaler.fit_transform(imputed_features)

    weight_dictionary = {
        'dem_min': 1,
        'dem_max': 1,
        'dem_std': 1,
        'dem_mean': 1,
        'ez_code': 1,
        'imd_percent_2018': 1,
        'treecover_percent_2018': 1.5,
        'class_11100': 1,
        'class_11210': 1,
        'class_11220': 1,
        'class_11230': 1,
        'class_11240': 1,
        'class_11300': 1,
        'class_12100': 1,
        'class_12210': 1,
        'class_12220': 1,
        'class_12230': 1,
        'class_12300': 1,
        'class_12400': 1,
        'class_13100': 1,
        'class_13300': 1,
        'class_13400': 1,
        'class_14100': 1,
        'class_14200': 1,
        'class_21000': 1,
        'class_22000': 1,
        'class_23000': 1,
        'class_24000': 1,
        'class_25000': 1,
        'class_31000': 1,
        'class_32000': 1,
        'class_33000': 1,
        'class_40000': 1,
        'class_50000': 1,
        'urban_green_percent': 2,
        'urban_blue_percent': 2,
        'avg_2m_temp_kelvin_2018': 2,
        'number_of_summer_days_2018': 2,
        'number_of_tropical_nights_2018': 2,
        'utci_heat_nights_2018': 2,
        'coastal_city': 1.5,
        'de1001v_2018': 2,
        'de1028v_2018': 1.5,
        'de1055v_2018': 1.5,
        'ec1174v_2018': 1,
        'ec1010v_2018': 1,
        'ec1020i_2018': 1.5,
        'ec3040v_2018': 1,
        'sa2013v_2018': 1,
        'de1028i_2018': 1.5,
        'de1055i_2018': 1.5,
        'ez_code_ALN': 1,
        'ez_code_ALS': 1,
        'ez_code_ARC': 1,
        'ez_code_ATC': 1,
        'ez_code_ATN': 1,
        'ez_code_BOR': 1,
        'ez_code_CON': 1,
        'ez_code_LUS': 1,
        'ez_code_MAC': 1,
        'ez_code_MDM': 1,
        'ez_code_MDN': 1,
        'ez_code_MDS': 1,
        'ez_code_NEM': 1,
        'ez_code_PAN': 1
    }
    feature_weights = []
    for i in features.columns:
        feature_weights.append(weight_dictionary[i])
    feature_weights /= np.sum(feature_weights)
    # Elbow Method
    # inertia = []
    # k_range = range(1, 25)
    # for k in k_range:
    #     weighted_kmeans  = WeightedKMeans(n_clusters=k, max_iter=100,random_state=42)
    #     weighted_kmeans.fit(normalized_features, feature_weights)
    #     inertia.append(weighted_kmeans.inertia_)
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_range, inertia, marker='o', linestyle='-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for Optimal k')
    # plt.grid(True)
    # plt.show()

    optimal_k = 17
    weighted_kmeans = WeightedKMeans(n_clusters=optimal_k, max_iter=100, random_state=42)
    weighted_kmeans.fit(normalized_features, feature_weights)
    df['Cluster'] = weighted_kmeans.labels_
    clustered_cities = df[['city_code', 'urau_name', 'Cluster']]



    cluster_counts = df['Cluster'].value_counts()
    print(cluster_counts)

    df = df.merge(df2[['urau_name', 'ez_code']], on='urau_name', how='inner')

    #print(df[['urau_name', 'Cluster', 'ez_code']])
    ez_counts = df['ez_code'].value_counts()
    print(ez_counts)


    result_dir = 'Results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    tsne = TSNE(n_components=3, random_state=42)
    tsne_features = tsne.fit_transform(normalized_features)
    cluster_colors = [plt.cm.jet(i / float(optimal_k)) for i in range(optimal_k)]

    # Create a 3D scatter plot for t-SNE visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_label, color in zip(range(optimal_k), cluster_colors):
        # Filter data points belonging to the current cluster
        cluster_indices = np.where(weighted_kmeans.labels_ == cluster_label)[0]
        ax.scatter(tsne_features[cluster_indices, 0], tsne_features[cluster_indices, 1],
                   tsne_features[cluster_indices, 2],
                   label=f'Cluster {cluster_label}', color=color, alpha=0.5)

    ax.set_title('t-SNE Visualization of Clusters')
    ax.set_xlabel('t-SNE Feature 1')
    ax.set_ylabel('t-SNE Feature 2')
    ax.set_zlabel('t-SNE Feature 3')
    ax.legend()

    file_name = 'Clustering_TSNE.pdf'
    file_path = os.path.join(result_dir, file_name)
    plt.savefig(file_path, format='pdf')
    plt.close()

    print(f"Figure saved as '{file_path}'.")

    for i in features.columns[-14:]:
        df[i] = df[i].astype(int)
    # Box plot of feature values within each cluster
    for i in features.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Cluster', y=i, data=df)
        plt.title(f'Boxplot of {i} by Cluster')
        file_name = f'Boxplot of {i} by Cluster.pdf'
        file_path = os.path.join(result_dir, file_name)
        plt.savefig(file_path, format='pdf')


    # Get all combinations of feature pairs
    feature_combinations = list(itertools.combinations(features.columns, 2))

    # Define a list of colors for each cluster
    cluster_colors = [plt.cm.jet(i / float(optimal_k)) for i in range(optimal_k)]

    # Create scatter plots for each pair of features
    for feature_pair in feature_combinations:
        feature1_index = features.columns.get_loc(feature_pair[0])
        feature2_index = features.columns.get_loc(feature_pair[1])

        plt.figure(figsize=(8, 6))
        for cluster_label, color in zip(range(optimal_k), cluster_colors):
            # Filter data points belonging to the current cluster
            cluster_indices = np.where(weighted_kmeans.labels_ == cluster_label)[0]
            plt.scatter(normalized_features[cluster_indices, feature1_index],
                        normalized_features[cluster_indices, feature2_index],
                        label=f'Cluster {cluster_label}', color=color, alpha=0.5)

        plt.title(f'Scatter Plot of Clusters for {feature_pair[0]} vs {feature_pair[1]}')
        plt.xlabel(feature_pair[0])
        plt.ylabel(feature_pair[1])
        plt.legend()
        file_name = f"{feature_pair[0]}_vs_{feature_pair[1]}_scatter_plot.pdf"
        file_path = os.path.join(result_dir, file_name)
        plt.savefig(file_path, format='pdf')
        break # Remove this line if you want to plot for all pairs
        plt.close()

    print(f"Figures saved in '{result_dir}' directory.")

