import numpy as np
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import rasterio as rio
from typing import Any, Optional, Tuple
import matplotlib.patches as mpatches

# Sentinel Hub
from sentinelhub import (
    CRS,
    BBox,
    BBoxSplitter,
    ByocCollection,
    ByocCollectionAdditionalData,
    ByocCollectionBand,
    ByocTile,
    DataCollection,
    DownloadFailedException,
    SentinelHubDownloadClient,
    MimeType,
    SentinelHubBYOC,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
    os_utils,
    Geometry
)

def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """
    This function plots an RGB image using matplotlib's imshow function. 
    The image intensity can be adjusted using the factor parameter, and the intensity range can be clipped using the clip_range parameter.

    Parameters:
    image (np.ndarray): Input image to plot.
    factor (float, optional): Factor to adjust the image intensity. Defaults to 1.0.
    clip_range (Tuple[float, float], optional): Tuple to define the range for intensity clipping. Defaults to None.
    **kwargs: Additional keyword arguments for imshow function.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def list_byoc_collections(list_tiles=False):
    """
    This function lists all BYOC (Bring Your Own CO2) collections and their tiles from the Sentinel Hub. 
    It uses the credentials from the environment variables to connect to the Sentinel Hub. 
    The function can also list the status, creation time, and path of each tile in each collection.

    Parameters:
    list_tiles (bool, optional): If True, the function also lists the tiles for each collection. Defaults to False.

    Returns:
    None
    """
    # Get the configuration parameters from the environment variables
    config = SHConfig()
    config.instance_id = os.environ.get("SH_INSTANCE_ID")
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    config.aws_access_key_id = os.environ.get("username")
    config.aws_secret_access_key = os.environ.get("password")

    # Initialize SentinelHubBYOC class with the configuration
    byoc = SentinelHubBYOC(config=config)
    
    # Get the iterator over the collections
    collections_iterator = byoc.iter_collections()
    my_collections = list(collections_iterator)

    # Iterate over the collections and print their names and ids
    for collection in my_collections:
        print("Collection name:", collection["name"])
        print("Collection id: ", collection["id"])
        
        # If list_tiles is True, list the tiles for each collection
        if(list_tiles):
            tiles = list(byoc.iter_tiles(collection))
            for tile in tiles:
                print("Tile status: ", tile['status'])
                print("Tile created: ", tile['created'])
                print("Tile path: ", tile['path'])
                if(tile['status'] == "FAILED"):
                    print("Ingestion failed error: ", tile['additionalData'])
        print("-------------------")

        
def bbox_optimal_subsize(bbox_size):
    if(bbox_size[0] < 2500 and bbox_size[1] < 2500):
        return 1
    else:
        max_size = np.max(bbox_size)
        size = np.max(bbox_size)//2500 + 1
        return size.item()
### reading raster
raster_2band = "./../../../s3/interim/LU001C/LU001C.tiff"
def combine_ua_table_from_city_subcube(raster_2band, output_file, bbox = None, bbox_size = None):
    # if raster_2band is a path (string), load the tiff with rasterio; otherwise raster_2band is a numpy array, and the bbox argument must be passed
    if(type(raster_2band) is str):
        xs, ys, b1, b2 = load_raster(raster_2band)
    else:
        # crs = "EPSG:4326"
        if((bbox == None) or (bbox_size == None)):
            print("Must provide bounding box for numpy arrays")
            exit()
        xs, ys, b1, b2 = load_numpyvector(raster_2band, bbox, bbox_size)
        
    data = {"X": pd.Series(xs.ravel()),
            "Y": pd.Series(ys.ravel()),
            "city_code": pd.Series(b1.ravel()),
            "urban_atlas_2018": pd.Series(b2.ravel())
           }

    df = pd.DataFrame(data=data)

    ### combine:
    combine_table_cube_urban=df.groupby(['city_code','urban_atlas_2018']).size().reset_index().rename(columns={0:'count'})
    combine_table_cube_urban.to_csv(index=False)

    # ouptut_table = "./../../../s3/data/c001_city_cube/tables/urban_cube_v1.csv"
    filepath = Path(output_file)  
    # filepath.parent.mkdir(parents=True, exist_ok=True)  
    combine_table_cube_urban.to_csv(filepath, index=False)  

def load_raster(raster_2band):
    with rio.Env():
        with rio.open(raster_2band) as src:
            crs = src.crs

            # create 1D coordinate arrays (coordinates of the pixel center)
            xmin, ymax = np.around(src.xy(0.00, 0.00), 9)  # src.xy(0, 0)
            xmax, ymin = np.around(src.xy(src.height-1, src.width-1), 9)  # src.xy(src.width-1, src.height-1)
            x = np.linspace(xmin, xmax, src.width)
            y = np.linspace(ymax, ymin, src.height)  # max -> min so coords are top -> bottom


            # create 2D arrays
            xs, ys = np.meshgrid(x, y)
            b1 = src.read(1)
            b2 = src.read(2)

            # Apply NoData mask
            mask = src.read_masks(1) > 0
            xs, ys, b1,b2 = xs[mask], ys[mask], b1[mask],b2[mask]
            return xs, ys, b1, b2

def load_numpyvector(raster_2band, bbox, bbox_size):
#     TODO change projection
    xmin, ymin, xmax, ymax = bbox
    width = bbox_size[0]
    height = bbox_size[1]
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymax, ymin, height)  # max -> min so coords are top -> bottom

    # create 2D arrays
    xs, ys = np.meshgrid(x, y)
    b1 = raster_2band[:,:,0]
    b2 = raster_2band[:,:,1]

    xs, ys, b1,b2 = xs, ys, b1,b2
    return xs, ys, b1, b2

def get_tiff_paths_from_code(URAU_CODE):
    lookuptable_folder = "./../../../s3/data/d000_lookuptables/"
    l_paths = pd.read_csv(lookuptable_folder+"city_UA_subcubes_path.csv")
    return l_paths[l_paths['URAU_CODE'] == URAU_CODE].path

def buffer_geometry(geometry, crs, buffer_size=1, resolution = 100):
    """
    This function creates a buffer around a given geometry, removes any inner holes, and returns the new geometry,
    its bounding box and its dimensions.

    Parameters:
    geometry (Geometry): The input geometry around which the buffer is to be created.
    crs: Coordinate Reference System  TODO
    buffer_size (int, optional): The size of the buffer to be created. Default is 1.
    resolution (int, optional): The resolution of the output bounding box. Default is 100

    Returns:
    geometry_b (Geometry): The new geometry after creating the buffer and removing inner holes.
    bbox_b (BBox): The bounding box of the new geometry.
    bbox_size_b (tuple): The dimensions of the bounding box of the new geometry.
    """

    # Create a buffer around the input geometry
    if(type(geometry) == gpd.GeoSeries):
        geometry = geometry.item()
    buffer = geometry.buffer(buffer_size)
    # Remove inner holes
    if(buffer.geom_type == 'Polygon'):
        new_poly = Polygon(buffer.exterior.coords, holes=[])
    else: # MultiPolygon
        list_geoms = []
        for poly in buffer.geoms:
            new_poly = Polygon(poly.exterior.coords, holes=[])
            list_geoms.append(new_poly)
        new_poly = MultiPolygon(list_geoms)

    # Create a Geometry object from the new polygon
    geometry_b = Geometry(geometry=new_poly, crs=crs)

    # Get the bounding box of the new geometry
    bbox_b = geometry_b.bbox
    bbox_b_ = BBox(bbox=bbox_b, crs=crs)
    
    # Get the dimensions of the bounding box of the new geometry
    bbox_size_b = bbox_to_dimensions(bbox_b_, resolution=resolution)
    
    return geometry_b, bbox_b, bbox_size_b

def split_geometry(geometry, bbox):
    if(type(bbox) is BBox):
        bbox_polygon = bbox.geometry
    elif(type(bbox) is Polygon):
        bbox_polygon = bbox
    elif(type(bbox) is tuple):
        bbox_polygon = Polygon(bbox)
    
    if(type(geometry) is Geometry):
        geometry = geometry.geometry
    geometry_split = geometry.intersection(bbox_polygon)
    return geometry_split

def CDS_get_utci_hourly_zipped(output_folder, year):
    import cdsapi
    c = cdsapi.Client()
    args = {
        "months": ['01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',],
        "days":   ['01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31'],
        }
    print(year)
    c.retrieve(
            'derived-utci-historical', 
        {
            'version': '1_1',
            'format': 'zip',
            'day': args["days"],
            'month': args["months"],
            'year': year,
            'product_type': 'consolidated_dataset',
            'variable': 'universal_thermal_climate_index',
        },
            output_folder+f'utci_hourly_{year}.zip')
    return f'utci_hourly_{year}'

def unzip_to_folder(input_folder, file_name):
    from zipfile import ZipFile
    climate_path = input_folder + file_name + ".zip"
    # opening the zip file in READ mode
    with ZipFile(climate_path, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()
    
        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall(input_folder+file_name)
        print('Done!')
    return input_folder+file_name+"/"

def bytes_to(bytes_value, to, bsize=1024):
    # convert bytes to megabytes, etc.
    a = {'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5, 'e': 6}
    r = float(bytes_value)
    for i in range(a[to]):
        r = r / bsize
    return r

if __name__ == '__main__':
    # test
    # read in the city polygons
    from src import db_connect
    from sqlalchemy import text
    import geopandas as gpd

    home_dir = os.environ.get('HOME')
    home_dir = 'C:/Users/MariaRicci/Projects_Cdrive/FAIRiCube'
    engine_postgresql = db_connect.create_engine(db_config = f"{home_dir}/uc1-urban-climate/database.ini")

    with engine_postgresql.begin() as conn:
        query = text("""
                SELECT urau_code, urau_name, geometry
                FROM lut.l_city_urau2021
                """)
        gdf = gpd.read_postgis(query, conn, geom_col='geometry')

    row = gdf[gdf.urau_name == 'Verona']
    geometry_gdf = row.geometry # input argument
    bbox_coords = geometry_gdf.bounds.minx, geometry_gdf.bounds.miny, geometry_gdf.bounds.maxx, geometry_gdf.bounds.maxy # input argument (or compute from geometry)

    geometry = Geometry(geometry=geometry_gdf.item(), crs=CRS.WGS84) # define here your geometry
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84) # define here your bounding box
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    # get only buffer zone
    geometry_b, bbox_b, bbox_size_b = buffer_geometry(geometry_gdf, buffer_size=100)

    def plot_variables(parameters, parameter_names, out_dir=""):
    # Plot all variables
    plt.figure(figsize=(15, 20))
    for i, param in enumerate(parameters):
        plt.subplot(4, 3, i + 1)  # Create a grid of subplots
        param.plot()
        plt.title(parameter_names[i])
    plt.tight_layout()
    if out_dir:
        plt.savefig(f"{out_dir}/all_variables.png")
    plt.show()
    plt.close()


def presence_data_extraction(nearest_habitat_values):
    presence_habitat_df = nearest_habitat_values.to_dataframe().reset_index()
    presence_habitat_df = presence_habitat_df.rename(
        columns={"x": "longitude", "y": "latitude"}
    )

    presence_habitat_df = presence_habitat_df.replace(-9999, np.nan)
    presence_habitat_df = presence_habitat_df.dropna()
    return presence_habitat_df


def background_data_extraction(
    habitat_parameter_cube, nearest_habitat_values, presence_size, background_ratio=20
):
    ## reading the FULL CUBE for Luxembourg, set to na points where no plant growth is possible (water & sealed areas)
    background_cube = habitat_parameter_cube.where(
        (habitat_parameter_cube["water_mask"] == 0)
        & (habitat_parameter_cube["not_sealed_mask"] == 1)
    )

    ## set to na points where species has occurred
    # get coordinates from occurrence cube
    x_coords_grid = list(nearest_habitat_values.x.values)
    y_coords_grid = list(nearest_habitat_values.y.values)

    # Create a boolean mask for species occurrences
    mask = background_cube.assign(
        mask=lambda x: (x.d01_L_light * 0 + 1).astype(bool)
    ).drop_vars(background_cube.keys())
    mask.mask.loc[dict(x=x_coords_grid, y=y_coords_grid)] = False
    # set locations of species occurrence to na
    background_habitat_values = background_cube.where(mask.mask)
    # Step 4: Convert the non-occurrence habitat data to a DataFrame and remove masked values (all na)
    background_habitat_df = (
        background_habitat_values.to_dataframe().reset_index().dropna()
    )

    # Randomly sample background points after filtering
    target_bg_size = min(len(background_habitat_df), background_ratio * presence_size)

    background_habitat_df = background_habitat_df.sample(
        n=target_bg_size, random_state=42
    )

    ## MAXENT: data preparation
    background_data = background_habitat_df.rename(
        columns={"x": "longitude", "y": "latitude"}
    )

    # Replace -9999 by NaN
    background_data = background_data.replace(-9999, np.nan)

    # Drop rows with NaN values
    background_data = background_data.dropna()

    return background_data

def plot_shap_detailed(species, shap_values, sample_features):

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample_features, show=False)
    plt.title("SHAP Summary Plot for Maxent (" + species + ")")
    plt.tight_layout()
    plt.savefig(
        f"../../images/shapMaxent/Shap_Detailed_{species}_{len(sample_features)}.png",
        dpi=300,
    )
    plt.close()


def plot_shap_global(species, shap_values, sample_features):

    # Compute global SHAP values using the sampled dataset
    global_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Compute correlation between sampled feature values and SHAP values
    feature_shap_correlation = []
    for i, feature in enumerate(sample_features.columns):
        correlation = np.corrcoef(sample_features[feature], shap_values[:, i])[
            0, 1
        ]  # Pearson correlation
        feature_shap_correlation.append(correlation)

    # Create a DataFrame with feature importance and correlation
    feature_importance_df = pd.DataFrame(
        {
            "Feature": sample_features.columns,  # Use sampled features
            "Mean_Abs_SHAP": global_shap_values,
            "Correlation": feature_shap_correlation,
        }
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="Mean_Abs_SHAP", ascending=False
    )

    sorted_features = feature_importance_df["Feature"].values
    sorted_shap_values = feature_importance_df["Mean_Abs_SHAP"].values
    sorted_correlation_values = feature_importance_df["Correlation"].values

    colors = ["green" if corr > 0 else "orange" for corr in sorted_correlation_values]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        sorted_features, sorted_shap_values, color=colors
    )  # Bars now sorted correctly
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title(f"Global Feature Importance for Maxent Suitability ({species})")

    for bar, corr in zip(bars, sorted_correlation_values):
        plt.text(
            bar.get_width() / 2,  # Position in the middle of the bar
            bar.get_y() + bar.get_height() / 2,  # Centered in the bar
            f"{corr:.2f}",  # Format to 2 decimal places
            va="center",
            ha="center",  # Center text inside bar
            fontsize=10,
            color="white" if abs(corr) > 0 else "black",  # Improve readability
            fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(color="green", label="Higher values increase suitability"),
        mpatches.Patch(color="orange", label="Lower values increase suitability"),
    ]
    plt.legend(handles=legend_patches, loc="lower right")

    plt.gca().invert_yaxis()  # Ensure most important feature remains on top
    plt.tight_layout()

    plt.savefig(
        f"../../images/shapMaxent/Shap_Global_{species}_{sample_size}.png", dpi=300
    )
    plt.close()

