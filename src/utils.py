import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import rasterio as rio
import IPython.display
from typing import Any, Optional, Tuple

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