
import datetime as dt
from datetime import datetime
# Utilities
import boto3
import dateutil
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import rasterio
import rasterio.mask
from rasterio.plot import show_hist
from rasterio.plot import show
from rasterio.windows import Window
from rasterio.windows import from_bounds # for aoi
import pprint
import random
import fiona
import numpy as np
from shapely.geometry import mapping, Polygon
from shapely import geometry
from loguru import logger

# Sentinel Hub
from sentinelhub import (
    CRS,
    BBox,
    ByocCollection,
    ByocCollectionAdditionalData,
    ByocCollectionBand,
    ByocTile,
    DataCollection,
    DownloadFailedException,
    MimeType,
    SentinelHubBYOC,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
    os_utils,
)

config = SHConfig()
config.instance_id = os.environ.get("SH_INSTANCE_ID")
config.sh_client_id = os.environ.get("SH_CLIENT_ID")
config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
config.aws_access_key_id = os.environ.get("username")
config.aws_secret_access_key = os.environ.get("password")

class QualityChecker:
    
    def __init__(self):
        # TODO connector to sentinelhub
        self.sh_config = ''
        # original raster dataset
        self.r_original = ''
        self.r_original_path = ''
        # transformed raster dataset
        self.r_transformed_path = ''
        # collectionId of ingested raster in sh
        self.r_ingested_collectionId = ''
        self.r_ingested_collectionName = ''
        self.r_DataCollection = ''
        self.r_ingested_collection_metadata = ''
        self.r_ingested_tiles_metadata = ''
        # qc report
        self.qc_report = pd.DataFrame()
    
    # initialize connection to SH
    def SH_connect(self):
        self.sh_config = SHConfig()
        self.sh_config.instance_id = os.environ.get("SH_INSTANCE_ID")
        self.sh_config.sh_client_id = os.environ.get("SH_CLIENT_ID")
        self.sh_config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
        self.sh_config.aws_access_key_id = os.environ.get("username")
        self.sh_config.aws_secret_access_key = os.environ.get("password")
        
    # open original raster dataset using local filesystem
    # TODO access file remotely
    # TODO support for multiband
    # TODO                           
    # startTimestamp,
    # endTimestamp
    def get_original_metadata(self, path):
        with rasterio.open(path) as raster:
            metadata_dict = {'raster_nbands' : raster.count,
                                    'raster_bounds' : raster.bounds,
                                    'raster_crs' : raster.crs,
                                    'raster_dtype' : raster.dtypes[0],
                                    'raster_nodata' : raster.nodata,
                                    'pixelSizeX' : raster.transform[0],
                                    'pixelSizeY' : -raster.transform[4]}
        pprint.pprint(metadata_dict)
        return metadata_dict

    ## OLD
    # define SH DataCollection from collectionId
    # to list of ingested data collections use utils.list_byoc_collections()
    # TODO catch wrong id error
    def set_sh_DataCollection(self, collectionId, collection_name = ''):
        if (collection_name == ''):
            self.r_ingested_collectionName = 'ingested_collection'
            print(f"Defaulting to DataCollection name {self.r_ingested_collectionName} for collectionId {collectionId}")
        else:
            self.r_ingested_collectionName = collection_name
            
        self.r_ingested_collectionId = collectionId
        self.r_ingested_DataCollection = DataCollection.define_byoc(self.r_ingested_collectionId, name=name_of_ingestion)
        return self.r_ingested_DataCollection
    
    # get collection metadata
    def get_sh_collection_metadata(self, collectionId, collection_name = ''):
        # Initialize SentinelHubBYOC class
        self.r_ingested_collectionId = collectionId
        byoc = SentinelHubBYOC(config=self.sh_config)
        self.r_ingested_collection_metadata = byoc.get_collection(self.r_ingested_collectionId)
        pprint.pprint(self.r_ingested_collection_metadata)
        return self.r_ingested_collection_metadata
    
    # get tiles metadata
    def get_sh_tiles_metadata(self, verbose=False):
        if(self.r_ingested_collection_metadata == ''):
            print("Get collection metadata first")
            print("Use get_sh_collection_metadata(collection_id)")
            return None
        else:
            byoc = SentinelHubBYOC(config=self.sh_config)
            self.r_ingested_tiles_metadata = list(byoc.iter_tiles(self.r_ingested_collection_metadata))
            if(verbose==True):
                pprint.pprint(self.r_ingested_tiles_metadata)
            return self.r_ingested_tiles_metadata
    
    
    # set up qc report dataframe and csv file
    def qc_report_init(self):
        self.qc_report = pd.DataFrame(columns=["collection_name", "collection_id", "timestamp", "qc_check_code", "qc_check_name", "qc_check_result", "original_data", "ingested_data"])
    
    # add row to qc report
    def qc_report_append(self, new_row):
        new_row['timestamp'] = str(datetime.now())
        self.qc_report = pd.concat([self.qc_report, pd.DataFrame([new_row])])
    
    
    #close qc report and save to file
    def qc_report_close(self, qc_folder="", mode="w"):
        self.qc_report['collection_name'] = self.r_ingested_collectionName
        self.qc_report['collection_id'] = self.r_ingested_collectionId
        save_path = qc_folder+f'QC_{self.r_ingested_collectionName}_raster_edc.csv'
        self.qc_report.to_csv(save_path, index=False, mode=mode)
        print("QC report saved to "+ save_path)
        print(self.qc_report)

        
    # check metadata of ingested raster
    # TODO read values from file or dictionary
    # TODO write report to file
    # QC check 1 - Metadata
    def check_metadata(self,
                       crs, 
                       dtype, 
                       nodata, 
                       bounds,
                       n_bands,
                       pixelSizeX, 
                       pixelSizeY,
                       startTimestamp,
                       endTimestamp):
        #####################################
        # initialize metadata report
        self.qc_report_init()
                     
        #####################################
        # crs
        ingested_crs = self.r_ingested_tiles_metadata[0]['tileGeometry'][ 'crs']['properties'][ 'name']#.split('crs:')[-1]
        if(rasterio.CRS.from_user_input(ingested_crs) == crs):
            self.qc_report_append({"qc_check_code" : "1.1", 
                              "qc_check_name" : "CRS",
                              "qc_check_result": "OK",
                              "original_data" : crs,
                              "ingested_data" : ingested_crs})
        else:
            self.qc_report_append({"qc_check_code" : "1.1", 
                  "qc_check_name" : "CRS",
                  "qc_check_result": "error",
                  "original_data" : crs,
                  "ingested_data" :  ingested_crs})
        
        #####################################
        # cell size
        ingested_size_x = self.r_ingested_tiles_metadata[0]['additionalData']['minMetersPerPixel'] 
        if(pixelSizeX == ingested_size_x  and pixelSizeY == ingested_size_x):
            self.qc_report_append({"qc_check_code" : "1.2", 
                  "qc_check_name" : "cell_size_x",
                  "qc_check_result": "OK",
                  "ingested_data" : ingested_size_x,
                  "original_data" : pixelSizeX})
            self.qc_report_append({"qc_check_code" : "1.2", 
                  "qc_check_name" : "cell_size_y",
                  "qc_check_result": "OK",
                  "ingested_data" : ingested_size_x,
                  "original_data" : pixelSizeY})
        else:
            self.qc_report_append({"qc_check_code" : "1.2", 
                  "qc_check_name" : "cell_size_x",
                  "qc_check_result": "error",
                  "ingested_data" : ingested_size_x,
                  "original_data" : pixelSizeX})
            self.qc_report_append({"qc_check_code" : "1.2", 
                  "qc_check_name" : "cell_size_y",
                  "qc_check_result": "error",
                  "ingested_data" : ingested_size_x,
                  "original_data" : pixelSizeY})
        
        #####################################
        # bounds
        tile_bounds = np.array([tile['tileGeometry']['coordinates'][0][0:4:2] for tile in self.r_ingested_tiles_metadata])
        left = min(tile_bounds[:,0,0])
        bottom = min(tile_bounds[:,1,1])
        top = max(tile_bounds[:,0,1])
        right = max(tile_bounds[:,1,0])
        if(left == bounds.left and 
           bottom == bounds.bottom and
           right == bounds.right and
           top == bounds.top):
            self.qc_report_append({"qc_check_code" : "1.3", 
                  "qc_check_name" : "bounds",
                  "qc_check_result": "OK",
                  "original_data" : [bounds.left, bounds.bottom, bounds.right, bounds.top],
                  "ingested_data" : [left, bottom, right, top]})
        else:
            self.qc_report_append({"qc_check_code" : "1.3", 
                  "qc_check_name" : "bounds",
                  "qc_check_result": "error",
                  "original_data" : [bounds.left, bounds.bottom, bounds.right, bounds.top],
                  "ingested_data" : [left, bottom, right, top]})
            
        #####################################
        # number of bands
        n_bands_ingested = len(self.r_ingested_collection_metadata['additionalData']['bands'])
        if(n_bands_ingested == n_bands):                
            self.qc_report_append({"qc_check_code" : "1.4", 
                  "qc_check_name" : "n_bands",
                  "qc_check_result": "OK",
                  "ingested_data" : n_bands_ingested,
                  "original_data" : n_bands})
        else:
            self.qc_report_append({"qc_check_code" : "1.4", 
                  "qc_check_name" : "n_bands",
                  "qc_check_result": "error",
                  "ingested_data" : n_bands_ingested,
                  "original_data" : n_bands})
        
        
        #####################################
        # data type
        # get name of band first
        # TODO implement for multiband datasets
        band_name = list(self.r_ingested_collection_metadata['additionalData']['bands'])[0]
        data_type = self.r_ingested_collection_metadata['additionalData']['bands'][band_name]['sampleFormat']
        bitdepth = self.r_ingested_collection_metadata['additionalData']['bands'][band_name]['bitDepth']
        dtype_ingested = data_type.lower()+str(bitdepth)
                
        if(dtype_ingested == dtype):
            self.qc_report_append({"qc_check_code" : "1.5", 
                  "qc_check_name" : "dtype",
                  "qc_check_result": "OK",
                  "ingested_data" : dtype_ingested,
                  "original_data" : dtype})
        else:
            self.qc_report_append({"qc_check_code" : "1.5", 
                  "qc_check_name" : "dtype",
                  "qc_check_result": "error",
                  "ingested_data" : dtype_ingested,
                  "original_data" : dtype})
            
        
        #####################################
        # nodata
        
        nodata_ingested = self.r_ingested_collection_metadata['noData']
        
        if(nodata_ingested == nodata):
            self.qc_report_append({"qc_check_code" : "1.6", 
                  "qc_check_name" : "nodata",
                  "qc_check_result": "OK",
                  "original_data" : nodata,
                  "ingested_data" : nodata_ingested})
        else:
            self.qc_report_append({"qc_check_code" : "1.6", 
                  "qc_check_name" : "nodata",
                  "qc_check_result": "error",
                  "original_data" : nodata,
                  "ingested_data" : nodata_ingested})
        
        
        #####################################
        # startTimestamp, endTimestamp
        if(self.r_ingested_collection_metadata['additionalData']['hasSensingTimes'] == 'NO'):
            self.qc_report_append({"qc_check_code" : "1.7", 
                  "qc_check_name" : "timestamp",
                  "qc_check_result": "error",
                  "original_data" : "no_sensing_time",
                  "ingested_data" : "no_sensing_time"})
        else:
            startTimestamp_ingested = self.r_ingested_collection_metadata['additionalData']['fromSensingTime']
            endTimestamp_ingested = self.r_ingested_collection_metadata['additionalData']['toSensingTime']
            if(startTimestamp_ingested == startTimestamp and endTimestamp_ingested == endTimestamp):
                self.qc_report_append({"qc_check_code" : "1.7", 
                  "qc_check_name" : "timestamp",
                  "qc_check_result": "OK",
                  "original_data" : [startTimestamp, endTimestamp],
                  "ingested_data" : [startTimestamp_ingested, endTimestamp_ingested]})
            else:
                self.qc_report_append({"qc_check_code" : "1.7", 
                  "qc_check_name" : "timestamp",
                  "qc_check_result": "error",
                  "original_data" : [startTimestamp, endTimestamp],
                  "ingested_data" : [startTimestamp_ingested, endTimestamp_ingested]})

            
     
    # get SentinelHub statisitcs
    # TODO check that bbox is in the right crs
    def get_stats_sh(self, bbox_coords):
        # set bounding box
        ingested_crs = self.r_ingested_tiles_metadata[0]['tileGeometry'][ 'crs']['properties'][ 'name'].split(':')[-1]
        resolution = self.r_ingested_tiles_metadata[0]['additionalData']['minMetersPerPixel'] 
        bbox=  BBox(bbox=bbox_coords, crs=CRS(ingested_crs))
        print(bbox, bbox_to_dimensions(bbox, resolution), resolution, CRS(ingested_crs))
        # set data collection
        self.r_ingested_collectionName = self.r_ingested_collection_metadata['name']
        self.r_ingested_DataCollection = DataCollection.define_byoc(self.r_ingested_collectionId, name=self.r_ingested_collectionName)
        
        # set evalscript
        band_name = list(self.r_ingested_collection_metadata['additionalData']['bands'])[0]
        n_bands_ingested = len(self.r_ingested_collection_metadata['additionalData']['bands'])
        evalscript = f"""

        //VERSION=3
        function setup() {{
          return {{
            input: ["{band_name}"],
            output: {{ 
                bands: {n_bands_ingested},
                sampleType: "UINT16" // raster format will be UINT16
                }}

          }};
        }}

        function evaluatePixel(sample) {{
          return [sample.{band_name}];
        }}
        """
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.r_ingested_DataCollection,
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG)
            ],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution),
            config=self.sh_config)

        data = request.get_data()[0]
        
        statistics = {
            "min" : round(np.min(data),3),
            "max" : round(np.max(data),3),
            "mean": round(np.mean(data),3),
            "std" : round(np.std(data),3),
            "count": len(data.ravel()),
            #TODO "no_data_count", "distinct_count", "date_range", count of most frequent vale
        }
        return statistics
        
    # check statistics of ingested raster
    def check_statistics(self, bbox_coords, path):
        # open raster file and get statistics
        with rasterio.open(path) as src:
            x1, y2, x2, y1 = bbox_coords
            band_data = src.read(1, window=from_bounds(x1, y2, x2, y1, src.transform))
            stats_original = {
            "min" : round(band_data.min(),3),
            "max" : round(band_data.max(),3),
            "mean": round(band_data.mean(),3),
            "std" : round(band_data.std(),3),
            "count": len(band_data.ravel()),
            #TODO "no_data_count", "distinct_count", "date_range", count of most frequent vale
            }
            print(band_data.shape)
        # get statistics of ingested raster
        stats_ingested = self.get_stats_sh(bbox_coords)
        # compare
        qc_check_codes = [2.1, 2.2, 2.3, 2.4, 2.5]
        qc_check_names = stats_ingested.keys()
        for code, name in zip(qc_check_codes, qc_check_names):
            if(stats_original[name] == stats_ingested[name]):
                self.qc_report_append({"qc_check_code" : code, 
                  "qc_check_name" : name,
                  "qc_check_result": "OK",
                  "original_data" : stats_original[name],
                  "ingested_data" : stats_ingested[name]})
            else:
                 self.qc_report_append({"qc_check_code" : code, 
                  "qc_check_name" : name,
                  "qc_check_result": "error",
                  "original_data" : stats_original[name],
                  "ingested_data" : stats_ingested[name]})

        

if __name__ == "__main__":
    ## Test quality checker with raster dataset ingested to SentinelHub
    logger.info("Test QualityChecker original data <-> ingested data")
    
    # change the following parameters to point to your data
    collection_name = "environmental_zones_1km"               # should be the collection name!!!"
    collection_id   ='5b45916e-6704-4581-824f-4d713198731b'  # collection ID
    original_raster ="./../../../../s3/data/d005_env_zones/raw_env_zones/env_zones_1km_3035.tif"   ## path to original data
    
    logger.info(f"Collection Id: {collection_id}")
    logger.info(f"Collection Name: {collection_name}")
    logger.info(f"path to original raster: {original_raster}")

    #set up quality checker
    qc = QualityChecker()
    qc.SH_connect()
    # get metadata
    c_meta = qc.get_sh_collection_metadata(collection_id)
    t_meta = qc.get_sh_tiles_metadata(verbose=False) # verbose=True prints metadata for all tiles
    o_meta = qc.get_original_metadata(original_raster)
    
    startTimestamp = endTimestamp = '2018-01-01T00:00:00Z'
    # compare metadata of original and ingested raster
    qc.check_metadata(o_meta['raster_crs'], 
                       o_meta['raster_dtype'], 
                       o_meta['raster_nodata'], 
                       o_meta['raster_bounds'],
                       o_meta['raster_nbands'],
                       o_meta['pixelSizeX'], 
                       o_meta['pixelSizeY'],
                       startTimestamp,
                       endTimestamp)
    
    # compare statistics of original and ingested raster
    # compute statisitcs only for an area of interest
    # maximum allowed size is (2500,2500) pixels
    
    # set bounding box in EPSG:3035 (unit: meters)
    x1 =      900000   # Left
    y1 =      5500000  # Top
    x2 = x1 + 20000   # Right
    y2 = y1 - 20000   # Bottom
    bbox_coords = x1, y2, x2, y1
    
    qc.check_statistics(bbox_coords, original_raster)
    
    # save results to file
    qc.qc_report_close()
