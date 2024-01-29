QC_environmental_zones_1km 
---------------------------------------------------------- 
START-time: 2023-12-14 13:56:33.135061 
(1) CHECK 1 - spatial check for raster files:   
----------------------------------------------   
./../../../../s3/data/d005_env_zones/raw_env_zones/env_zones_1km_3035.tif   
 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
check 1.1 (CRS) -START 
-RASTER:  
  EPSG code: EPSG:3035 
-CUBE_TILE:  
  EPSG code: EPSG:3035 
check 1.1 - EPSG (crs): OK   
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
check 1.2 (cellsize)-START  
-RASTER:  
  Pixel size x: 100.0 
  Pixel size y: 100.0 
-CUBE_TILE:  
  CUBE size x: 100.0 
  CUBE size y: 100.0 
check 1.2 - cell size  :  OK   
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
check 1.3 (extend)-START 
-RASTER:  
  extent: BoundingBox(left=900000.0, bottom=900000.0, right=7400000.0, top=5500000.0) 
  with: 65000 
  height: 46000 
  --------  
  left: 900000.0        
  bottom: 900000.0    
  right: 7400000.0       
  top: 5500000.0            
-CUBE_TILE:  
  left: 900000.0         
  bottom: 900000.0         
  right: 7400000.0         
  top: 5500000.0         
check 1.3 - extend  :  OK   
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
check 1.4 ( statistics)-START  
-RASTER:  
  max raster value: 7 
  min raster value: 7 
  avg raster value: 7.0 
-CUBE_TILE:  
  max cube value: 7 
  min cube value: 7 
  avg cube  value: 7.0 
check 1.4 - data : OK   
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
check 1.5 (data type) -START 
-RASTER:  
  raster data type: uint8 
  raster nodata value: 0.0 
-CUBE_TILE:  
  cube data type: uint8 
  cube nodata value:  
check 1.5 - data : OK   
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
check 1.6 (number of bands) -START 
-RASTER:  
  raster number of bands: 1 
-CUBE_TILE:  
  cube number of bands: 1 
check 1.6 - #bands : OK   
------------------------------------------------------------------------------------ 
END-time: 2023-12-14 13:56:33.572778 
