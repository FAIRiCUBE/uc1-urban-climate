## GDAL tool for splitting/tiling raster file into tiles
## PARAMETERS: NEW
##		-of Format (→ COG = Cloud Optimized GeoTIFF)
##		-ps Tilesize (Number of cells per line and column)
## https://docs.sentinel-hub.com/api/latest/api/byoc/#gdal-example-command

cd C:\ProgramData\Anaconda3\Scripts

python gdal_retile.py -of COG -co BLOCKSIZE=1024 -co COMPRESS=DEFLATE -co PREDICTOR=YES -ps 100000 100000 -levels 1 -s_srs EPSG:3035 -r near -tileIndexField clc18_tile_ -targetDir R:/INSPIRE_Annex_II/02_Land_cover/01_Corine_Land_Cover_CLC/8_CLC_plus/CLCplus_Backbone2018/clc2018/jedi_upload/tiles/ R:/INSPIRE_Annex_II/02_Land_cover/01_Corine_Land_Cover_CLC/8_CLC_plus/CLCplus_Backbone2018/clc2018/jedi_upload/raw/CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif


254 no data

1. zip all tiles - name of zip = name of dim Clc_plus2018_10m

1. First, create a zip that includes the tiles.  The name of this zip file will be the name of the field in the dimension.
2. Then zip it again. This name doesn´t really matter. 
