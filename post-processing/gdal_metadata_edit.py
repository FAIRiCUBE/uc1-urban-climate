"""
Set Band descriptions
Usage:
    python set_band_desc.py /path/to/file.ext band desc nodata [band desc nodata...]
Where:
    band = band number to set (starting from 1)
    desc = band description string (enclose in "double quotes" if it contains spaces)
    nodata = NoData value for the band (use 'None' if no NoData value is to be set)
Example:
    python set_band_desc.py /path/to/dem.tif 1 "Band 1 desc" None 2 "Band 2 desc" 0 3 "Band 3 desc" -9999

"""
import sys
from osgeo import gdal
import glob

def set_band_descriptions(filepath, bands):
    """
    filepath: path/virtual path/uri to raster
    bands:    ((band, description, nodata), (band, description, nodata),...)
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc, nodata in bands:
        rb = ds.GetRasterBand(band)
        print(f"band {band} description: {rb.GetDescription()}")
        rb.SetDescription(desc)
        if nodata is not None:
            rb.SetNoDataValue(nodata)
    ds = None

if __name__ == '__main__':
    # filepath = sys.argv[1]
    # bands = [int(i) for i in sys.argv[2::3]]
    # names = sys.argv[3::3]
    # nodata_values = [float(i) if i.lower() != 'none' else None for i in sys.argv[4::3]]
    folder_path = "N:/C2205_FAIRiCUBE/f02_data/d060_data_LUXEMBOURG/f05_distribution/f03_distribution_maxent/2025*_2169.tif"
    for filepath in glob.glob(folder_path):
        print(filepath)
        bands = [1]
        names = ["Suitability index"]
        nodata_values = [-1]
        set_band_descriptions(filepath, zip(bands, names, nodata_values))
