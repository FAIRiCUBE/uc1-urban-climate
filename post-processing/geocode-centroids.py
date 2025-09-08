import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from shapely.geometry import Point
from tqdm import tqdm
import pyproj

def get_location_name(lat, lon, geolocator, max_retries=3):
    """
    Get location name from latitude and longitude using Nominatim
    with retry mechanism for handling timeouts.
    """
    for attempt in range(max_retries):
        try:
            # Format coordinates for geocoding
            coordinates = f"{lat}, {lon}"
            
            # Get location information
            location = geolocator.reverse(coordinates)
            
            # Return address if successful
            if location and location.address:
                return location.address
            return "Unknown location"
            
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt == max_retries - 1:
                return f"Geocoding error: {str(e)}"
            # Wait before retrying (exponential backoff)
            time.sleep(1 * (2 ** attempt))
    
    return "Failed to geocode after retries"

def geocode_geodataframe(gdf, user_agent="geocoding_script"):
    """
    Process a GeoDataFrame by computing centroids and finding location names.
    Returns the original GeoDataFrame with added location information.
    """
    # Create a copy of the input GeoDataFrame to avoid modifying the original
    result_gdf = gdf.copy()
    
    # Initialize the geocoder with a user agent (required by Nominatim)
    geolocator = Nominatim(user_agent=user_agent)
    
    # Create empty lists to store centroids and location names
    centroid_points = []
    location_names = []
    
    # Process each geometry
    print("Computing centroids and geocoding locations...")
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        # Compute centroid of the geometry
        centroid = row.geometry.centroid
        centroid_points.append(Point(centroid.x, centroid.y))
        # transform coordinates from EPSG:2169 to EPSG:4326 using pyproj
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(2169),
            pyproj.CRS.from_epsg(4326),
            always_xy=True  # Ensures x,y (easting,northing) input order
        )
        lon, lat = transformer.transform(centroid.x, centroid.y)
        # Get location name for the centroid
        # Note: We need to switch coordinates for geopy (lat, lon) vs (x, y) in geometry
        location_name = get_location_name(lat, lon, geolocator)
        
        # Add a delay to respect Nominatim usage policy (max 1 request per second)
        time.sleep(1)
        
        # Store the location name
        location_names.append(location_name)
    
    # Add new columns to the GeoDataFrame
    result_gdf["centroid"] = centroid_points
    result_gdf["location_name"] = location_names
    
    return result_gdf

def main():
    # File path to your GeoDataFrame (in a format that geopandas can read)
    # For example: shapefile, GeoJSON, etc.
    file_path = "N:/C2205_FAIRiCUBE/f02_data/d060_data_LUXEMBOURG/f04_vector_collection/f06_POIs/AOIs_school_kind_100m_buffer.gpkg"  # Change this to your file path
    
    try:
        # Read the GeoDataFrame
        print(f"Reading geodata from {file_path}...")
        gdf = gpd.read_file(file_path)
        
        # Check if the GeoDataFrame has a geometry column
        if not gdf.geometry.any():
            print("Error: The GeoDataFrame does not have valid geometries.")
            return
        
        # Process the GeoDataFrame
        result_gdf = geocode_geodataframe(gdf)
        
        # Save the results to a new file
        output_file = "geocoded_" + file_path.split("/")[-1]
        result_gdf.to_file(output_file)
        print(f"Successfully geocoded {len(result_gdf)} geometries.")
        print(f"Results saved to {output_file}")
        
        # Display a sample of the results
        print("\nSample of results:")
        print(result_gdf[["location_name"]].head())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
