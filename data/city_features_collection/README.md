# City features collection

A collection of features for ~700 European cities, for the reference year 2018.

## Features

The features are divided in three main thematic areas: land, climate and socioeconomic characteristics. Find more information about the features in the codebook `cities_features_collection_codebook.csv`.
Codelists for categorical features are in the same folder `codelist_<feature>.csv`.

## Cities

City selection (and outline polygon) is taken from the Eurostat Urban Atlas. More information [here](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/urban-audit). The original list of cities with geometries can be downloaded at these links:

- WGS84 <https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_01M_2018_3857_CITIES.geojson>
- EPSG:3035 <https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_01M_2018_3035_CITIES.geojson>

Note: the dataset `city_features_collection.geojson` only contains the city outline in CRS WGS84.

## Metadata

<link>

## Example usage

Clustering analysis of European cities: check out this interactive demo notebook: `notebooks\demo\cities_clustering_interactive_demo.ipynb`.
