# European cities atlas

A collection of features for ~700 European cities, for the reference year 2018.

## Features

The features are divided in three main thematic areas: land, climate and socioeconomic characteristics. Find more information about the features in the codebook `eu_cities_atlas_codebook.csv`.
Codelists for categorical features are in the same folder `codelist_<feature>.csv`.

## Cities

City selection (and outline polygon) is taken from the Eurostat Urban Atlas. More information [here](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/urban-audit). The original list of cities with geometries can be downloaded at these links:

- EPSG:4326 (WGS84) <https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_01M_2018_4326_CITIES.geojson>
- EPSG:3035 <https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_01M_2018_3035_CITIES.geojson>

Note: the dataset `eu_cities_atlas.geojson` only contains the city outline in CRS EPSG:4326.

## Metadata

STAC metadata record available on the FAIRiCUBE Data Catalogue: https://catalog.eoxhub.fairicube.eu/collections/index/items/eu_cities_atlas

## Example usage

Clustering analysis of European cities: check out this interactive demo notebook: `notebooks\demo\cities_clustering_interactive_demo.ipynb`.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
