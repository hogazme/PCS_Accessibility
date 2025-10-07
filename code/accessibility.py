#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import ast
from geopy.distance import geodesic

# Function to map GEOIDs between different years
def cbg_mapping(year):
    cbg = pd.read_csv(f'cbg_{year}_allcbgs.csv')
    cbg['census_block_group'] = cbg['census_block_group'].apply(lambda x: str(x).zfill(12))
    cbg.rename(columns={'census_block_group': f'GEOID_{year}'}, inplace=True)
    cbg = gpd.GeoDataFrame(cbg, geometry=gpd.points_from_xy(cbg['longitude'], cbg['latitude']), crs='EPSG:4269')
    return cbg

# Function to compute accessibility to nearby stations
def nearby_stations_accessibility(sample_cbg, sample_pcs, k):
    pcs_coords = list(zip(sample_pcs['pcs_y'], sample_pcs['pcs_x']))
    cbg_coords = list(zip(sample_cbg['cbg_y'], sample_cbg['cbg_x']))
    facility_tree = cKDTree(pcs_coords)
    nearby_facility_distances = []
    for pfas_point in cbg_coords:
        distances, indices = facility_tree.query(pfas_point, k=k)
        nearby_dists = [geodesic(pfas_point, pcs_coords[i]).kilometers for i in indices]
        nearby_dists = [-pcsdist for pcsdist in nearby_dists] 
        nearby_facility_distances.append(nearby_dists)
    return nearby_facility_distances 

# Function to process distance-based accessibility metrics
def process_accessibility(cbg, pcs):
    k = pcs.shape[0]
    cbg['nearby_pcs_dists'] = nearby_stations_accessibility(cbg, pcs, k)
    for index, row in cbg.iterrows():
        cbg.loc[index, 'dist_accessibility'] = np.sum(np.exp(row['nearby_pcs_dists']))
    return cbg

# Function to compute Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Function to assign POIs to nearest stations
def poi_assign_nearest_pcs(poi, pcs):
    pcs_coords = list(zip(pcs['pcs_y'], pcs['pcs_x']))
    poi_coords = list(zip(poi['poi_y'], poi['poi_x']))
    facility_tree = cKDTree(pcs_coords)
    _, nearest_indices = facility_tree.query(poi_coords)
    poi['pcs_id_tree'] = nearest_indices.tolist()
    poi['pcs_id_tree'] = poi['pcs_id_tree'].apply(lambda x: pcs.iloc[x]['pcs_id_tree'])
    poi = poi.merge(pcs[['pcs_id_tree', 'pcs_x', 'pcs_y', 'total_evse']], on='pcs_id_tree', how='left')
    poi['distance_km'] = poi.apply(lambda row: haversine(row['poi_y'], row['poi_x'], row['pcs_y'], row['pcs_x']), axis=1)
    poi.reset_index(drop=True, inplace=True)
    return poi

# Function to expand POI data with visit percentages
def poi_expand(poi, pcs, mp):
    poi = poi_assign_nearest_pcs(poi, pcs)
    poi = poi.merge(mp, on='PLACEKEY', how='left')
    poi.dropna(subset=['VISITOR_HOME_CBGS'], inplace=True)
    poi.reset_index(drop=True, inplace=True)
    poi['VISITOR_HOME_CBGS'] = poi['VISITOR_HOME_CBGS'].apply(ast.literal_eval)
    dataframes = [pd.DataFrame(row['VISITOR_HOME_CBGS'].items(), columns=['origin_geoid', 'visit_pct'])
                  .assign(PLACEKEY=row['PLACEKEY']) for _, row in poi.iterrows()]
    expanded_df = pd.concat(dataframes, ignore_index=True)
    expanded_df['sum_pct'] = expanded_df.groupby('origin_geoid')['visit_pct'].transform('sum')
    expanded_df['visit_pct'] = expanded_df['visit_pct'] / expanded_df['sum_pct']
    expanded_df.drop(columns=['sum_pct'], inplace=True)
    poi = poi.merge(expanded_df, on='PLACEKEY', how='right')
    poi.drop(columns=['geometry'], inplace=True)
    poi_cbg = poi[poi['distance_km'] <= 0.25][['origin_geoid', 'visit_pct']]
    poi_cbg = poi_cbg.groupby('origin_geoid').sum().reset_index()
    return poi_cbg

# Main function to process visit- and distance-based accessibility
def process_visit_distance_access(metro_name):
    cbg20 = pd.read_csv('/data/cbg_mapping.csv')
    cbg20['GEOID_2019'] = cbg20['GEOID_2019'].apply(lambda x: str(x).zfill(12))
    cbg20['GEOID_2020'] = cbg20['GEOID_2020'].apply(lambda x: str(x).zfill(12))

    cbg = gpd.read_file(f'/data/cbg_{metro_name}.geojson')
    cbg = cbg.merge(cbg20, left_on='GEOID', right_on='GEOID_2020', how='left')
    cbg['cbg_x'] = cbg.geometry.centroid.x
    cbg['cbg_y'] = cbg.geometry.centroid.y

    pcs = gpd.read_file(f'/data/pcs_{metro_name}.geojson')
    cbg = process_accessibility(cbg, pcs)

    # # the poi dataset is not publicly available therefore it is not included in the repo
    # poi = gpd.read_file('poi_{}.geojson'.format(metro_name))
    # # monthly patterns dataset is not publicly available therefore it is not included in the repo
    # mp_cols = ['PLACEKEY', 'VISITOR_HOME_CBGS']
    # mp = pd.read_csv('{}_2023_monthly_patterns.csv'.format(metro_name), usecols=mp_cols)
    # expanded_poi = poi_expand(poi, pcs, mp)

    expanded_poi = pd.read_csv(f'/data/expanded_poi_{metro_name}.csv')
    cbg = cbg.merge(expanded_poi, left_on='GEOID_2019', right_on='origin_geoid', how='left')
    return cbg

# Execute the main function for the specified metro area
if __name__ == "__main__":
    metro_name = 'Albany--Schenectady, NY'
    result = process_visit_distance_access(metro_name)
    result.to_file(f'/results/cbg_{metro_name}.geojson', driver='GeoJSON')
    print(result)