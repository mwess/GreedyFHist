import geojson
import pandas as pd
import numpy as np

def ring2table(coords):
    return pd.DataFrame(np.array(coords))

def multipolygon2table(coordinates, id_):
    mp_table = pd.DataFrame()
    for outer_idx, polygon in enumerate(coordinates):
        p_table = polygon2table(polygon)
        p_table['outer_idx'] = outer_idx
        mp_table = pd.concat([mp_table, p_table])
    mp_table['id'] = id_
    mp_table['type'] = 'MultiPolygon'
    return mp_table

def polygon2table(coordinates, id_ = None):
    poly_table = pd.DataFrame()
    for cur_idx, poly_array in enumerate(coordinates):
        poly_table_ = ring2table(poly_array)
        poly_table_['polygon_idx'] = cur_idx
        poly_table = pd.concat([poly_table, poly_table_])
    poly_table['type'] = 'Polygon'
    if id_ is not None:
        poly_table['id'] = id_
    return poly_table

def point2table(coordinates, id_ = None):
    table = pd.DataFrame(np.array(coordinates)).transpose()
    if id_ is not None:
        table[id_] = id_
    table['type'] = 'Point'
    return table

def multipoint2table(coordinates, id_ = None):
    table = ring2table(coordinates)
    if id_ is not None:
        table[id_] = id_
    table['type'] = 'MultiPoint'
    return table    

def linestring2table(coordinates, id_ = None):
    table = ring2table(coordinates)
    if id_ is not None:
        table[id_] = id_
    table['type'] = 'LineString'
    return table

def multilinestring2table(coordinates, id_):
    table = pd.DataFrame()
    for idx, linestring_array in enumerate(coordinates):
        table_ = linestring2table(linestring_array)
        table = pd.concat([table, table_])
        table['outer_idx'] = idx
    table['id'] = id_
    table['type'] = 'MultiLineString'
    return table

def table2polygon_coords(df):
    polygon_idxs = sorted(list(df.polygon_idx.unique()))
    coordinates = []
    for polygon_idx in polygon_idxs:
        sub_df = df[df.polygon_idx == polygon_idx]
        coordinates_ = sub_df.iloc[:, :2].to_numpy()
        # coordinates_ = sub_df[[0, 1]].to_numpy()
        coordinates_ = [list(x) for x in list(coordinates_)]
        coordinates.append(coordinates_)
    return coordinates

def table2multipolygon_coords(df):
    outer_idxs = sorted(list(df.outer_idx.unique()))
    coordinates = []
    for outer_idx in outer_idxs:
        sub_df = df[df.outer_idx == outer_idx]
        coordinates_ = table2polygon_coords(sub_df)
        coordinates.append(coordinates_)
    return coordinates

def table2point_coords(df):
    coordinates = [df[0].iloc[0], df[1].iloc[1]]
    return coordinates

def table2multipoint_coords(df):
    coordinates = [list(x) for x in list(df.iloc[:, :2].to_numpy())]
    return coordinates

def table2linestring_coords(df):
    coordinates = [list(x) for x in list(df.iloc[:, :2].to_numpy())]
    return coordinates

def table2multilinestring_coords(df):
    coordinates = []
    idxs = sorted(list(df.outer_idx.unique()))
    for idx in idxs:
        sub_df = df[df.outer_idx == idx]
        coordinates_ = table2linestring_coords(sub_df)
        coordinates.append(coordinates_)
    return coordinates

def table2coords(sub_df):
    type_ = sub_df.type.iloc[0]
    if type_ == 'Polygon':
        return table2polygon_coords(sub_df)
    elif type_ == 'MultiPolygon':
        return table2multipolygon_coords(sub_df)
    elif type_ == 'Point':
        return table2point_coords(sub_df)
    elif type_ == 'MultiPoint':
        return table2multipoint_coords(sub_df)
    elif type_ == 'LineString':
        return table2linestring_coords(sub_df)
    elif type_ == 'MultiLineString':
        return table2multilinestring_coords(sub_df)
    return None
    
def convert_table_2_geo_json(geo_template, table):
    for idx, feature in enumerate(geo_template['features']):
        id_ = feature['id']
        sub_df = table[table.id == id_]
        coords = table2coords(sub_df)
        geo_template['features'][idx]['geometry']['coordinates'] = coords
    return geo_template  

def geojson_2_table(geojson_fc):

    geojson_df = pd.DataFrame(geojson_fc['features'])
    table = pd.DataFrame()
    for idx, row in geojson_df.iterrows():
        geometry = row['geometry']
        type_ = geometry['type']
        id_ = row['id']
        coordinates = geometry['coordinates']
        if type_ == 'MultiPolygon':
            table_ = multipolygon2table(coordinates, id_)
        elif type_ == 'Polygon':
            table_ = polygon2table(coordinates, id_)
        elif type_ == 'Point':
            table_ = point2table(coordinates, id_)
        elif type_ == 'MultiPoint':
            table_ = multipoint2table(coordinates, id_)
        elif type_ == 'LineString':
            table_ = linestring2table(coordinates, id_)
        elif type_ == 'MultiLineString':
            table_ = multilinestring2table(coordinates, id_)
        table = pd.concat([table, table_])
    table.rename(columns={0: 'x', 1: 'y'}, inplace=True)
    return table

def read_geojson(path):
    with open(path, 'rb') as f:
        data = geojson.load(f)
    return data
