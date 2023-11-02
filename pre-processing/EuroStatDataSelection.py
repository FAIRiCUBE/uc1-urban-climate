import sqlite3
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy
import collections

if __name__ == '__main__':
    city_cube = 'C_urban_cube_sh.sqlite'
    # get Eurostat number of population data
    con = sqlite3.connect(city_cube)
    # read full table
    data = pd.read_sql_query("SELECT * FROM c_urban_cube_eurostat", con)
    target = "EN1003V"
    features = ["EN1002V", "EN1004V"]

    years = range(1991, 2022)

    subdata = data[data["indic_code"].isin([target] + features)]

    d = {}

    cities = []
    indic = {}
    for i in features + [target]:
        indic[i] = []
    y = []
    for k in years:
        inter = subdata[(~subdata[str(k)].isna())][['indic_code', 'urau_code', str(k)]]
        if len(inter) > 0 & (len(inter.indic_code.unique()) == len(features) + 1):
            lcities = inter.urau_code.unique()
            for i in inter.indic_code.unique():
                result = collections.Counter(lcities) & collections.Counter(inter[(inter['indic_code'] == i)].urau_code.unique())
                lcities = list(result.elements())
            if len(lcities)>0:
                cities = cities + lcities
                y = y + ([str(k)]*len(lcities))
                for j in lcities:
                    for i in [target] + features:
                        if(len(list(inter[(inter['urau_code']==j)& (inter['indic_code']==i)][str(k)]))>0):
                            indic[i].append(list(inter[(inter['urau_code']==j)& (inter['indic_code']==i)][str(k)])[0])
                        else:
                            indic[i].append(None)
    d['City'] = cities
    d['Year'] = y
    for i in [target] + features:
        d[i] = indic[i]

    df = pd.DataFrame(data=d)
    for i in [target] + features:
        df = df[~df[i].isna()]
    print('Train/Test data size: ', len(df))
    df.to_csv('SelectedData_' + target+ '.csv', index=False)

    d2 = {}
    d2['City'] = []
    d2['Indicator'] = []
    d2['Year'] = []
    missing = pd.DataFrame(data=d2)
    subdata = data[data["indic_code"].isin([target] + features)]
    for y in years:
        for j in subdata.urau_code.unique():
            mask = True
            inter = subdata[(subdata["urau_code"] == j)]
            for i in features:
                if(len(inter[inter['indic_code'] == i][str(y)])>0):
                    if (list(inter[inter['indic_code'] == i][str(y)].isna())[0]):

                        mask = False
                        break
                else:

                    mask = False
                    break
            if(len(inter[inter['indic_code'] == target][str(y)])>0):
                if (not list(inter[inter['indic_code'] == target][str(y)].isna())[0]):
                    mask = False
            if mask:
                missing = missing.append(dict(zip(missing.columns, [j, y, target])), ignore_index=True)

    print('Missing data size: ', len(missing))
    missing.to_csv('Missing_' + target+ '.csv', index=False)

