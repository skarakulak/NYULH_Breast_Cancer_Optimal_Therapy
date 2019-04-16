import pandas as pd
import numpy as np
import os
import json

from data_prep_scripts.data_manipulation import divide_repetitive_blocks

def prep_country_data(file_name='country_features.tab', overwrite=False):
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    
    res_folder = paths["GEN_DATA"]
    gen_data_path = os.path.join(os.getcwd(), res_folder, file_name)
    
    if not os.path.isfile(gen_data_path) or overwrite:
        data_folder = paths["COUNTRY_DATA"]
        countries_path = os.path.join(os.getcwd(),data_folder)

        # https://www.kaggle.com/fernandol/countries-of-the-world/version/1#
        c1Cols=['Country','Population','Area (sq. mi.)','Pop. Density (per sq. mi.)','Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)','GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Birthrate','Deathrate']
        countryData1 = pd.read_csv(os.path.join(countries_path,'countries of the world.csv'), sep=',',decimal=',',usecols=c1Cols)
        # http://gsociology.icaap.org/dataupload.html
        c2Cols=['AltCountryName1','DevelopmentLevel','pop1960','pop2010','imr19551960','imr20052010','AvgYrsSchool70','AvgYrsSchool10','GDP1970','GDP2010','GDPPerCap1970','GDPPerCap2010','WHR score','HDI1980','HDI2013']
        countryData2 = pd.read_excel(os.path.join(countries_path,'QualityOfLifeCountryData_all.xlsx'),nrows=232)[c2Cols].replace('..',np.NaN)
        # https://worldmap.harvard.edu/data/geonode:country_centroids_az8
        c3Cols=['admin','the_geom','pop_est','gdp_md_est','economy','income_grp','continent','subregion','Longitude','Latitude']
        countryData3 = pd.read_excel(os.path.join(countries_path,'country_centroids_az8.xls'))[c3Cols]

        # load mapping of the merge keys
        country_mergekeys = pd.read_excel(os.path.join(countries_path,'country_mergekeys.xlsx'))
        country_mergekeys=country_mergekeys[country_mergekeys.admin_centroids.isna() == False]

        country_data = country_mergekeys.merge(countryData3,left_on = 'admin_centroids', right_on='admin', how='left')
        country_data = country_data.merge(countryData1,left_on = 'Country_cot_world', right_on='Country', how='left')
        country_data = country_data.merge(countryData2,left_on = 'AltCountryName1_quality', right_on='AltCountryName1', how='left')

        # separate longtitude and latitude from the 'the_geom' column
        temp = country_data.the_geom.str.slice(start=7,stop=-1).str.split(' ',expand=True)
        country_data['country_longitude'] = temp.iloc[:,0]
        country_data['country_latitude'] = temp.iloc[:,1]

        country_data.drop(['the_geom','admin_centroids', 'Country_cot_world','AltCountryName1_quality', 'admin','Country','AltCountryName1'], axis=1, inplace=True)

        country_data.to_csv(gen_data_path, sep='\t',index=False)
        return country_data
    else: 
        return pd.read_csv(gen_data_path, sep='\t')

def get_country_of_origin_data(df):
    country_features = prep_country_data().applymap(lambda s:s.lower() if type(s) == str else s).drop_duplicates()
    country_features = country_features[['MEDDATA', 'income_grp',
           'continent', 'subregion','Pop. Density (per sq. mi.)',
           'Literacy (%)', 'Phones (per 1000)', 'Birthrate', 'Deathrate',
           'DevelopmentLevel', 'imr20052010',
           'AvgYrsSchool10', 'GDPPerCap2010', 'WHR score','HDI2013',
           'country_longitude', 'country_latitude']]
    country_features.columns = ['country_of_origin']+list(country_features.columns)[1:]


    df_f = df[['Country of Origin for Father']].merge(
        country_features,left_on='Country of Origin for Father', 
        right_on='country_of_origin',
        how='left'
    ).drop('Country of Origin for Father', axis=1)
    df_f = df_f.add_prefix('f.')

    df_m = df[['Country of Origin for Mother']].merge(
        country_features,left_on='Country of Origin for Mother', 
        right_on='country_of_origin',
        how='left'
    ).drop('Country of Origin for Mother', axis=1)
    df_m = df_m.add_prefix('m.')

    df_p = df[['Country of Origin for Patient']].merge(
        country_features,left_on='Country of Origin for Patient', 
        right_on='country_of_origin',
        how='left'
    ).drop('Country of Origin for Patient', axis=1)
    df_p = df_p.add_prefix('p.')


    df_concat = pd.concat([df_p,df_m,df_f],ignore_index=True,axis=1)
    df_concat.columns = list(df_p.columns)+list(df_m.columns)+list(df_f.columns)

    result = divide_repetitive_blocks(
        df_concat,
        3,
        categ_cols = [0,1,2,3,9],
        float_cols = [4,5,6,7,8,10,11,12,13,14,15,16],
        col_trim_begin=2,
        col_trim_end=0,
        lower_case=True,
        group_values=True,
        group_min_count = 150,
        null_fields = ['country_of_origin']
    )

    return result 