import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

def get_path(x):
    ''' returns the path of the given file/folder from path.json file'''
    with open('paths.json', 'r') as f:
        enum_path = json.load(f)[x]
    return enum_path

def read_enum_dict():
    '''
    reads the json files that contains the category->enum mappings
    and returns as a dictionary. If the dictionary doesn't exists,
    the function creates one.
    '''
    enum_path = os.path.join(get_path('FOLDER_COL_VAL_DICTS'), get_path('ENUM_DICT'))

    if os.path.isfile(enum_path):
        with open(enum_path, 'r') as f: 
            return defaultdict(
                lambda: defaultdict(lambda: 0),
                {k:defaultdict(lambda: 0,v) for k,v in json.load(f).items()}
            )
    else: 
        return defaultdict(lambda: defaultdict(lambda: 0) )

def write_enum_dict(enum_dict, na_vals=['', np.nan,'---','NaN']):
    '''
    args:
      enum_dict: dictionary to save in the json format
      na_vals:   values to be treated as NaN and consequently 
                 would have zero values in the dictionary
    '''
    enum_path = os.path.join(get_path('FOLDER_COL_VAL_DICTS'), get_path('ENUM_DICT'))

    for k,v in enum_dict.items():
        for n in na_vals: 
            if n in v: v.pop(n)
        del_keys = []
        for k_ in v.keys():
            if pd.isnull(k_): del_keys.append(k_)
        for dk in del_keys: del enum_dict[k][dk]
        if len(k)==0: enum_dict.pop(k)

    with open(enum_path, 'w') as f:
        json.dump(dict(enum_dict), f, sort_keys=True, indent=4)

def cond_add_to_enumdict(val,col,enum_dict,na_vals):
    '''returns a bool indicating whether a new key should be added  
    to the enum_dict for 'val' '''
    return (val not in na_vals) and (val not in enum_dict[col].keys()) and (not pd.isnull(val))

def add_to_enumdict_with_threshold(
    pd_series, enum_dict, col, na_vals, group_min_count, max_enum=-1, save_oth_vals=True):
    ''' Adds categorical values to the enum_dict, that occured more 
    times than the given threshold. Values that do not pass the given 
    threshold are mapped to the value 1. '''
    if max_enum == -1:
        max_enum = 0 if len(enum_dict[col]) == 0 else max(enum_dict[col].values())
    val_counts = pd_series.value_counts() > group_min_count
    vals_to_add = [
        val for val in val_counts.index.tolist() if cond_add_to_enumdict(val,col,enum_dict,na_vals)]
    # if all values occur more times than the given threshold value,
    # than we start indexing with 0, otherwise with 1.    
    if (val_counts[vals_to_add]).all():
        enum_dict[col].update({v:ii for ii,v in enumerate(vals_to_add,start=max_enum+1)})
    else:
        current_enum = max(max_enum,1)
        for ind, val in (val_counts[vals_to_add]).items():
            if val: 
                current_enum += 1
                enum_dict[col][ind] = current_enum
            elif save_oth_vals:
                enum_dict[col][ind] = 1
    return enum_dict

def df_enum_categ_vars(
    df_block, 
    col_names_categ,
    enum_dict, 
    na_vals=['', np.nan,'---','NaN'],
    group_values=False,
    group_min_count=50,
    save_oth_vals = False
    ): 
    '''
    Converts categorical values into enum valus and using the `enum_dict`.
    If a given value is not in the `enum_dict`, creates a enum value and saves
    over the existing `enum_dict`
    '''   
    for col in col_names_categ:        
        max_enum = 0 if len(enum_dict[col]) == 0 else max(enum_dict[col].values())

        if not group_values:
            col_vals = [k for k in list(df_block[col].unique()) if cond_add_to_enumdict(k,col,enum_dict,na_vals)]
            enum_dict[col].update({v:ii for ii,v in enumerate(col_vals,start=max_enum+1)})
        else:
            enum_dict = add_to_enumdict_with_threshold(
                df_block[col], enum_dict, col, na_vals, group_min_count, max_enum, save_oth_vals=save_oth_vals)


        df_block[col]=df_block[col].apply(lambda x: enum_dict[col][x])
    
    write_enum_dict(enum_dict, na_vals)

    return df_block[col_names_categ]


def enums_group(
    df,
    colname,
    enum_dict,
    group_min_count=50,
    na_vals=['', np.nan,'---','NaN'],
    save_oth_vals = True
    ):
    '''
    prepares enums_dict for a block of repitetive columns in the raw data.
    NaN values are mapped to '0', and the values whose counts are less then
    'group_min_count' are assigned to 1.
    '''
    enum_dict = add_to_enumdict_with_threshold(
        df.unstack(), enum_dict, colname, na_vals, group_min_count,save_oth_vals=save_oth_vals)
    return enum_dict


def process_block(
    df_block,
    col_names,
    col_names_categ,
    col_names_float,
    colnames_categ_final,
    colnames_float_final,
    col_names_drop,
    enum_dict=None,
    col_names_categ_d=[],
    col_names_float_d=[], 
    col_trim_begin=0, 
    col_trim_end=0,
    lower_case=True,
    na_vals = ['', np.nan,'---','NaN'],
    derive_fields=[],
    null_fields=[]
):
    if df_block.isnull().all().all(): return 
    df_block = df_block.copy()

    df_block.columns = col_names
    
    df_block[col_names_float]=df_block[col_names_float].apply(pd.to_numeric,errors='coerce',result_type='broadcast')
    
    for field in null_fields:
        df_block[f'{field}_null']  = df_block[field].isna().astype('float') 
    
    if not enum_dict: enum_dict = read_enum_dict()
    
    if lower_case:
        df_block[col_names_categ]=df_block[col_names_categ].apply(
            lambda x: x.str.lower() if(x.dtype == 'object') else np.nan,
            result_type='broadcast'
        )
    
    # derives given fields, if given any 
    if derive_fields:
        for ii,f in enumerate(derive_fields):
            df_block[f[1]] = df_block.apply(f[0], axis=1)
                
    df_block[colnames_categ_final] = df_enum_categ_vars(df_block,colnames_categ_final, enum_dict, na_vals)
    df_block.drop(col_names_drop,axis=1,inplace=True)
    return df_block
    
    

def divide_repetitive_blocks(
    df,div_n,
    categ_cols,
    float_cols,
    derive_fields=[],
    null_fields=[],
    col_trim_begin=0,
    col_trim_end=0,
    lower_case=True,
    group_values = False,
    group_min_count = 50
):
    '''
    takes the repitetive blocks in the raw data, process them and 
    returns list of dataframes, each of which represent a processed
    data of a block.
    '''
    block_n = len(list(df.columns)) // div_n
    colnames_org = df.columns[:block_n]

    assert type(col_trim_begin) == type(col_trim_end), "types of the following arguments should be the same: 'col_trim_begin', 'col_trim_end'"
    if isinstance(col_trim_begin,list) and isinstance(col_trim_end,list):
        col_names = []
        for c_ind, (tb, te) in enumerate(zip(col_trim_begin,col_trim_end)):
            org_cn = colnames_org[c_ind]
            col_names.append(org_cn[tb:len(org_cn)-te] )
    else:
        col_names = [k[col_trim_begin:len(k)-col_trim_end] for k in colnames_org]

    col_names_categ = [col_names[k] for k in categ_cols]
    col_names_float = [col_names[k] for k in float_cols]


    null_fields_add = [
        k for k in col_names_float if k not in null_fields
    ]
    if null_fields_add: null_fields += null_fields_add

    if derive_fields:
        col_names_categ_d = [k[1] for k in derive_fields if k[2]=='categ']
        col_names_float_d = [k[1] for k in derive_fields if k[2]=='float']    
    else: 
        col_names_categ_d, col_names_float_d = [],[]

    colnames_categ_final = col_names_categ+col_names_categ_d
    colnames_float_final = col_names_float+col_names_float_d
    col_names_drop= set(list(col_names)) - set(colnames_categ_final+colnames_float_final)

    enum_dict = read_enum_dict()

    # temp implementation for country of origin features
    if group_values: 
        for col in col_names_categ:
            cols_to_group = []
            for dfcol in df.columns:
                assert isinstance(col_trim_begin,int) and isinstance(col_trim_end,int)
                if dfcol[col_trim_begin:len(dfcol)-col_trim_end] == col:
                    cols_to_group.append(dfcol)

            enum_dict = enums_group(df[cols_to_group],col,enum_dict,group_min_count)    

    result = [
        process_block(
            df.iloc[:,i*block_n:(i+1)*block_n],
            col_names,
            col_names_categ,
            col_names_float,
            colnames_categ_final,
            colnames_float_final,
            col_names_drop,
            enum_dict=enum_dict,
            col_names_categ_d=col_names_categ_d,
            col_names_float_d=col_names_float_d, 
            col_trim_begin=col_trim_begin, 
            col_trim_end=col_trim_end,
            lower_case=True,
            derive_fields=derive_fields,
            null_fields=null_fields
        )
        for i in range(div_n)
    ]
    return result


def process_single_cols(
    df,
    colname,
    categ=True,
    lower_case=True,
    group_values = False,
    group_min_count = 50
):
    '''
    takes a block of repitetive column in the raw data, process them 
    and returns list of dataframes, each of which represent a processed
    column of in the given dataframe.
    '''
    block_n = len(list(df.columns)) // df.shape[1]
    colnames_org = [colname]

    enum_dict = read_enum_dict()
    if group_values: 
        enum_dict = enums_group(df,colname,enum_dict,group_min_count)

    if categ:
        result = [
            process_block(
                df.iloc[:,[i]],
                colnames_org,
                colnames_org,
                [],
                colnames_org,
                [],
                [],
                enum_dict=enum_dict
            )
            for i in range( df.shape[1])
        ]
    else:
        result = [
            process_block(
                df.iloc[:,[i]],
                colnames_org,
                [],
                colnames_org,
                [],
                colnames_org,
                [],
                enum_dict=enum_dict
            )
            for i in range( df.shape[1])
        ]
    return result


def process_remaining_categ_cols(
    df,
    group_values = True,
    group_min_count = 50
):
    '''
    takes a block of repitetive column in the raw data, process them 
    and returns list of dataframes, each of which represent a processed
    column of in the given dataframe.
    '''
    enum_dict = read_enum_dict()
    result = df_enum_categ_vars(
        df.copy(), 
        list(df.columns),
        enum_dict, 
        na_vals=['', np.nan,'---','NaN'],
        group_values=True,
        group_min_count=50,
        save_oth_vals = True
    )
    return result




def read_replaceColVals_dict(str_vals=True, cont_vals=False):
    '''
    reads the json files that contains the value->new_value mappings
    and returns as a dictionary. If the dictionary doesn't exists,
    the function creates one.
    '''
    assert (str_vals ^ cont_vals), "exactly one of the following arguments must be 'True': str_vals, cont_vals"
    
    colvals_path = os.path.join(
        get_path('FOLDER_COL_VAL_DICTS'), 
        get_path('COL_VALS_REPLACE_STR' if str_vals else 'COL_VALS_REPLACE_FLOAT_OUTLIERS')
        )

    if os.path.isfile(colvals_path):
        with open(colvals_path, 'r') as f: 
            return defaultdict(
                lambda: {},
                {k:defaultdict(lambda: np.nan,v) for k,v in json.load(f).items()}
            )
    else: 
        return defaultdict(lambda: defaultdict(lambda: np.nan) )


def write_replaceColVals_dict(colvals_dict, str_vals=True, cont_vals=False):
    '''
    args:
      colvals_dict: dictionary to save in the json format
    '''
    assert (str_vals ^ cont_vals), "exactly one of the following arguments must be 'True': str_vals, cont_vals"

    colvals_path = os.path.join(
        get_path('FOLDER_COL_VAL_DICTS'), 
        get_path('COL_VALS_REPLACE_STR' if str_vals else 'COL_VALS_REPLACE_FLOAT_OUTLIERS')
        )

    with open(colvals_path, 'w') as f:
        json.dump(dict(colvals_dict), f, sort_keys=True, indent=4)
        

def df_replaceColVals_vars(df, col_indices=None, str_vals=True, cont_vals=False): 
    '''
    Converts categorical values into enum valus and using the `enum_dict`.
    If a given value is not in the `enum_dict`, creates a enum value and saves
    over the existing `enum_dict`
    '''   
    assert (str_vals ^ cont_vals), "exactly one of the following arguments must be 'True': str_vals, cont_vals"
    
    if str_vals:
        colvals_dict = read_replaceColVals_dict()

        for c in col_indices:
            colname = df.columns[c]
            vals = df.iloc[:,c].dropna().unique()
            temp_dict = colvals_dict[colname]
            temp_dict.update({
                k:k for k in vals[vals == vals.astype(str)] 
                if k not in temp_dict
                })

        
        write_replaceColVals_dict(colvals_dict)

        for col in colvals_dict:
            for k,v in colvals_dict[col].items():
                if v == '<np.nan>': colvals_dict[col][k]=np.nan
        return colvals_dict

    else:
        colvals_dict = read_replaceColVals_dict(str_vals=False, cont_vals=True)

        for ii, (a,b) in enumerate(df.dtypes.iteritems()):
            if b == np.float64 : 
                vals = df[a].dropna().values 
                outliers = vals[np.abs(vals - np.mean(vals)) > 4 * np.std(vals)]
                if len(outliers) > 0: 
                    colvals_dict[a].update({
                        str(k):k for k in outliers if str(k) not in colvals_dict[a]
                        })

        
        write_replaceColVals_dict(colvals_dict,str_vals=False, cont_vals=True)
        cols_to_del = []
        for col in colvals_dict:
            vals_to_del = []
            for k,v in colvals_dict[col].items():
                if v == float(k): vals_to_del.append(k)
            for k in vals_to_del: del colvals_dict[col][k]
            if len(colvals_dict[col])==0: cols_to_del.append(col)
        for k in cols_to_del: del colvals_dict[k]
        
        return colvals_dict


