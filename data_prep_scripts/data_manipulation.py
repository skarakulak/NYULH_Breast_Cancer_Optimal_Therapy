import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict


def read_enum_dict():
    '''
    reads the json files that contains the category->enum mappings
    and returns as a dictionary. If it doesn't exists, the function
    creates one.
    '''
    with open('paths.json', 'r') as f:
        enum_path = json.load(f)['ENUM_DICT']

    if os.path.isfile(enum_path):
        with open(enum_path, 'r') as f: 
            return defaultdict(
                lambda: {},
                {k:defaultdict(lambda: 0,v) for k,v in json.load(f).items()}
            )
    else: 
        return defaultdict(lambda: defaultdict(lambda: 0) )

def write_enum_dict(enum_dict, na_vals=['', np.nan,'---']):
    with open('paths.json', 'r') as f:
        enum_path = json.load(f)['ENUM_DICT']

    with open(enum_path, 'w') as f:
        for v in enum_dict.values():
            for n in na_vals: 
                if n in v: v.pop(np.nan)
        json.dump(dict(enum_dict), f, sort_keys=True, indent=4)


    
def df_enum_categ_vars(df_block, col_names_categ, enum_dict, na_vals=['', np.nan,'---']):    
    for col in col_names_categ:
        max_enum = 0 if len(enum_dict[col]) == 0 else max(enum_dict[col].values())
        col_vals = [k for k in list(df_block[col].unique())
                    if k not in na_vals 
                    and k not in enum_dict[col].keys() 
                    and not pd.isnull(k)
                   ]
        enum_dict[col].update({v:ii for ii,v in enumerate(col_vals,start=max_enum+1)})
        
        df_block[col]=df_block[col].apply(lambda x: enum_dict[col][x])
        
    write_enum_dict(enum_dict, na_vals)
    return df_block[col_names_categ]
        

def processBlock(
    df_block,
    col_names_categ,
    col_names_float,
    enum_dict=None,
    col_names_categ_d=[],
    col_names_float_d=[], 
    col_trim_begin=0, 
    col_trim_end=0,
    lower_case=True,
    na_vals = ['', np.nan,'---'],
    derive_fields=[],
    null_fields=[]
):
    df_block.columns = [k[col_trim_begin:len(k)-col_trim_end] for k in df_block.columns]
    colnames_categ_end = col_names_categ+col_names_categ_d
    colnames_float_end = col_names_float+col_names_float_d
    col_names_drop= set(list(df_block.columns)) - set(colnames_categ_end+colnames_float_end)
    na_vals.append('-'*(col_trim_begin+col_trim_end+1))
    
    df_block[col_names_float]=df_block[col_names_float].apply(pd.to_numeric,errors='coerce')
    
    for field in null_fields:
        df_block[f'{field}_null']  = df_block[field].isna().astype('float')    
    
    if not enum_dict: enum_dict = read_enum_dict()
    
    if lower_case:
        df_block[col_names_categ]=df_block[col_names_categ].apply(
            lambda x: x.str.lower() if(x.dtype == 'object') else na_vals
        )
    
    # derives given field if given any 
    if derive_fields:
        for ii,f in enumerate(derive_fields):
            df_block[f[1]] = df_block.apply(f[0], axis=1)
                
    df_block[colnames_categ_end] = df_enum_categ_vars(df_block,colnames_categ_end, enum_dict, na_vals)
    df_block.drop(col_names_drop,axis=1,inplace=True)
    return df_block
    
    

def divideRepetitiveBlocks(
    df,div_n,categ_cols,float_cols,derive_fields=[],null_fields=[],col_trim_begin=0, col_trim_end=0,lower_case=True
):
    block_n = len(list(df.columns)) // 7
    cols_to_drop = set(range(block_n)) - set(categ_cols+float_cols)
    col_names = [k[col_trim_begin:len(k)-col_trim_end] for k in df.columns[:block_n]]
    col_names_categ = [col_names[k] for k in categ_cols]
    col_names_float = [col_names[k] for k in float_cols]
    if derive_fields:
        col_names_categ_d = [k[1] for k in derive_fields if k[2]=='categ']
        col_names_float_d = [k[1] for k in derive_fields if k[2]=='float']    


    enum_dict = read_enum_dict()

    result = [
        processBlock(
            df.iloc[:,i*block_n:(i+1)*block_n],
            col_names_categ,
            col_names_float,
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
