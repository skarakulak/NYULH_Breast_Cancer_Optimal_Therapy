import pandas as pd
import numpy as np

def derive_relatives_categ(row):
    row.fillna('--',inplace=True)
    row = row.apply(lambda x: str(x) if type(x) != str else x)
    if row['Relative:'] in ['sister','brother', 'half-sister','daughter']:
        categ = 'F'
    elif ('mat' in row['Relative:'] or row['Relative:'] == 'mother' or
    'mat' in row['If other, specify']):
        categ = 'M'
    elif ('pat' in row['Relative:'] or row['Relative:'] == 'father' or
    'pat' in row['If other, specify']):
            categ = 'P'
    else:
        categ = 'O'
    return categ
   