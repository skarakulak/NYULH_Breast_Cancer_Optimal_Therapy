import pandas as pd
import numpy as np

def derive_relatives_categ(row,relative_col = 'Relative:',if_other_col='If other, specify'):
    row.fillna('--',inplace=True)
    row = row.apply(lambda x: str(x) if type(x) != str else x)
    if row[relative_col] in ['sister','brother', 'half-sister','daughter']:
        categ = 'F'
    elif ('mat' in row[relative_col] or row[relative_col] == 'mother' or
    'mat' in row[if_other_col]):
        categ = 'M'
    elif ('pat' in row[relative_col] or row[relative_col] == 'father' or
    'pat' in row[if_other_col]):
            categ = 'P'
    else:
        categ = 'O'
    return categ
   