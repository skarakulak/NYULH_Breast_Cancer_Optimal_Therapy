import pandas as pd
import numpy as np
import os

from data_prep_scripts.data_manipulation import divide_repetitive_blocks,process_single_cols
from data_prep_scripts.field_derivations import derive_relatives_categ



def get_repetitive_cols(medData):
    result = []
    result.append(
        divide_repetitive_blocks(
            medData.iloc[:,47:89],
            7,
            categ_cols = [0,2,4,5],
            float_cols = [3],
            derive_fields=[(derive_relatives_categ,'relative_categ','categ')],
            null_fields=['Relative:','Age at diagnosis'],
            col_trim_begin=3,
            col_trim_end=0,
            lower_case=True
        )
    )
    result.append(
        divide_repetitive_blocks(
            medData.iloc[:,89:117],
            7,
            categ_cols = [0,2],
            float_cols = [3],
            derive_fields=[(
                (lambda x: derive_relatives_categ(x,'Relative:','If other, please specify'))
                ,'relative_categ','categ')],
            null_fields=['Relative:','Age at diagnosis'],
            col_trim_begin=[3]*4,
            col_trim_end=[2,0,2,2],
            lower_case=True
        )
    )
    result.append(
        divide_repetitive_blocks(
            medData.iloc[:,117:152],
            7,
            categ_cols = [0,2],
            float_cols = [4],
            derive_fields=[(
                (lambda x: derive_relatives_categ(x,'Relative:','If other, specify:'))
                ,'relative_categ','categ')],
            null_fields=['Relative:','Age at diagnosis'],
            col_trim_begin=[3]*5,
            col_trim_end=[2,0,0,0,2],
            lower_case=True
        )
    )
    result.append(
        divide_repetitive_blocks(
            medData.iloc[:,163:191],
            7,
            categ_cols = [0,3],
            float_cols = [2],
            null_fields=['Type:','In-Vitro Fertilization (IVF) cycles:','Infertility treatment duration: '],
            col_trim_begin=[3]*4,
            col_trim_end=[0,2,0,0],
            lower_case=True
        )
    )
    result.append(
        divide_repetitive_blocks(
            medData.iloc[:,214:249],
            5,
            categ_cols = [3,4,5],
            float_cols = [],
            null_fields=[],
            col_trim_begin=3,
            col_trim_end=0,
            lower_case=True
        )
    )
    result.append(
        divide_repetitive_blocks(
            medData.iloc[:,294:348],
            6,
            categ_cols = [1,3,4,5,6,7],
            float_cols = [0],
            null_fields=[],
            col_trim_begin=[3]*9,
            col_trim_end=[2,0,0,0,0,0,0,0,2],
            lower_case=True
        )
    )
    result.append(
        process_single_cols(
            medData.iloc[:,427:435],
            'sport_activity',
            categ=True
        )
    )

    result.append(
        process_single_cols(
            medData.iloc[:,357:364],
            'drug_allergy',
            categ=True,
            group_values=True,
            group_min_count = 50
        )
    )
    return result
