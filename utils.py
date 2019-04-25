import torch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score


colors = np.array([x for x in 'bgrcmybgrcmybgrcmybgrcmybgrcmy'])
colors = np.hstack([colors] * 100 + ['k'])

def plot_clusters(coordinates, labels, m_y):
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121)
    for i in set(labels):
        coor = np.mean(coordinates[labels==i,:],axis=0)
        ax.text(coor[0],coor[1], str(i), ha='center', va='center')

    ax.scatter(coordinates[:,0], coordinates[:,1],alpha = 0.10, c =colors[labels].tolist())
    
    ax = fig.add_subplot(122)
    m_y = m_y.detach().numpy() if isinstance(m_y,torch.Tensor) else m_y
    ax.scatter(coordinates[:,0], coordinates[:,1],alpha = 0.20, c = ['c' if k==0 else 'r' for k in m_y ])
    plt.show()

def div_cluster(coor,cls_labels, clus_id_div = 1, num_of_clus = 2, n_neighbors = 15, id_add = 100):
    knn_graph = kneighbors_graph(coor[cls_labels==clus_id_div], n_neighbors, include_self=False)

    clusmodel = AgglomerativeClustering(linkage='ward',
                                    connectivity=knn_graph,
                                    n_clusters=num_of_clus)
    clusmodel.fit(coor[cls_labels==clus_id_div])
    cls_labels_new = clusmodel.labels_

    labels_temp = cls_labels.copy()
    labels_temp[cls_labels==clus_id_div] = cls_labels_new + id_add
    dict_new_clus_ids = {b:(-1 if b == -1 else a) for a,b in enumerate(set(labels_temp))}
    labels_temp = np.array([dict_new_clus_ids[k] for k in labels_temp])
    return labels_temp


def prep_report(df_MI_X, medData_float,df_corr,train_ind, dummy_cols,target_cols, discrete_flag,dummy_cols_map, quantile_1 =.125, quantile_2 = .875 ):
    report_df = pd.DataFrame(index = df_MI_X.columns)
    t_float_vars = medData_float.iloc[train_ind]
    for ii,clus in enumerate(target_cols):
        report_df[clus+'_MI'] = mutual_info_regression(df_MI_X.astype('float64').values,df_corr[clus].astype('float64'), discrete_features=discrete_flag)
        report_df[clus+'_COR'] = df_MI_X.astype('float64').corrwith(df_corr[clus])

        # categ vars 
        c_stats = np.zeros((len(dummy_cols),2))
        for id1, lb1 in enumerate(dummy_cols): 
            c_stats[id1,0] = ((df_corr[lb1] == 1)&(df_corr[clus] == 1)).sum() / (df_corr[lb1] == 1).sum()
            c_stats[id1,1] = ((df_corr[lb1] == 1)&(df_corr[clus] == 1)).sum() / (df_corr[clus] == 1).sum()
        c_stats = pd.DataFrame(c_stats,index =dummy_cols,  columns=[clus + s for s in ['_(%)Label','_(%)Cluster']])
        report_df = pd.concat([report_df,c_stats],axis=1,sort=False)

        # float vars
        cls_float_vals = t_float_vars[(df_corr[clus]==1).values]
        quartile_cols = [f'{clus}_Q{100*quantile_1:.3}%', f'{clus}_Q{100*quantile_2:.3}%']
        report_df[quartile_cols]  = cls_float_vals.quantile([quantile_1,quantile_2]).T
        report_df[clus+'_MEAN'] = cls_float_vals.mean()
        report_df[clus+'_NON_NAN'] = cls_float_vals.count() / (df_corr[clus]==1).values.sum()
        np_q_perc = np.zeros(report_df.shape[0])
        for f_idx, f_col in enumerate(t_float_vars):
            q_bool = (
                (t_float_vars[f_col]>=report_df.loc[f_col,quartile_cols[0]]) &(t_float_vars[f_col]<=report_df.loc[f_col,quartile_cols[1]])
            ).values
            qbs = q_bool.sum()
            np_q_perc[f_idx] = ((q_bool & (df_corr[clus]==1).values).sum() / qbs ) if qbs>0 else -99
        report_df[clus+'_(%)Percentile'] = np_q_perc
    report_df['categ_long'] = pd.Series([dummy_cols_map[k] if k in dummy_cols_map else '-' for k in report_df.index ],index=report_df.index)
    return report_df

def rep_prep_report(df_corr,rep_cols,target_cols,train_ind, categs_json, quantile_1 =.25, quantile_2 = .75):
    exist_cond = lambda x: isinstance(x,pd.DataFrame) and x.shape[1]>0 
    rep_report_df = None
    rep_dummy_cols_map = {}
    clus_dummies = df_corr[target_cols]

    for g_ind, group in enumerate(rep_cols):
        rep_dummy_cols=[]
        cols_to_concat_f = [k[1].iloc[train_ind] for k in group if k and exist_cond(k[1]) ]
        cols_to_concat_c = [k[0].iloc[train_ind] for k in group if k and exist_cond(k[0]) ]
        
        if cols_to_concat_c: 
            clusters_tiled_c = pd.concat([clus_dummies]*len(cols_to_concat_c), ignore_index=True,sort=False)
            cols_concat_c = pd.concat(cols_to_concat_c, axis=0, ignore_index=True,sort=False)
        if cols_to_concat_f: 
            clusters_tiled_f = pd.concat([clus_dummies]*len(cols_to_concat_f), ignore_index=True,sort=False)
            rep_t_float_vars = pd.concat(cols_to_concat_f, axis=0, ignore_index=True,sort=False)
            rep_df_MI_X = rep_t_float_vars.copy()
        else:
            rep_df_MI_X = pd.DataFrame(index=cols_concat_c.index)

        col_sfx = str(g_ind)+'.'
        for k in list(cols_concat_c.columns):
            cols_concat_c[k] = cols_concat_c[k].astype('category')
            for val in cols_concat_c[k].cat.categories:
                rep_df_MI_X[col_sfx+ k+'|'+str(val)] = (cols_concat_c[k] == val).astype('int32').astype('category')
                rep_dummy_cols.append(col_sfx+k+'|'+str(val))
                rep_dummy_cols_map[col_sfx+k+'|'+str(val)] = k+'|'+str(val) + '|'+', '.join([a for a,b in categs_json[k].items() if b ==int(val)]) 

    #     rep_report_df = pd.DataFrame(index = rep_df_MI_X.columns)
        discrete_flag = (
            ([True]*rep_t_float_vars.shape[1] if cols_to_concat_f else []) +
            ([False]*len(rep_dummy_cols) if cols_to_concat_c else [])
        )

        temp_df = pd.DataFrame(index= rep_df_MI_X.columns)
        for ii,clus in enumerate(target_cols):

            if clus == 18 and g_ind == 4: 
                ipdb.set_trace()
            temp_df[clus+'_MI'] = mutual_info_regression(
                rep_df_MI_X.astype('float64').fillna(0).values,
                clusters_tiled_c[clus].astype('float64'), 
                discrete_features=discrete_flag
            )
            temp_df[clus+'_COR'] = rep_df_MI_X.astype('float64').fillna(0).corrwith(clusters_tiled_c[clus])

            # categ vars 
            if cols_to_concat_c:
                c_stats = np.zeros((len(rep_dummy_cols),2))
                for id1, lb1 in enumerate(rep_dummy_cols): 
                    total_lb1 = (rep_df_MI_X[lb1] == 1).sum()
                    total_cl = (clusters_tiled_c[clus] == 1).sum()
                    c_stats[id1,0] = (
                        ((rep_df_MI_X[lb1] == 1)&(clusters_tiled_c[clus] == 1)).sum() / total_lb1
                    ) if total_lb1>0 else -99
                    c_stats[id1,1] = (
                        ((rep_df_MI_X[lb1] == 1)&(clusters_tiled_c[clus] == 1)).sum() / total_cl
                    ) if total_cl>0 else -99
                c_stats = pd.DataFrame(c_stats,index =rep_dummy_cols,  columns=[clus + s for s in ['_(%)Label','_(%)Cluster']])
                temp_df = pd.concat([temp_df,c_stats],axis=1,sort=False)

            # float vars
            if cols_to_concat_f:
                cls_float_vals = rep_t_float_vars[(clusters_tiled_f[clus]==1).values]
                quartile_cols = [f'{clus}_Q{100*quantile_1:.3}%', f'{clus}_Q{100*quantile_2:.3}%']
                temp_df[quartile_cols]  = cls_float_vals.quantile([quantile_1,quantile_2]).T
                temp_df[clus+'_MEAN'] = cls_float_vals.mean()
                temp_df[clus+'_NON_NAN'] = cls_float_vals.count() / (clusters_tiled_f[clus]==1).values.sum()
                np_q_perc = np.zeros(temp_df.shape[0])

                for f_idx, f_col in enumerate(rep_t_float_vars):
                    q_bool = (
                        (rep_t_float_vars[f_col]>=temp_df.loc[f_col,quartile_cols[0]]) &(rep_t_float_vars[f_col]<=temp_df.loc[f_col,quartile_cols[1]])
                    ).values
                    qbs = q_bool.sum()
                    np_q_perc[f_idx] = ((q_bool & (clusters_tiled_f[clus]==1).values).sum() / qbs) if qbs>0 else -99
                temp_df[clus+'_(%)Percentile'] = np_q_perc

        temp_df['categ_long'] = pd.Series([rep_dummy_cols_map[k] if k in rep_dummy_cols_map else '-' for k in temp_df.index ],index=temp_df.index)    
        if not isinstance(rep_report_df, pd.DataFrame): rep_report_df=temp_df
        else: rep_report_df=rep_report_df.append(temp_df,sort=False)
    return rep_report_df