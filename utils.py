import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

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
    m_y = m_y.detach().numpy() if isinstance(m_y,torch.Tensor) else: m_y
    ax.scatter(coordinates[:,0], coordinates[:,1],alpha = 0.20, c = ['c' if k==0 else 'r' for k in ])
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


def prep_report(medData_float,df_corr,train_ind, dummy_cols,target_cols, quantile_1 =.125, quantile_2 = .875 ):
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
            np_q_perc[f_idx] = (q_bool & (df_corr[clus]==1).values).sum() / q_bool.sum()
        report_df[clus+'_(%)Percentile'] = np_q_perc
    report_df['categ_long'] = pd.Series([dummy_cols_map[k] if k in dummy_cols_map else '-' for k in report_df.index ],index=report_df.index)
    return report_df