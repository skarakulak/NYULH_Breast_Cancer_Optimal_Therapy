import torch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
import re


colors = np.array([x for x in 'bgrcmybgrcmybgrcmybgrcmybgrcmy'])
colors = np.hstack([colors] * 100 + ['k'])


def extract_num(s):
    s=str(s)
    t = re.search(r'\d+', s)
    return -1 if t is None else int(t.group())

def transform_vals(title,df_corr=None,categs_json=None):
    t_dict = {v:k for k,v in categs_json[title].items()}
    return np.array([t_dict[k] if k in t_dict else 'na' for k in df_corr[title].astype('int')])

def plot_clusters(coordinates, labels,alpha=.20):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_title('Clusters');  ax.set_xticklabels([]); ax.set_yticklabels([])
    for i in set(labels) - set([-1]):
        lab_idx = (labels==i)
        coor = np.mean(coordinates[lab_idx,:],axis=0)
        ax.text(coor[0],coor[1], str(i), ha='center', va='center')
        ax.scatter(coordinates[lab_idx,0], coordinates[lab_idx,1],alpha = alpha)

    plt.show()

def plot_clusters_w_feat_plot(coordinates, labels, alpha=.15,medData=None, *args):
    nplots = len(args)+1
    nrows = nplots//2 + 1
    ncols = nplots if nplots < 3 else 2
    plot_ind = 1

    fig = plt.figure(figsize=(ncols*11,nrows*8))
    ax = fig.add_subplot(nrows, ncols, plot_ind); plot_ind +=1 
    ax.set_title('Clusters');  ax.set_xticklabels([]); ax.set_yticklabels([])
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(labels))))
    for ii,i in enumerate(set(labels) - set([-1])):
        lab_idx = (labels==i)
        coor = np.mean(coordinates[lab_idx,:],axis=0)
        ax.text(coor[0],coor[1], str(i), ha='center', va='center')
        ax.scatter(coordinates[lab_idx,0], coordinates[lab_idx,1],alpha = alpha,color=colors[ii])

    for title in args:
        if medData[title].dtype == np.object:
            ax = fig.add_subplot(nrows, ncols, plot_ind); plot_ind +=1 
            ax.set_title(title); ax.set_xticklabels([]); ax.set_yticklabels([])
        
            m_y = medData[title]
            smy = set(m_y)
            n_nums = len(set([extract_num(k) for k in smy]) - set([-1]))
            if len(smy)<4: t_cmap=plt.cm.tab10
            elif len(smy)<6: t_cmap=plt.cm.hsv
            elif n_nums>5: t_cmap = plt.cm.cool
            else: t_cmap=plt.cm.tab10
            colors = t_cmap(np.linspace(0, 1, len(smy)))

            for ii,i in enumerate(sorted(map(str,set(m_y)), key = lambda x: (extract_num(x),x))):
                lab_idx = (m_y==i)
                ax.scatter(coordinates[lab_idx,0], coordinates[lab_idx,1],alpha = alpha,label = i,color=colors[ii])
            ax.legend(loc='best', shadow=True)
        else:
            title = title
            ax = fig.add_subplot(nrows, ncols, plot_ind); plot_ind +=1 
            ax.set_title(title); ax.set_xticklabels([]); ax.set_yticklabels([])
            
            m_y = medData[title].values
            pl = ax.scatter(coordinates[:,0], coordinates[:,1],alpha = alpha,cmap=plt.cm.jet,c = m_y)
            cbar = fig.colorbar(pl)

    plt.show()
    
def plot_clusters_w_feat_save(coordinates, labels, alpha=.15,medData=None, *args):

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Clusters');  ax.set_xticklabels([]); ax.set_yticklabels([])
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(labels))))
    for ii,i in enumerate(set(labels) - set([-1])):
        lab_idx = (labels==i)
        coor = np.mean(coordinates[lab_idx,:],axis=0)
        ax.text(coor[0],coor[1], str(i), ha='center', va='center')
        ax.scatter(coordinates[lab_idx,0], coordinates[lab_idx,1],alpha = alpha,color=colors[ii])
    fig.savefig('plots/clusters.png')
    
    for arg_ind,title in enumerate(args):
        
        if medData[title].dtype == np.object:
            fig = plt.figure(figsize=(11,8))
            ax = fig.add_subplot(1, 1,1)
            ax.set_title(title); ax.set_xticklabels([]); ax.set_yticklabels([])
        
            m_y = medData[title]
            smy = set(m_y)
            n_nums = len(set([extract_num(k) for k in smy]) - set([-1]))
            if len(smy)<4: t_cmap=plt.cm.tab10
            elif len(smy)<6: t_cmap=plt.cm.hsv
            elif n_nums>5: t_cmap = plt.cm.cool
            else: t_cmap=plt.cm.tab10
            colors = t_cmap(np.linspace(0, 1, len(smy)))

            for ii,i in enumerate(sorted(map(str,set(m_y)), key = lambda x: (extract_num(x),x))):
                lab_idx = (m_y==i)
                ax.scatter(coordinates[lab_idx,0], coordinates[lab_idx,1],alpha = alpha,label = i,color=colors[ii])
            ax.legend(loc='best', shadow=True)
        else:
            fig = plt.figure(figsize=(13,8))
            title = title
            ax = fig.add_subplot(1, 1,1)
            ax.set_title(title); ax.set_xticklabels([]); ax.set_yticklabels([])
            
            m_y = medData[title].values
            pl = ax.scatter(coordinates[:,0], coordinates[:,1],alpha = alpha,cmap=plt.cm.jet,c = m_y)
            cbar = fig.colorbar(pl)
        for ii,i in enumerate(set(labels) - set([-1])):
            lab_idx = (labels==i)
            coor = np.mean(coordinates[lab_idx,:],axis=0)
            ax.text(coor[0],coor[1], str(i), ha='center', va='center')
        fig.savefig(f'plots/{arg_ind}_{title}.png')



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