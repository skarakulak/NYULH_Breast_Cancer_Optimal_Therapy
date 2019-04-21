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