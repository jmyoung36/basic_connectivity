# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:14:22 2016

@author: jonyoung
"""

# import what we need
import numpy as np
import connectivity_utils as utils
import networkx as nwx
import bct

# define function giving log-euclidean kernel between a pair of SPD connectvity matrices
def get_graph_metrics(connectivity_vector) :
    
    # reshape into matrix
    connectivity_matrix = np.reshape(connectivity_vector, (90, 90))
    
    # convert to networkx graph
    connectivity_graph = nwx.from_numpy_matrix(connectivity_matrix)
    
    # convert to distance graph as some metrics need this instead
    distance_matrix = connectivity_matrix
    distance_matrix[distance_matrix == 0] = np.finfo(np.float32).eps
    distance_matrix = 1.0 / distance_matrix
    distance_graph = nwx.from_numpy_matrix(distance_matrix)
    
    # intialise vector of metrics
    metrics = np.zeros((21,))
    # fill the vector of metrics
    # 1 and 2: degree distribution
    degrees = np.sum(connectivity_matrix, axis = 1)
    metrics[0] = np.mean(degrees)
    metrics[1] = np.std(degrees)
    
    # 3 and 4: weight distribution
    weights = np.tril(connectivity_matrix, k = -1)
    metrics[2] = np.mean(weights)
    metrics[3] = np.std(weights)

    # 5: average shortest path length
    # transform weights to distances so this makes sense    
    metrics[4] = nwx.average_shortest_path_length(distance_graph, weight='weight')

    # 6: assortativity
    metrics[5] = nwx.degree_assortativity_coefficient(connectivity_graph, weight='None')
    
    # 7: clustering coefficient
    metrics[6] = nwx.average_clustering(connectivity_graph, weight='weight')
    
    # 8: transitivity
    metrics[7] = nwx.transitivity(connectivity_graph)
    
    # 9 & 10: local and global efficiency
    metrics[8] = np.mean(bct.efficiency_wei(connectivity_matrix, local=True))
    metrics[9] = bct.efficiency_wei(connectivity_matrix, local=False)
    
    # 11: Clustering coefficient
    metrics[10] = np.mean(nwx.clustering(connectivity_graph, weight='weight').values())
    
    # 12 & 13: Betweeness centrality
    metrics[11] = np.mean(nwx.betweenness_centrality(distance_graph, weight='weight').values())
    metrics[12] = np.mean(nwx.current_flow_betweenness_centrality(distance_graph, weight='weight').values())
    
    # 14: Eigenvector centrality
    metrics[13] = np.mean(nwx.eigenvector_centrality(distance_graph, weight='weight').values())
    
    # 15: Closeness centrality
    metrics[14] = np.mean(nwx.closeness_centrality(distance_graph, distance='weight').values())
    
    # 16: PageRank
    metrics[15] = np.mean(nwx.pagerank(connectivity_graph, weight='weight').values())
    
    # 17: Rich club coefficient
    metrics[16] = np.mean(nwx.rich_club_coefficient(connectivity_graph).values())
    
    # 18: Density    
    metrics[17] = bct.density_und(connectivity_matrix)[0]
    
    # 19, 20, 21: Eccentricity, radius, diameter
    spl_all = nwx.shortest_path_length(distance_graph, weight='weight')
    eccs = np.zeros(90,)
    for i in range(90) :
        
        eccs[i] = np.max(spl_all[i].values())
        
    metrics[18] = np.mean(eccs)
    metrics[19] = np.min(eccs)
    metrics[20] = np.max(eccs)  
    
    return metrics

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/kernels/'

# read in connectivity data and labels
connectivity_data = utils.load_connectivity_data(data_dir)
labels = np.array([utils.load_labels(data_dir), ])

print np.shape(connectivity_data)

# set negative connectivities to 0
connectivity_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, connectivity_data)

# extract vectors of graph metrics
metrics_data = np.apply_along_axis(lambda x: get_graph_metrics(x), 1, connectivity_data)

print np.shape(metrics_data)

# caclulate the kernel values
K = np.dot(metrics_data, np.transpose(metrics_data))

# attach the labels and save
K = np.hstack((np.transpose(labels), K))
np.savetxt(kernel_dir + 'K_metrics.csv', K, delimiter=',')