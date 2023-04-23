from data_prep import get_prepared_data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

def get_array_DICN(graph, node):
    """get_array_DICN

    Computes the vector N, useful in the calculation of the DICN index

    Args:
        graph : The networkx graph containing data
        node : The label of the node for which the vector N is computed 

    Returns:
        N : A vector of size |V|
    """
    N = np.zeros(graph.number_of_nodes())
    first_order_neighbors = list(graph.neighbors(node))
    nodes = list(graph.nodes)

    for i in range(len(N)):
        
        tested_node = nodes[i]
                
        if node == tested_node:
            N[i] = graph.degree[node]

        elif tested_node in first_order_neighbors:
            N[i] = len(list(nx.common_neighbors(graph, node, tested_node))) + 1
        
        elif tested_node in [graph.neighbors(n) for n in first_order_neighbors]:
            N[i] = len(list(nx.common_neighbors(graph, node, tested_node)))
    
    return N

def DICN(graph, edge):
    
    """DICN

    Args:
        graph : The networkx graph containing data
        edge : The edge for which the DICN must me computed

    Returns:
        dicn : A float equals to the DICN index of the input edge
    """
    dicn = 0
    
    source_node, target_node = edge[0], edge[1]

    pre_N_i = get_array_DICN(graph, source_node)
    pre_N_j = get_array_DICN(graph, target_node)
    
    N = len(pre_N_i)
    
    N_i = np.array([pre_N_i[i] for i in range(N) if ((pre_N_i[i] > 0) or (pre_N_j[i] > 0))])
    N_j = np.array([pre_N_j[i] for i in range(N) if ((pre_N_i[i] > 0) or (pre_N_j[i] > 0))])
    
    
    if len(N_i)>1:

        corr_ij = pearsonr(N_i, N_j)[0]
        dicn = (1 + corr_ij) * (1 + len(list(nx.common_neighbors(graph, source_node, target_node))))
        
        if not ((dicn >= 0) or (dicn <= 0)):
            dicn = 0
        
    else:
        
        dicn = 0
        
    return dicn


def feature_extractor(data, graph):
        
    """feature_extractor
    
        graph : A networkx graph containing the data
        data : A set of couple of nodes to be considered
    
    Returns:
        feature_vector : A matrix containing the feature vectors computed for each tuple of the data list 
    """
    feature_vector = []
    deg_centrality = nx.degree_centrality(graph)
    betweeness_centrality = nx.betweenness_centrality(graph)
    
    count = [0, 0]

    for edge in data:
        
        source_node, target_node = edge[0], edge[1]
        
        try:
            # Calculation of all indexes
            source_degree_centrality = deg_centrality[source_node]
            target_degree_centrality = deg_centrality[target_node]
            diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]
            hopcroft_coeff = list(nx.cn_soundarajan_hopcroft(graph, [(source_node, target_node)], community = 'mature'))[0][2]
            hopcroft_index = list(nx.ra_index_soundarajan_hopcroft(graph, [(source_node, target_node)], community = 'mature'))[0][2]
            jaccard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]
            pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]
            aai = list(nx.adamic_adar_index(graph, [(source_node, target_node)]))[0][2]
            dicn = DICN(graph, edge)
            
            count[0] += 1
            #feature_vector.append(np.array([source_degree_centrality, target_degree_centrality, diff_bt, jaccard_coeff, pref_attach, aai]))
            #feature_vector.append(np.array([hopcroft_coeff, hopcroft_index]))
            #feature_vector.append(np.array([dicn]))
            feature_vector.append(np.array([source_degree_centrality, target_degree_centrality, diff_bt, jaccard_coeff, pref_attach, aai, hopcroft_coeff, hopcroft_index, dicn]))
        except:
            # In case of failure, sets all features to 0
            count[1] += 1
            #feature_vector.append(np.array([0] * 6))
            #feature_vector.append(np.array([0] * 2)) 
            #feature_vector.append(np.array([0]))
            feature_vector.append(np.array([0] * 9)) 
    
    print('Processing success / fail repartition : ', count)
    return feature_vector

def prediction(graph, X_train, y_train, X_test, lr = 0.01, ne = 100):
    """prediction
    
    Achieves the whole training and prediction process of the classical method
    """
    
    train_features = feature_extractor(X_train, graph)
    test_features = feature_extractor(X_test, graph)
    
    clf = GradientBoostingClassifier(learning_rate = lr, n_estimators = ne, tol = 1e-6)
    clf.fit(train_features, y_train)

    train_preds = clf.predict(test_features)
    
    return train_preds


""" 
Prediction with confusion matrix 
"""

graph, X_train, y_train, X_test, y_test = get_prepared_data()

y_hat_test = prediction(graph, X_train, y_train, X_test, lr = 2, ne = 1)
confusion = confusion_matrix(y_test, y_hat_test)
df_cm = pd.DataFrame(confusion, range(2), range(2))

sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".0f", cmap = 'viridis')

plt.show()


""" 
Learning rate and number of estimators finetuning 

lr_list = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100]
#lr_list = [1, 10]
ne_list = [1, 5, 10, 50, 100]
accuracy_list = []

graph, X_train, y_train, X_test, y_test = get_prepared_data()

for ne in ne_list:
    
    print('--- Making computations for number of estimators = ' + str(ne) + " ---")
    accuracy_line = []
    
    for lr in lr_list:
    
        print('- Making computations for learning rate = ' + str(lr) + "-")
        y_hat_test = prediction(graph, X_train, y_train, X_test, lr = lr, ne = ne)
        accuracy_line.append(accuracy_score(y_test, y_hat_test))
    
    accuracy_list.append(accuracy_line)

plt.plot(lr_list, accuracy_list[0], 'g-o', label = 'ne = 1')
plt.plot(lr_list, accuracy_list[1], 'b-o', label = 'ne = 5')
plt.plot(lr_list, accuracy_list[2], 'r-o', label = 'ne = 10')
plt.plot(lr_list, accuracy_list[3], 'k-o', label = 'ne = 50')
plt.plot(lr_list, accuracy_list[4], 'm-o', label = 'ne = 100')

plt.xlabel('Learning Rate', fontsize=14)
plt.xscale('log')
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.show()
"""