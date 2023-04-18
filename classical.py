from data_prep import get_prepared_data
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

def feature_extractor(data, graph):
        
    feature_vector = []
    deg_centrality = nx.degree_centrality(graph)
    betweeness_centrality = nx.betweenness_centrality(graph)
    
    count = [0, 0]

    for edge in data:
        
        source_node, target_node = edge[0], edge[1]
        
        try:
            source_degree_centrality = deg_centrality[source_node]
            target_degree_centrality = deg_centrality[target_node]
            diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]
            hopcroft_coeff = list(nx.cn_soundarajan_hopcroft(graph, [(source_node, target_node)], community = 'affiliate'))[0][2]
            hopcroft_index = list(nx.ra_index_soundarajan_hopcroft(graph, [(source_node, target_node)], community = 'affiliate'))[0][2]
            jaccard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]
            pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]
            aai = list(nx.adamic_adar_index(graph, [(source_node, target_node)]))[0][2]
            
            count[0] += 1
            feature_vector.append(np.array([source_degree_centrality, target_degree_centrality, hopcroft_coeff, hopcroft_index, diff_bt, jaccard_coeff, pref_attach, aai]))
            
        except:
            count[1] += 1
            feature_vector.append(np.array([1] * 8)) 
            
    #print(count)
    return feature_vector

def prediction(graph, X_train, y_train, X_test):

    train_features = feature_extractor(X_train, graph)
    test_features = feature_extractor(X_test, graph)
    
    clf = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 1000, tol = 1e-12)
    clf.fit(train_features, y_train)

    train_preds = clf.predict(test_features)
    
    return train_preds



graph, X_train, y_train, X_test, y_test = get_prepared_data()

y_hat_test = prediction(graph, X_train, y_train, X_test)
print(confusion_matrix(y_test, y_hat_test))