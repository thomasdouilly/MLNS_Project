import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random as rd


def read_data():
    
    with open("data/large_twitch_edges.csv", "r") as f:
        reader = csv.reader(f)
        edges_set = list(reader)[1:]

    edges_set = [(int(element[0]), int(element[1])) for element in edges_set]

    
    with open("data/large_twitch_features.csv", "r") as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        features = reader_list[0]
        features_set = reader_list[1:]
    
    features_dic = {}

    for n in range(len(features_set)):
    
        N_features = len(features)
        features_node = {}
        
        for i in range(N_features):
            
            features_node[features[i]] = features_set[n][i]
    
        features_dic[n] = features_node
    
    return edges_set, features_dic

def feature_builder(pre_features):
    
    features = {}
    
    for vertice in pre_features:
        
        pre_processed_features = pre_features[vertice]
        
        views = int(pre_processed_features['views'])
        mature = int(pre_processed_features['mature'])
        lifetime = int(pre_processed_features['life_time'])
        creation = int(pre_processed_features['created_at'][:4])
        update = int(pre_processed_features['updated_at'][:4])
        id = int(pre_processed_features['numeric_id'])
        affiliate = int(pre_processed_features['affiliate'])
        
        features[vertice] = {'views' : views, 'mature' : mature, 'life_time' : lifetime, 'created_at' : creation, 'updated_at' : update, 'numeric_id' : id, 'affiliate' : affiliate}
    
    return features



def nationality_filtering(features, edges, nationality = None):
    
    if nationality != None:
        
        def dic_nationality_filter(pair):
            _, x = pair
            return (x['language'] == nationality) and (x['dead_account'] == '0')

        filtered_features = dict(filter(dic_nationality_filter, features.items()))

        def list_nationality_filter(x):
            return ((x[0] in filtered_features) and (x[1] in filtered_features))
        
        filtered_edges = list(filter(list_nationality_filter, edges))
    
    else:
        filtered_features = features
        filtered_edges = edges
    
    return filtered_features, filtered_edges


def split_data(data_to_split, rate = 0.2):
    
    rd.shuffle(data_to_split)
    N = len(data_to_split)
    split_1, split_2 = data_to_split[:int(N*rate)], data_to_split[int(N*rate):]
    
    return split_1, split_2


def get_graph(edges, features):
    
    graph = nx.Graph()
    
    for key in features:
        graph.add_node(key)
    
    for [source_node, target_node] in edges:
        graph.add_edge(source_node, target_node)
        
    nx.set_node_attributes(graph, features)
    
    return graph

def arrange(known_edges, known_non_edges, unknown_edges, unknown_non_edges, N_test = 1000):
    
    print('Composition of the datasets :')
    print('Training dataset : ', len(known_edges), " edges and ", len(known_non_edges), " non_edges.")
    print('Testing dataset : ', len(unknown_edges), " edges and ", len(unknown_non_edges), " non_edges.")

    
    X_test = np.concatenate([unknown_edges, unknown_non_edges])[:N_test]
    y_test = np.array([1] * len(unknown_edges) + [0] * len(unknown_non_edges))[:N_test]
    
    X_train = np.concatenate([known_edges, known_non_edges])
    y_train = np.array([1] * len(known_edges) + [0] * len(known_non_edges))
    
    return X_train, y_train, X_test, y_test

def get_prepared_data(use = 'classical'):
    
    edges, complete_features = read_data()
    
    features, edges = nationality_filtering(complete_features, edges, 'NL')
    
    complete_features = feature_builder(complete_features)
    features = feature_builder(features)
    
    known_edges, unknown_edges = split_data(edges, rate = 0.8)

    graph = get_graph(known_edges, features)

    graph_non_edges = list(nx.non_edges(get_graph(edges, features)))
    known_non_edges, unknown_non_edges = split_data(graph_non_edges, rate = 0.8)

    X_train, y_train, X_test, y_test = arrange(known_edges, known_non_edges, unknown_edges, unknown_non_edges, N_test = 2000)
    
    if use == 'classical':
        return graph, X_train, y_train, X_test, y_test
    
    if use == 'gnn':
        return graph, complete_features, known_edges, known_non_edges, unknown_edges, unknown_non_edges
