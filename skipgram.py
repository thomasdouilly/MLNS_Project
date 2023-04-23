from data_prep import get_prepared_data
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from gensim.models import Word2Vec
import pandas as pd


G, X_train, y_train, X_test, y_test = get_prepared_data()

def generate_random_walk(graph, root, L):
    """
    :param graph: networkx graph
    :param root: the node where the random walk starts
    :param L: the length of the walk
    :return walk: list of the nodes visited by the random walk
    """
    walk = [str(root)]
    while len(walk) < L:
        current_node = int(walk[-1])
        if graph.degree[current_node] == 0:
            walk = [str(current_node)]*L
            return walk
        candidates = list(nx.neighbors(graph, current_node))
        next = np.random.choice(candidates)
        walk.append(str(next))
    return walk



def deep_walk(graph, N, L):
    '''
    :param graph: networkx graph
    :param N: the number of walks for each node
    :param L: the walk length
    :return walks: the list of walks
    '''
    walks = []
    np.random.seed(4) # fix random seed to obtain same random shuffling when repeating experiment
    nodes = list(graph.nodes)
    for _ in range(N):
        np.random.shuffle(nodes) # shuffle the ordering of nodes, it helps speed up the convergence of stochastic gradient descent
        for node in nodes:
            # generate a random walk from the current visited node
            walk = generate_random_walk(graph, node, L)
            walks.append(walk)
        
    return walks

def edge_prediction(node2embedding, train_samples, test_samples, train_labels, test_labels):
    
    # --- Construct feature vectors for edges ---
    feature_func = lambda x,y: abs(x-y)
    
    train_features = [feature_func(node2embedding[str(edge[0])], node2embedding[str(edge[1])]) for edge in train_samples]
    test_features = [feature_func(node2embedding[str(edge[0])], node2embedding[str(edge[1])]) for edge in test_samples]
    
    # --- Build the model and train it ---
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(train_features, train_labels)

    test_preds = clf.predict(test_features)
    acc_test = accuracy_score(test_labels, test_preds)

    print(f'Accuracy on test set: {100*acc_test}%')        
    return 100*acc_test

# Define parameters
num_of_walks = [50, 100, 200, 300, 400, 500, 750, 1000]
walk_length = [1, 10, 20, 50, 100]
window_sizes = [2, 5, 10, 20]
embedding_size = 32
results = []



"""
Skipgram whole process
"""

nb = len(num_of_walks)*len(walk_length)*len(window_sizes)
i = 0
for N in num_of_walks:
    for L in walk_length:
        residual_walks = deep_walk(graph=G, N=N, L=L)
        for ws in window_sizes:
            # Learn representations of nodes
            model = Word2Vec(residual_walks, vector_size=embedding_size, sg=1, window=ws, min_count=0, workers=7, hs=1, epochs=1, negative=0)

            acc_test = edge_prediction(model.wv, X_train, X_test, y_train, y_test)
            results.append({
                'num_of_walks': N,
                'walk_length': L,
                'window_size': ws,
                'embedding_method': 'DeepWalk',
                'Accuracy on test set (%)': acc_test
            })
            print(f'{100*i/nb}% done')
            i+=1

df = pd.DataFrame(results)
df_sorted = df.sort_values(by='Accuracy on test set (%)', ascending=False)
print(df_sorted.head())
df_sorted.to_csv('sorted_data.csv', index=False)