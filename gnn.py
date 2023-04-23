from data_prep import get_prepared_data
import numpy as np
import networkx as nx
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import torch_geometric.nn as torchnn
import matplotlib.pyplot as plt


def data_refining_pipeline(features_dic, known_edges, known_non_edges):
    
    """data_refining pipeline
    
    Refines the data to form adjancy matrices as well as a node features matrix

    Returns:
        features_matrix, true_adjancy_matrix, false_adjancy_matrix
        """
    features_matrix = []
    
    for node in features_dic:
        features = list(features_dic[node].values())
        features_matrix.append(features)
    
    features_matrix = np.array(features_matrix, dtype = np.float32)

    features_matrix = torch.from_numpy(features_matrix).to(torch.long)
    true_adjancy_matrix = torch.from_numpy(np.array(known_edges).T).to(torch.long)
    false_adjancy_matrix = torch.from_numpy(np.array(known_non_edges).T).to(torch.long)
    
    return features_matrix, true_adjancy_matrix, false_adjancy_matrix

class LinkPredictionGNN(torch.nn.Module):
    
    """LinkPredictionGNN
    
    Implementation of our GNN model with number of hidden channels set to 16
    """
    def __init__(self, in_channels, hidden_channels = 16):
        
        super(LinkPredictionGNN, self).__init__()
        
        self.conv1 = torchnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torchnn.GCNConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)
        
        output = []
    
        for edge in torch.transpose(edge_index, 0, 1):
        
            source_node, target_node = edge
            source_node, target_node = source_node.item(), target_node.item()
            
            source_feature, target_feature = x[source_node, :], x[target_node, :]

            output.append(torch.dot(source_feature, target_feature).reshape(1))            
        
        output = torch.cat(output)
        output = torch.sigmoid(output)
        
        return output


def train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index, device):
    """train

    Function used to train the model on training data
    
    Args:
        model : The untrained model
        optimizer : Optimizer to be used here
        x : Node Feature Matrix
        train_pos_edge_index : Adjancy Matrix of known Edges
        train_neg_edge_index : Adjancy Matrix of known non-Edges
        device : The device to be used

    Returns:
        The loss on training data obtained after training process
    """
    model.train()
    
    optimizer.zero_grad()
    
    # Positive examples
    pos_out = model(x.float().to(device), train_pos_edge_index.to(device))
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones(pos_out.size(0), device=device))

    # Negative examples
    neg_out = model(x.float().to(device), train_neg_edge_index.to(device))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros(neg_out.size(0), device=device))

    # Total loss
    loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
    
    return pos_loss.item() + neg_loss.item()

def launch_training_process(model, x, train_pos_edge_index, train_neg_edge_index, num_epochs, learning_rate, device):

    """launch_training_process
    
    Launches the training process of a model for a certain number of epochs and prints the evolution graph of the loss

    Returns:
        _type_: _description_
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    epochs_list = []
    loss_list = []
    
    for epoch in range(num_epochs):
        loss = train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index, device)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss))
        epochs_list.append(epoch + 1)
        loss_list.append(loss)

    plt.plot(epochs_list, loss_list, 'm-o')

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Value of Binary Cross-Entropy Loss', fontsize=14)
    plt.show()    

    return model


"""
GNN whole process
"""

_, features, known_edges, known_non_edges, unknown_edges, unknown_non_edges = get_prepared_data(use = 'gnn')
features_matrix, true_adjancy_matrix, false_adjancy_matrix = data_refining_pipeline(features, known_edges, known_non_edges)
_, test_true_adjancy_matrix, test_false_adjancy_matrix = data_refining_pipeline(features, unknown_edges, unknown_non_edges)

y_true = torch.ones(test_true_adjancy_matrix.size(1))
y_false = torch.zeros(test_false_adjancy_matrix.size(1))

y_test = torch.cat([y_true, y_false], dim=0)
X_test = torch.cat([test_true_adjancy_matrix, test_false_adjancy_matrix], dim = -1)

random_index = torch.randperm(y_test.size(0))

X_test = X_test[:, random_index]
y_test = y_test[random_index]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LinkPredictionGNN(features_matrix.shape[1], 64).to(device)

model = launch_training_process(model, features_matrix, true_adjancy_matrix, false_adjancy_matrix, 25, 0.1, device)

#y_hat_test = model(features_matrix.float().to(device), X_test.to(device)).detach().numpy()
#print(confusion_matrix(y_test, y_hat_test))