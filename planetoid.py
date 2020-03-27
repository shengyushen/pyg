import torch
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)

print(len(dataset))

print(dataset.num_classes)

print(dataset.num_node_features)
