#https://github.com/yixinliu233/SIGNET/blob/ba7529f301cc0cd8a73339908f95e3b068e695a1/datasets/mutag.py#L141

import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data


class BA(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     raise NotImplementedError

    def process(self):
        data = self.raw_dir.split('/')[1]
        with open(self.raw_dir + f'/{data}.pkl', 'rb') as fin:
            _, original_features, graph_labels, node_type_lists, edge_label_lists, edge_lists, motif_lists  = pkl.load(fin)
        data_list = []

        # len(edge_list) = len(node_type_lists) = original_labels.shape[0] =4337    node_type_lists -- > [[0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3],...]
        for i in range(len(graph_labels)): # i is the label of graphs
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T
            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)  # y=0 mutag  y=1 non-mutag
            x = torch.tensor(original_features[i][:num_nodes]).float()
            assert original_features[i][num_nodes:].sum() == 0
            edge_label = torch.tensor(edge_label_lists[i]).float()
            motif_type = torch.tensor(motif_lists[i]).float()
            node_label = torch.tensor(node_type_lists[i]).float()

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label,
                                  node_type=torch.tensor(node_type_lists[i]),motif_type=torch.tensor(motif_type)))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    graphs = BM(root='./BA_TYPE')