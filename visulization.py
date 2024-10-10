import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch

dataset = 'mutagen'

colors = ['orange', 'red', 'lime', 'lightseagreen', 'royalblue', 'orchid', 'tan', 'green', 'blue', 'navy', 'darksalmon','bisque', 'indigo', 'darkslategray']
atom = ["C","O","Cl","H","N","F","Br","S","P","I","Na","K","Li","Ca"]

def visual(dataset):
    for explain_type in ['global']:
        feats = torch.load('./data/{}/explain/{}/feats.pt'.format(dataset,explain_type))
        graph_indictor = torch.load('./data/{}/explain/{}/indicator.pt'.format(dataset,explain_type))
        edge_index = torch.load('./data/{}/explain/{}/edge_idx.pt'.format(dataset,explain_type))
        node_index = torch.load('./data/{}/explain/{}/node_idx.pt'.format(dataset,explain_type))
        node_score = torch.load('./data/{}/explain/{}/node_score.pt'.format(dataset,explain_type))
        edge_score = torch.load('./data/{}/explain/{}/edge_score.pt'.format(dataset,explain_type))
        label = torch.load('./data/{}/explain/{}/label.pt'.format(dataset,'local'))
        prototype_graph = torch.load('./data/{}/explain/{}/prototype_graph.pt'.format(dataset,explain_type))

        try:
            os.mkdir('./visual/{}'.format(dataset))
        except:
            pass

        try:
            for i in range(prototype_graph.shape[0]):
                os.mkdir('./visual/{}/prototype_{}'.format(dataset,i))
        except:
            pass

        edge_l = edge_index[0].numpy()
        edge_r = edge_index[1].numpy()
        edge_score = edge_score.numpy()
        graph_indictor = graph_indictor.numpy()
        feats = feats.numpy()

        for graph_id in np.unique(graph_indictor):

            node_k, edge_k = 3, 3

            if graph_id == 133:
                node_k = 5
            elif graph_id == 164:
                node_k = 9

            min_nodes = int(node_index[np.where(graph_indictor == graph_id)].min())
            max_nodes = int(node_index[np.where(graph_indictor == graph_id)].max())
            edge_idx = np.where((min_nodes<=edge_l) & (edge_l<=max_nodes))
            left = edge_l[edge_idx]
            right = edge_r[edge_idx]
            e_score = edge_score[edge_idx].squeeze()
            node_idx = np.where(graph_indictor == graph_id)
            node_id = node_index[node_idx]
            n_score = node_score[node_idx].squeeze().numpy()

            orgin_edges = [(l, r) for l, r in zip(left, right)]

            G = nx.Graph()
            G.add_nodes_from(range(min_nodes, max_nodes + 1))
            G.add_edges_from(orgin_edges)

            if edge_k < e_score.size:
                thres = np.sort(e_score)[-edge_k]
            else:
                thres = 0

            reserve_id = np.where(e_score >= thres)
            reserve_edges = [(r, c) for r, c in zip(left[reserve_id], right[reserve_id])]
            pos_edges = [(u, v) for (u, v) in reserve_edges if u in G.nodes() and v in G.nodes()]

            if node_k < n_score.size:
                thres = np.sort(n_score)[-node_k]
            else:
                thres = 0

            reserve_id = np.where(n_score >= thres)
            reserve_nodes = [i for i in node_id[reserve_id].numpy()]
            pos_nodes = [v for v in reserve_nodes if v in G.nodes()]

            pos = nx.kamada_kawai_layout(G)

            feat = feats[node_idx, :].squeeze(0)
            node_label = [int(np.nonzero(x)[0]) for x in feat]
            max_label = np.max(node_label) + 1
            label2nodes = []
            for i in range(max_label):
                label2nodes.append([])
            for i in range(min_nodes, max_nodes + 1):
                if i in G.nodes():
                    label2nodes[node_label[i - min_nodes]].append(i)
            nx.draw_networkx_nodes(G, pos, nodelist=pos_nodes, node_size=500, node_color='black')
            for i in range(max_label):
                nx.draw_networkx_nodes(G, pos, nodelist=label2nodes[i], node_color=colors[i], node_size=200)
                nx.draw_networkx_labels(G, pos, {k: atom[i] for k in label2nodes[i]})
            nx.draw_networkx_edges(G, pos, width=2, edge_color='grey')
            nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=7)
            plt.axis('off')
            for num in range(prototype_graph.shape[0]):
                if graph_id in prototype_graph[num]:
                    prototype = num
            plt.savefig(f'./visual/{dataset}/prototype_{prototype}/{graph_id}')
            plt.show()



if __name__ == '__main__':
    visual(dataset)