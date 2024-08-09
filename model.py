import torch, torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from layers import GIN
import numpy as np


class GAD_Explainer(nn.Module):
    def __init__(self,args,input_dim):
        super(GAD_Explainer,self).__init__()

        self.epsilon = 1e-7
        self.device = args.device
        self.encoder = GIN(args,input_dim)
        self.MLP = nn.Sequential(
            nn.Linear(args.out_dim*2, args.out_dim),
            nn.ReLU(),
            nn.Linear(args.out_dim, 1)
        )
        self.r = args.r
        self.batch_size = args.batch_size

        self.prototypes = nn.Parameter(torch.FloatTensor(args.n_prot, args.out_dim),requires_grad=True)
        #self.prototypes = nn.Parameter(torch.rand(args.n_prot, args.out_dim), requires_grad=True)
        nn.init.orthogonal(self.prototypes)
        self.assign_train = None

    def forward(self,graphs,batch_num=0):

        graph_embs, node_embs = self.encoder(graphs.x,graphs.edge_index,graphs.batch)
        if self.training:
            assign_prot = self.assign_train[batch_num]
        else:
            similarity = self.prototype_distances(graph_embs, self.prototypes)
            assign_prot = torch.argmax(similarity, dim=1, keepdim=True)

        node_bern, edge_bern = self.sample_subgraph(node_embs,graphs.edge_index,assign_prot,graphs.batch)
        x = graphs.x * node_bern
        subgraph_embs,_ = self.encoder(x, graphs.edge_index, graphs.batch, edge_atten=edge_bern)
        similarity = self.prototype_distances(subgraph_embs,self.prototypes)  # batch * num_prototypes  -->  graph1 -> [p1,p2,p3]

        NCE_loss = self.NCE_loss(similarity,batch_num)

        KL_node = (node_bern*torch.log(node_bern/self.r + self.epsilon) + (1-node_bern)*torch.log((1-node_bern)/(1-self.r+self.epsilon)+self.epsilon)).mean()
        KL_edge = (edge_bern*torch.log(edge_bern/(self.r*self.r) + self.epsilon) + (1-edge_bern)*torch.log((1-edge_bern)/(1- (self.r*self.r) + self.epsilon)+self.epsilon)).mean()
        KL_loss = KL_node + KL_edge

        data_sim = self.prototype_distances(subgraph_embs,subgraph_embs)

        if self.training:
            assign_prot = torch.argmax(similarity, dim=1, keepdim=True)
            self.assign_train[batch_num] = assign_prot

        return KL_loss, NCE_loss, similarity, node_bern, edge_bern, data_sim, subgraph_embs, node_embs

    def sample_subgraph(self,node_embs,edge_index,assign_prot,graph_idx):

        node_prototype = assign_prot[graph_idx].squeeze(-1)
        # gumble softmax
        node_prob = self.MLP(torch.cat([node_embs,self.prototypes[node_prototype]],dim=-1))
        temp = 1
        if self.training:
            random_noise = torch.empty_like(node_prob).uniform_(self.epsilon, 1 - self.epsilon)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            node_bern = ((node_prob + random_noise) / temp).sigmoid()
        else:
            node_bern = (node_prob).sigmoid()

        edge_bern = node_bern[edge_index[0]] * node_bern[edge_index[1]]

        return node_bern, edge_bern

    def prototype_distances(self, x, x_2):

        #distance = torch.cdist(x,x_2,p=2)**2
        #similarity = torch.log( (distance + 1) / (distance + self.epsilon) )

        x1_abs = x.norm(dim=1)
        x1_abs[torch.nonzero(x1_abs == 0)] += self.epsilon

        x2_abs = x_2.norm(dim=1)
        x2_abs[torch.nonzero(x2_abs == 0)] += self.epsilon

        similarity = torch.einsum('ik,jk->ij', x, x_2) / torch.einsum('i,j->ij', x1_abs, x2_abs)

        if self.training:
            return nn.Softmax(dim=-1)(similarity)
        else:
            return similarity

    def NCE_loss(self,similarity,batch_num,temp=0.2):

        if self.training:
            assign_prot = self.assign_train[batch_num]
        else:
            assign_prot = torch.argmax(similarity, dim=1, keepdim=True)

        assign_mx = torch.zeros(similarity.shape).to(self.device).scatter_(1,assign_prot,1)  # hard one-hot  --> graph1 -> [1,0,0] if graph1 belongs to p1
        similarity = torch.exp(similarity / temp)
        positive_sim = similarity * assign_mx
        negative_sim = similarity * (1 - assign_mx)
        NCE_loss = (positive_sim.sum(dim=1)) / (negative_sim.sum(dim=1))
        NCE_loss = - torch.log(NCE_loss).mean()

        return NCE_loss

    def init_prototypes(self,train_loader):
        with torch.no_grad():
            self.assign_train = [0] * len(train_loader)
            batch_num = 0
            for graphs in train_loader:
                graphs = graphs.to(self.device)
                graph_embs, node_embs = self.encoder(graphs.x, graphs.edge_index, graphs.batch)
                similarity = self.prototype_distances(graph_embs, self.prototypes)
                assign_prot = torch.argmax(similarity, dim=-1, keepdim=True)
                self.assign_train[batch_num] = assign_prot
                batch_num += 1

    def plot_fig(self,train_loader,epoch=0):
        with torch.no_grad():
            all_embs = []
            all_assign = []
            all_motif = []
            for i,graphs in enumerate(train_loader):
                graphs = graphs.to(self.device)
                graph_embs, node_embs = self.encoder(graphs.x, graphs.edge_index, graphs.batch)
                node_bern, edge_bern = self.sample_subgraph(node_embs, graphs.edge_index, self.assign_train[i], graphs.batch)
                x = graphs.x * node_bern
                subgraph_embs, _ = self.encoder(x, graphs.edge_index, graphs.batch, edge_atten=edge_bern)
                all_embs.append(subgraph_embs)
                all_assign.append(self.assign_train[i])
                all_motif.append(graphs.motif_type)
            x = torch.cat(all_embs)
            assign = torch.cat(all_assign).squeeze()
            motif = torch.cat(all_motif).squeeze()
            print('assignment distribution',torch.bincount(assign))
            ts = PCA(n_components=2,random_state=0)
            x_2d = ts.fit_transform(x.detach().cpu().numpy())
            prot_2d = ts.fit_transform(self.prototypes.detach().cpu().numpy())
            colors = ['b','r','g','k','m','y','p']
            #for p in range(self.prototypes.shape[0]):
            plt.scatter(x_2d[motif == 1, 0], x_2d[motif == 1, 1],label='NO2',c=colors[0])
            plt.scatter(x_2d[motif == 2, 0], x_2d[motif == 2, 1], label='NH2', c=colors[1])

            plt.legend(prop={'size': 15})
            for p in range(self.prototypes.shape[0]):
                plt.scatter(prot_2d[p, 0], prot_2d[p, 1], marker="x", s=60, label=p, c='black')
                plt.text(prot_2d[p, 0], prot_2d[p, 1],f'P{p+1}',fontsize=18)
            #plt.title(f'epoch:{epoch}')
            plt.plot()
            #plt.show()
            plt.savefig(f'visual/{epoch}.jpg')
            plt.clf()

    def del_proto(self, train_embs,num_train):
        with torch.no_grad():
            similarity = self.prototype_distances(train_embs,self.prototypes)
            prototypes_sim = self.prototype_distances(self.prototypes, self.prototypes)
            data_sim = self.prototype_distances(train_embs,train_embs)
            MMD_loss = prototypes_sim.mean() + data_sim.mean() - 2 * similarity.mean()

            MMD_change = []
            for proto_index in range(self.prototypes.shape[0]):
                res_similarity = similarity[torch.arange(similarity.size(0)) != proto_index]
                res_prototypes_sim = prototypes_sim[
                    torch.arange(prototypes_sim.size(0)) != proto_index]  # remove rows as index proto_index
                res_prototypes_sim = res_prototypes_sim.t()[
                    torch.arange(res_prototypes_sim.size(1)) != proto_index].t()  # remove cols
                MMD_reduction = (prototypes_sim.mean() - res_prototypes_sim.mean()) - 2 * (
                            similarity.mean() - res_similarity.mean())
                MMD_change.append(MMD_reduction)

            less_reduction = max(MMD_change) # In general, less_reduction < 0, means remove prototypes, lead the MMD increase
            theta = 2/num_train # if you want to delete more prototypes, set a bigger theta
            if -(less_reduction / MMD_loss) < theta:  # when to stop remove prototypes
                remove_proto = MMD_change.index(less_reduction)
                prototypes_sim[remove_proto, remove_proto] = 0
                replace_proto = int(torch.max(prototypes_sim[remove_proto], 0)[1])
                self.prototypes = nn.Parameter(self.prototypes[torch.arange(self.prototypes.size(0)) != remove_proto])
                # refresh assign list after remove protos
                for b_num in range(len(self.assign_train)):
                    self.assign_train[b_num] = torch.where(self.assign_train[b_num] == remove_proto, replace_proto, self.assign_train[b_num])
                    self.assign_train[b_num] = torch.where(self.assign_train[b_num] >= remove_proto, self.assign_train[b_num]-1, self.assign_train[b_num])
                print(f'After Delete, Prototype Number is {self.prototypes.shape[0]}')
                return remove_proto


    def update_prototypes(self,train_embs,train_sim,train_loader): #Momentum update prototype
        with torch.no_grad():
            #train_sim = nn.Softmax(dim=0)(train_sim)
            assign_train = torch.argmax(train_sim, dim=-1, keepdim=True)
            assign_mx = torch.zeros(train_sim.shape).to(self.device).scatter_(1,assign_train,1)
            assign_sim = train_sim * assign_mx
            for i in range(self.prototype[0]):
                update_vector = torch.mm(assign_sim[:,i].unsqueeze(0),train_embs).squeeze()
                r = 0.9
                self.prototypes[i] = r * self.prototypes[i] + (1-r) * update_vector

            for i,graphs in enumerate(train_loader):
                self.assign_train[i] = assign_train[i*self.batch_size:i*self.batch_size+len(graphs),:]

            print(torch.bincount(assign_train.squeeze()))

    def kmeans_init(self,train_loader): # initial based on K-means
        with torch.no_grad():
            all_embs = []
            for graphs in train_loader:
                graphs = graphs.to(self.device)
                graph_embs, node_embs = self.encoder(graphs.x, graphs.edge_index, graphs.batch)
                all_embs.append(graph_embs)
            x = torch.cat(all_embs)
            kmeans = KMeans(n_clusters=self.prototype[0], random_state=42).fit(x.detach().cpu())
            self.prototypes.copy_(torch.from_numpy(kmeans.cluster_centers_).to(torch.float32))

            assign_onelst = kmeans.labels_
            self.assign_train = [0] * len(train_loader)
            for i,graphs in enumerate(train_loader):
                self.assign_train[i] = torch.from_numpy(assign_onelst[i*self.batch_size:i*self.batch_size+len(graphs)]).unsqueeze(-1).to(torch.long).to(self.device)

            #print(torch.bincount(torch.from_numpy(assign_onelst)))

    def project(self,train_loader):
        with torch.no_grad():
            all_embs = []
            all_assign = []
            for i,graphs in enumerate(train_loader):
                graphs = graphs.to(self.device)
                graph_embs, node_embs = self.encoder(graphs.x, graphs.edge_index, graphs.batch)
                node_bern, edge_bern = self.sample_subgraph(node_embs, graphs.edge_index, self.assign_train[i], graphs.batch)
                x = graphs.x * node_bern
                subgraph_embs, _ = self.encoder(x, graphs.edge_index, graphs.batch, edge_atten=edge_bern)
                all_embs.append(subgraph_embs)
                all_assign.append(self.assign_train[i])
            x = torch.cat(all_embs)
            assign = torch.cat(all_assign).squeeze()
            for i in range(self.prototype[0]):
                mask = (assign == i)
                self.prototypes[i].copy_(torch.mean(x[mask, :],dim=0))