import numpy as np
import torch,time
torch.set_printoptions(profile='full')
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from utils import *
from data_loader import load_data
from torch_geometric.loader import DataLoader
from model import GAD_Explainer
from torch_geometric.nn.models.autoencoder import GAE

def run_split(args,graphs,fold_idx,n_fold,save_fold,best_gn_auc):
    train_idx, test_idx = fold_idx[0], fold_idx[1]
    if args.dataset == 'mnist':
        train_graphs = [graphs[0][i] for i in train_idx]
        test_graphs = [graphs[1][i] for i in test_idx]
    else:
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

    input_dim = train_graphs[0].num_node_features
    batch_train = DataLoader(train_graphs,batch_size=args.batch_size,shuffle=True)
    batch_test = DataLoader(test_graphs,batch_size=args.batch_size_test,shuffle=False)

    model = GAE(GAD_Explainer(args=args, input_dim=input_dim)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def train(batch_graphs,batch_num):
        optimizer.zero_grad()
        KL_loss,NCE_loss,similarity,_,_,_,embs,node_embs = model.encode(batch_graphs,batch_num)
        GAE_loss = model.recon_loss(node_embs, batch_graphs.edge_index)
        regular_loss = ( torch.norm(torch.sum(similarity,axis=0)) /similarity.shape[0] * np.sqrt(similarity.shape[1]) - 1 ) / (np.sqrt(similarity.shape[1])-1)
        loss =  args.kl * KL_loss + args.nce * NCE_loss +  args.regular * regular_loss + args.gae * GAE_loss
        loss.backward()
        optimizer.step()

        return loss, similarity, embs

    def test(batch_graphs):
        KL_loss,NCE_loss,similarity,node_bern,edge_bern,data_sim,embs,node_embs = model.encode(batch_graphs)
        logits = - torch.max(similarity, dim=-1).values

        return logits, similarity, node_bern, edge_bern, data_sim

    performance_auc_fold = []
    node_auc_fold = []
    global_node_auc_fold = []
    edge_auc_fold = []
    global_edge_auc_fold = []

    model.encoder.init_prototypes(batch_train)
    local_optim = float("inf")
    for epoch in range(1, args.epochs+1):

        model.train()
        epoch_loss = 0
        epoch_time = 0

        i = 0
        embs_lst = []
        for batch_graphs in batch_train:
            start_time = time.time()
            batch_graphs = batch_graphs.to(args.device)
            loss,sim,embs = train(batch_graphs,i)
            end_time = time.time()
            epoch_time += end_time - start_time
            epoch_loss += loss.item()
            i += 1
            embs_lst.append(embs)

        model.eval()
        i = 0
        for batch_graphs in batch_test:
            batch_graphs = batch_graphs.to(args.device)
            logits, similarity, node_bern, edge_bern, data_sim = test(batch_graphs)
            if i == 0:
                logits_ = logits
                labels_ = batch_graphs.y
            else:
                logits_ = torch.cat((logits_, logits), dim=0)
                labels_ = torch.cat((labels_, batch_graphs.y), dim=0)
            i += 1

        auc = compute_metrics(logits_, labels_)
        print(f'Fold_idx:{n_fold + 1}, Epoch: {epoch}, auc: {auc}, loss: {epoch_loss / len(batch_train)}, time: {epoch_time}s')

        performance_auc_fold.append(auc)

        if args.explain:
            if args.dataset == 'mutagen':
                batch_explain = DataLoader(graphs[fold_idx[2]], batch_size=9999, shuffle=False) # fold_idx[2] == explain_idx (only cotain normal graphs)
            else:
                batch_explain = DataLoader(test_graphs, batch_size=9999, shuffle=False)

            for batch_graphs in batch_explain:
                batch_graphs = batch_graphs.to(args.device)
                logits, sim_matrix, node_score, edge_score, data_sim = test(batch_graphs)
                anomaly_prototype = torch.mean(data_sim, dim=-1) - torch.mean(sim_matrix,dim=-1)  # finding big values as typical anomaly
                node_true = batch_graphs.node_label
                edge_true = batch_graphs.edge_label

            k = 5
            if args.dataset == 'mutagen':
                k = 3
                values, indices = sim_matrix.topk(k, dim=0, largest=True)
                select_graphs = indices.flatten()
            else:
                values, indices = sim_matrix.topk(k, dim=0, largest=True) # each prototype have K explanations
                select_normal = indices.flatten()
                values, indices = anomaly_prototype.topk(k*args.n_prot, dim=0, largest=True)
                select_anomaly = indices.flatten()
                select_graphs = torch.cat((select_normal,select_anomaly),dim=0)

            global_node = torch.isin(batch_graphs.batch, select_graphs)
            global_node_true = batch_graphs.node_label[global_node]
            global_node_score = node_score[global_node]

            global_node_idx = torch.nonzero(global_node).squeeze()
            global_edge = torch.isin(batch_graphs.edge_index[0], global_node_idx)
            global_edge_true = batch_graphs.edge_label[global_edge]
            global_edge_score = edge_score[global_edge]

            node_auc = roc_auc_score(node_true.detach().cpu().numpy(), node_score.detach().cpu().numpy())
            edge_auc = roc_auc_score(edge_true.detach().cpu().numpy(), edge_score.detach().cpu().numpy())
            global_node_auc = roc_auc_score(global_node_true.detach().cpu().numpy(), global_node_score.detach().cpu().numpy())
            global_edge_auc = roc_auc_score(global_edge_true.detach().cpu().numpy(), global_edge_score.detach().cpu().numpy())
            node_auc_fold.append(node_auc)
            edge_auc_fold.append(edge_auc)
            global_node_auc_fold.append(global_node_auc)
            global_edge_auc_fold.append(global_edge_auc)

        if epoch > 50 and (not args.nofilter):
            if epoch_loss <= local_optim:
                count = 0
                local_optim = epoch_loss
            else:
                count += 1
                if count >= 5:
                    count = 0
                    local_optim = float("inf")
                    # filter prototypes
                    train_embs = torch.cat(embs_lst)
                    model.encoder.del_proto(train_embs,len(train_idx))
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f'Final Prototype Number is {model.encoder.prototypes.shape[0]}')
    return performance_auc_fold,node_auc_fold,edge_auc_fold,global_node_auc_fold,global_edge_auc_fold,save_fold,best_gn_auc

if __name__ == '__main__':
    args = load_args()
    if args.dataset in ['mnist','BA-TYPE','mutagen','BA-COUNT','BA-SIZE']:
        args.explain = True

    performance_auc = []
    node_auc = []
    global_node_auc = []
    edge_auc = []
    global_edge_auc = []
    save_fold, best_gn_auc = 0, 0
    for split in range(args.n_split):
        graphs, split_idx = load_data(args,split)
        auc_fold,node_auc_fold,edge_auc_fold,global_node_auc_fold,global_edge_auc_fold,save_fold,best_gn_auc = run_split(args,graphs,split_idx,split,save_fold,best_gn_auc)
        performance_auc.append(auc_fold)
        if args.explain:
            node_auc.append(node_auc_fold)
            edge_auc.append(edge_auc_fold)
            global_node_auc.append(global_node_auc_fold)
            global_edge_auc.append(global_edge_auc_fold)

    print(args)
    if args.explain:
        node_auc_ = np.array(node_auc)
        edge_auc_ = np.array(edge_auc)
        global_node_auc_ = np.array(global_node_auc)
        global_edge_auc_ = np.array(global_edge_auc)

        node_auc_mean = np.mean(node_auc_,axis=0)
        edge_auc_mean = np.mean(edge_auc_, axis=0)
        gnode_auc_mean = np.mean(global_node_auc_,axis=0)
        gedge_auc_mean = np.mean(global_edge_auc_, axis=0)
        # __________________________________________________
        idx = np.argmax(gnode_auc_mean)
        best_node_auc = node_auc_mean[idx]
        best_edge_auc = edge_auc_mean[idx]
        node_auc_std = np.std(node_auc_[:, idx])
        edge_auc_std = np.std(edge_auc_[:, idx])

        best_gnode_auc = gnode_auc_mean[idx]
        best_gedge_auc = gedge_auc_mean[idx]
        gnode_auc_std = np.std(global_node_auc_[:, idx])
        gedge_auc_std = np.std(global_edge_auc_[:, idx])

        print(f'dataset:{args.dataset},save_fold:{save_fold}')
        print(f'**under the situation of best node-auc, the best_idx:{idx}')
        print('node auc:%.4f +- %.4f' % (best_node_auc, node_auc_std))
        print('edge auc:%.4f +- %.4f' % (best_edge_auc, edge_auc_std))
        print('global node auc:%.4f +- %.4f' % (best_gnode_auc, gnode_auc_std))
        print('global edge auc:%.4f +- %.4f' % (best_gedge_auc, gedge_auc_std))

    auc_ = np.array(performance_auc)

    auc_mean = np.mean(auc_, axis=0)
    # __________________________________________________
    idx = np.argmax(auc_mean)
    best_auc_mean = auc_mean[idx]
    auc_std = np.std(auc_[:, idx])

    print(f'dataset:{args.dataset}')
    print(f'**under the situation of best AUC, the best_idx:{idx}')
    print('auc:%.4f +- %.4f' % (best_auc_mean, auc_std))

    best_fold = np.argmax(auc_[:,idx])
