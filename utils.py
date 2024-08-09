import argparse,torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, classification_report

def load_args():
    parser = argparse.ArgumentParser()

    # main model
    parser.add_argument('--dataset', default='mutagen', help='Dataset name')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=9999, help='Input batch size for training')
    parser.add_argument('--dropout', type=float, default=0.3, help='ratio of dropout')
    parser.add_argument('--r', type=float, default=0.7, help='probability for reserving nodes')  # for KL loss

    # GNN
    parser.add_argument('--hidden_dim', type=int, default=512, help='dim for hidden layers')
    parser.add_argument('--out_dim', type=int, default=256, help='dim for graph embeddings')
    parser.add_argument('--conv_layers', type=int, default=2, help='layers')
    parser.add_argument('--pooling_type', type=str, default='max', choices=["sum", "max", "avg"], help='the type of graph pooling')

    # Prototype
    parser.add_argument('--n_prot', type=int, default=2, help='number of prototypes')

    # Loss
    parser.add_argument('--kl', type=float, default=0.5, help='weight of KL_loss')
    parser.add_argument('--nce', type=float, default=0.5, help='weight of nce_loss')
    parser.add_argument('--gae', type=float, default=1, help='weight of GAE_loss')
    parser.add_argument('--regular', type=float, default=50, help='weight of regular_loss')

    # other set
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--n_split', type=int, default=5, help='cross validation')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use')
    parser.add_argument('--explain', type=bool,default=False, help='using explain metrics')
    parser.add_argument('--nofilter', action='store_true',help='not filtering redundant prototypes')

    args = parser.parse_args()

    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    return args

def compute_metrics(logits, labels):

    logits,labels = logits.detach().cpu().numpy(), labels.detach().cpu().numpy()
    try:
        auc = roc_auc_score(labels, logits)
    except:
        auc = np.float64(0)

    return auc


