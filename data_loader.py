import numpy as np
import torch, random, math
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedShuffleSplit
from data import Mutagen, BA  #MNIST75sp

def load_data(args,random_state=0):
    randomsplit_data = ['mnist','ENZYMES']
    if args.dataset not in randomsplit_data:
        return load_TU(args,random_state)

    elif args.dataset == 'mnist':
        train = MNIST75sp(root='./data/mnist', mode='train')
        test = MNIST75sp(root='./data/mnist', mode='test')
        num_normal = 3
        random.seed(random_state)
        normal_class = random.sample(range(10), num_normal)
        train_idx, test_idx = get_loaders_mnist(train, test, random_state, normal_class)
        print('num_train:', len(train_idx), 'num_test:', len(test_idx))
        print(f"normal class are {normal_class}")
        for i, graph in enumerate(train):
            train.data.y[i] = 0 if graph.y in normal_class else 1
        for i, graph in enumerate(test):
            test.data.y[i] = 0 if graph.y in normal_class else 1
        return [train,test],[train_idx,test_idx]

    elif args.dataset == 'ENZYMES':
        graphs = TUDataset(root='./data/', name=f'{args.dataset}')
        num_normal = 4
        random.seed(random_state)
        normal_class = random.sample(range(6),num_normal)
        print(f"normal class are {normal_class}")
        for i,graph in enumerate(graphs):
            graphs.data.y[i] = 0 if graph.y in normal_class else 1

    # random split dataset
    np.random.seed(random_state)
    idx = np.arange(len(graphs))
    np.random.shuffle(idx)
    n_test = int(0.1 * len(idx))
    test_idx = idx[:n_test]
    train_idx_lst = idx[n_test:]
    train_idx = [i for i in train_idx_lst if graphs[i].y == 0]  # only retain normal graphs
    print('num_train:',len(train_idx),'num_test:',len(test_idx))

    return graphs,[train_idx,test_idx]

def get_loaders_mnist(train_dataset, test_dataset,random_state, normal_class):
    num_train, num_test_normal, num_test_anomaly = 1000, 400, 100
    if random_state is not None:
        np.random.seed(random_state)

    #print('[INFO] Randomly split dataset!')

    train_idx = np.arange(len(train_dataset))
    np.random.shuffle(train_idx)
    train_idx_lst = []
    for normal_label in normal_class:
        normal_mask_tr = (train_dataset.data.y[train_idx] == normal_label).numpy()
        normal_train_idx = train_idx[normal_mask_tr][:math.ceil(num_train/len(normal_class))]
        train_idx_lst.append(normal_train_idx)
    train_idx = np.concatenate(train_idx_lst)

    test_idx = np.arange(len(test_dataset))
    np.random.shuffle(test_idx)
    test_idx_lst = []
    test_anomaly_lst = []

    for label in range(10):
        if label in normal_class:
            normal_mask_te = (test_dataset.data.y[test_idx] == label).numpy()
            normal_test_idx = test_idx[normal_mask_te][:math.ceil(num_test_normal/len(normal_class))]
            test_idx_lst.append(normal_test_idx)
        else:
            anomaly_mask_te = (test_dataset.data.y[test_idx] == label).numpy()
            anomaly_test_idx = test_idx[anomaly_mask_te]
            test_anomaly_lst.append(anomaly_test_idx)

    test_anomaly_idx = np.concatenate(test_anomaly_lst)
    np.random.shuffle(test_anomaly_idx)  # since anomaly sequence follows class range, thus need shuffle again
    test_idx_lst.append(test_anomaly_idx[:num_test_anomaly])
    test_idx = np.concatenate(test_idx_lst)

    return train_idx, test_idx

def load_TU(args,random_state):
    if args.dataset == 'mutagen':
        graphs = Mutagen(root='./data/mutagen')
    elif args.dataset == 'BA-TYPE':
        graphs = BA(root='./data/BA-TYPE')
    elif args.dataset == 'BA-COUNT':
        graphs = BA(root='./data/BA-COUNT')
    elif args.dataset == 'BA-SIZE':
        graphs = BA(root='./data/BA-SIZE')
    elif args.dataset  in ['IMDB-BINARY', 'REDDIT-BINARY']:
        graphs = TUDataset(root='./data/', name=f'{args.dataset}', transform=(Constant(1, cat=False)))
    else:
        graphs = TUDataset(root='./data/', name=f'{args.dataset}')
        if args.dataset not in ['COX2','DD','BZR','PROTEINS']:
            graphs.data.y = (graphs.data.y == 0).int()  # reverse y == 0 to y = 1 (anomalies); others as 0 (normal)

    label_0 = (graphs.data.y == 0).sum()
    label_1 = (graphs.data.y == 1).sum()
    print('normal:', int(label_0), 'anomaly:', int(label_1))

    skf = StratifiedShuffleSplit(args.n_split, test_size=0.2, train_size=0.8, random_state=args.seed)
    split_idx = []
    for train_index_lst, test_index_lst in skf.split(np.zeros(len(graphs)), graphs.data.y):
        train_index_lst = [i for i in train_index_lst if graphs[i].y == 0]  # only retain normal graphs
        explain_idx_lst = [i for i in test_index_lst if graphs[i].y == 0]
        split_idx.append([train_index_lst, test_index_lst, explain_idx_lst])

    print('num_train:', len(train_index_lst), 'num_test:', len(test_index_lst))

    return graphs,split_idx[random_state]