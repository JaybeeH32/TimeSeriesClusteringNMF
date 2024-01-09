from preprocess import load_ts_dataset, distance_matrix, epsilon_graph_hard, epsilon_graph_mean
import scipy.sparse as sp
import argparse
import numpy as np
import networkx as nx
import pandas as pd

def main(args):
    print(args.dataset != 'ptb')
    if args.dataset != 'ptb':
        dataset = load_ts_dataset(args.dataset)
        data = dataset[0]
    else:
        MAX_LEN = 100
        data_normal = pd.read_csv('ptbdb_normal.csv').iloc[:MAX_LEN].to_numpy()
        data_abnormal = pd.read_csv('ptbdb_abnormal.csv').iloc[:MAX_LEN].to_numpy()
        data = np.concatenate([data_normal, data_abnormal], axis=0)
    
    dist = distance_matrix(data)
    dist = dist / np.max(dist)
    A = epsilon_graph_hard(dist, epsilon=float(args.eps))
    # A = epsilon_graph_mean(dist)
    print(A)
    G = nx.from_numpy_array(A)
    npz_file_path = "adjacency\." + args.dataset + ';eps=' + args.eps +'.npz'

    print(npz_file_path)
    sp.save_npz(npz_file_path, nx.adjacency_matrix(G))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--eps', default=0.16677)

    args = parser.parse_args()
    main(args)