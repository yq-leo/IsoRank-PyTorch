from args import make_args
from utils.data import load_dataset, get_adj_from_edge_index
from utils.metrics import *
from isorank import isorank

import time


if __name__ == '__main__':
    args = make_args()
    assert args.device == 'cpu' or torch.cuda.is_available(), 'CUDA is not available'

    edge_index1, edge_index2, anchor_links, test_pairs = load_dataset(f'datasets/{args.dataset}', 0.2)

    n1, n2 = edge_index1.max() + 1, edge_index2.max() + 1
    adj1 = get_adj_from_edge_index(edge_index1, n1)
    adj2 = get_adj_from_edge_index(edge_index2, n2)

    anchor_links = torch.from_numpy(anchor_links)
    start = time.time()
    similarity = isorank(adj1, adj2, anchor_links, args.alpha, args.max_iter, device=args.device)
    print(f'IsoRank time: {time.time() - start:.4f}s')

    test_pairs = torch.from_numpy(test_pairs)
    hits_ks = hits_ks_ltr_scores(similarity, test_pairs, ks=[1, 10, 30, 50])
    mrr = mrr_ltr_score(similarity, test_pairs)

    print(f'Dataset: {args.dataset}')
    print(f'Hits@1: {hits_ks[1]:.4f}')
    print(f'Hits@10: {hits_ks[10]:.4f}')
    print(f'Hits@30: {hits_ks[30]:.4f}')
    print(f'Hits@50: {hits_ks[50]:.4f}')
    print(f'MRR: {mrr:.4f}')
