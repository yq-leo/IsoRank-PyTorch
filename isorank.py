import torch
import torch.nn.functional as F


def isorank(adj1, adj2, anchors, alpha, max_iter):
    n1, n2 = adj1.shape[0], adj2.shape[0]
    H = torch.zeros(n1, n2).to(torch.float32)
    H[anchors[:, 0], anchors[:, 1]] = 1

    adj1 = F.normalize(adj1, p=1, dim=0)
    adj2 = F.normalize(adj2, p=1, dim=0)

    S = torch.zeros(n1, n2).to(torch.float32)
    for i in range(max_iter):
        S = alpha * adj1 @ S @ adj2.T + (1 - alpha) * H
    return S
