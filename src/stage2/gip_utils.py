import torch

def compute_gip_edges(cd_src, cd_dst, num_circ, num_dis, k=10):

    # Build adjacency matrix
    A = torch.zeros((num_circ, num_dis))
    A[cd_src, cd_dst] = 1

    # -------- circRNA GIP --------
    gamma = 1 / A.shape[1]

    K_circ = torch.exp(-gamma * torch.cdist(A, A) ** 2)

    K_circ.fill_diagonal_(0)

    vals_circ, idx_circ = torch.topk(K_circ, k=k, dim=1)

    src_circ = torch.arange(num_circ).repeat_interleave(k)
    dst_circ = idx_circ.reshape(-1)

    circ_edge = torch.stack([src_circ, dst_circ])
    circ_weight = vals_circ.reshape(-1)

    # -------- disease GIP --------
    A_T = A.T

    gamma_dis = 1 / A_T.shape[1]

    K_dis = torch.exp(-gamma_dis * torch.cdist(A_T, A_T) ** 2)

    K_dis.fill_diagonal_(0)

    vals_dis, idx_dis = torch.topk(K_dis, k=k, dim=1)

    src_dis = torch.arange(num_dis).repeat_interleave(k)
    dst_dis = idx_dis.reshape(-1)

    dis_edge = torch.stack([src_dis, dst_dis])
    dis_weight = vals_dis.reshape(-1)

    return circ_edge, circ_weight, dis_edge, dis_weight