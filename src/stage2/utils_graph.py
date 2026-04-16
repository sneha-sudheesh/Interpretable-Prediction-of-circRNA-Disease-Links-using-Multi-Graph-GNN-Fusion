import torch
import pandas as pd
from torch_geometric.data import Data


def load_nodes(node_file, id_col):
    df = pd.read_csv(node_file)
    ids = df[id_col].values
    mapping = {id_: i for i, id_ in enumerate(ids)}
    return mapping, len(ids)


def build_edge_index(df, src_map, dst_map, src_col, dst_col):
    src = [src_map[x] for x in df[src_col]]
    dst = [dst_map[x] for x in df[dst_col]]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index