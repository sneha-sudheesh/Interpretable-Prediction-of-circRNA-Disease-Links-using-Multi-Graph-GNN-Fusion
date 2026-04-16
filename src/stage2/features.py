import pandas as pd
import networkx as nx
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "data_cleaned"

def normalize_node(node):

    node = str(node).strip().lower()

    return node

def compute_node_features():

    cd = pd.read_csv(DATA / "circRNA_disease_edges.csv")
    cm = pd.read_csv(DATA / "circRNA_miRNA_edges.csv")
    md = pd.read_csv(DATA / "miRNA_disease_edges.csv")

    G = nx.Graph()

    # circRNA–disease
    for _, r in cd.iterrows():
        G.add_edge(r["circRNA"], r["disease"])

    # circRNA–miRNA
    for _, r in cm.iterrows():
        G.add_edge(r["circRNA"], r["miRNA"])

    # miRNA–disease
    for _, r in md.iterrows():
        G.add_edge(r["miRNA"], r["disease"])

    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)

    features = {}

    for node in G.nodes():

        key = normalize_node(node)
    
        deg = degree[node]
        logdeg = torch.log(torch.tensor(deg + 1.0)).item()
        btw = betweenness[node]
    
        if "circ" in key:
            onehot = [1,0,0]
        elif "mir" in key:
            onehot = [0,1,0]
        else:
            onehot = [0,0,1]
    
        features[key] = [deg, logdeg, btw] + onehot

    return features