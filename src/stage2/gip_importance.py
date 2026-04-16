"""
gip_importance.py  —  Candidate A, Task 3
==========================================

Quantifies GIP edge contribution specifically for low-degree circRNA nodes.

Steps:
  1. Compute biological degree of each circRNA node (counting only
     circRNA–disease edges, not GIP self-loops).
  2. Bin circRNA nodes into degree quartiles (Q1 = fewest connections).
  3. Train two models per fold: WITH GIP edges vs WITHOUT GIP edges.
  4. Compute per-node AUC per quartile and compare.
  5. Write gip_importance.txt report.

Run from src/stage2/:
    python gip_importance.py
"""

import sys
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "graphs"
DATA_DIR  = ROOT / "data" / "data_cleaned"
RESULTS   = ROOT / "results" / "interpretability"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
FOLDS     = [0, 1, 2, 3, 4]
EPOCHS    = 60
LR        = 1e-3
SEED      = 42

print(f"Device : {DEVICE}")
print(f"Results: {RESULTS}")


# ── Model (same as relation_ablation.py) ──────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden=64, out=64, dropout=0.2):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden)
        self.conv2   = SAGEConv(hidden, hidden)
        self.conv3   = SAGEConv(hidden, hidden)
        self.conv4   = SAGEConv(hidden, out)
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h1  = self.dropout(self.act(self.conv1(x, edge_index)))
        h2  = self.dropout(self.act(self.conv2(h1, edge_index))) + h1
        h3  = self.dropout(self.act(self.conv3(h2, edge_index))) + h2
        return F.normalize(self.conv4(h3, edge_index), p=2, dim=1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remove_gip_edges(edge_index: torch.Tensor, n_circ: int) -> torch.Tensor:
    """Keep only circRNA–disease edges (remove GIP circ–circ and dis–dis)."""
    src, dst = edge_index[0], edge_index[1]
    circ_dis = (src < n_circ) & (dst >= n_circ)
    dis_circ = (src >= n_circ) & (dst < n_circ)
    keep     = circ_dis | dis_circ
    return edge_index[:, keep]


def load_node_maps():
    circ_df  = pd.read_csv(DATA_DIR / "circRNA_nodes_clean.csv")
    dis_df   = pd.read_csv(DATA_DIR / "disease_nodes_clean.csv")
    circ_map = {str(v): i for i, v in enumerate(circ_df["circRNA"])}
    dis_map  = {str(v): i for i, v in enumerate(dis_df["disease"])}
    return circ_map, dis_map


def load_fold_tensors(fold: int, circ_map: dict, dis_map: dict):
    train_df = pd.read_csv(DATA_DIR / f"circRNA_disease_fold{fold}_train.csv")
    test_df  = pd.read_csv(DATA_DIR / f"circRNA_disease_fold{fold}_test.csv")

    def to_tensors(df):
        c = torch.tensor([circ_map[str(x)] for x in df["circRNA"]], dtype=torch.long)
        d = torch.tensor([dis_map[str(x)]  for x in df["disease"]],  dtype=torch.long)
        l = torch.tensor(df["label"].values, dtype=torch.float)
        return torch.stack([c, d]).to(DEVICE), l.to(DEVICE)

    return to_tensors(train_df), to_tensors(test_df), test_df


def train_model(graph_x, edge_index, train_edges, train_labels,
                test_edges, test_labels, n_circ: int) -> GraphSAGEEncoder:
    set_seed(SEED)
    model     = GraphSAGEEncoder(graph_x.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        emb    = model(graph_x, edge_index)
        logits = (emb[:n_circ][train_edges[0]] * emb[n_circ:][train_edges[1]]).sum(1)
        loss_fn(logits, train_labels).backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                emb  = model(graph_x, edge_index)
                sc   = torch.sigmoid(
                    (emb[:n_circ][test_edges[0]] * emb[n_circ:][test_edges[1]]).sum(1)
                ).cpu().numpy()
                auc = roc_auc_score(test_labels.cpu().numpy(), sc)
            print(f"      epoch {epoch:03d}  test_auc={auc:.4f}")
            if auc > best_auc:
                best_auc   = auc
                best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
    return model


def get_scores(model, graph_x, edge_index, edges, n_circ):
    model.eval()
    with torch.no_grad():
        emb    = model(graph_x, edge_index)
        logits = (emb[:n_circ][edges[0]] * emb[n_circ:][edges[1]]).sum(1)
        return torch.sigmoid(logits).cpu().numpy()


def per_node_auc(circ_indices, scores, labels):
    """
    Compute per-circRNA-node AUC.
    Returns dict: circ_idx -> auc (only nodes with both label classes present).
    """
    node_data = defaultdict(lambda: {"s": [], "l": []})
    for i, c in enumerate(circ_indices):
        node_data[int(c)]["s"].append(float(scores[i]))
        node_data[int(c)]["l"].append(float(labels[i]))

    result = {}
    for idx, d in node_data.items():
        y = np.array(d["l"])
        if len(np.unique(y)) < 2:
            continue
        result[idx] = roc_auc_score(y, np.array(d["s"]))
    return result


def circRNA_bio_degree(graph, n_circ: int) -> np.ndarray:
    """
    Count only circRNA–disease biological edges per circRNA node
    (exclude GIP self-similarity edges).
    """
    degree = np.zeros(n_circ, dtype=np.int64)
    src, dst = graph.edge_index[0].cpu().numpy(), graph.edge_index[1].cpu().numpy()
    mask = (src < n_circ) & (dst >= n_circ)
    np.add.at(degree, src[mask], 1)
    return degree


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    circ_map, dis_map = load_node_maps()
    n_circ = len(circ_map)
    n_dis  = len(dis_map)

    Q_LABELS = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]

    # Accumulators across folds
    delta_by_quartile = defaultdict(list)  # quartile_label -> list of per-node delta AUC
    overall_gip    = []
    overall_no_gip = []

    for fold in FOLDS:
        print(f"\n{'='*56}")
        print(f"Fold {fold}")
        print(f"{'='*56}")

        graph = torch.load(
            GRAPH_DIR / f"gcd_graph_fold{fold}.pt",
            map_location=DEVICE, weights_only=False,
        )
        x          = graph.x.to(DEVICE)
        edge_full  = graph.edge_index.to(DEVICE)
        edge_nogip = remove_gip_edges(edge_full, n_circ).to(DEVICE)

        print(f"  Edges with GIP   : {edge_full.shape[1]}")
        print(f"  Edges without GIP: {edge_nogip.shape[1]}")

        # Degree quartiles from this fold's graph
        degree    = circRNA_bio_degree(graph, n_circ)
        quartiles = np.percentile(degree, [25, 50, 75])

        def quartile_label(d):
            if d <= quartiles[0]: return "Q1 (low)"
            if d <= quartiles[1]: return "Q2"
            if d <= quartiles[2]: return "Q3"
            return "Q4 (high)"

        circ_q = {i: quartile_label(degree[i]) for i in range(n_circ)}

        (train_edges, train_labels), (test_edges, test_labels), test_df = \
            load_fold_tensors(fold, circ_map, dis_map)

        # Train WITH GIP
        print("\n  Training WITH GIP edges...")
        model_gip = train_model(x, edge_full, train_edges, train_labels,
                                test_edges, test_labels, n_circ)

        # Train WITHOUT GIP
        print("\n  Training WITHOUT GIP edges...")
        model_no_gip = train_model(x, edge_nogip, train_edges, train_labels,
                                   test_edges, test_labels, n_circ)

        # Overall metrics
        sc_gip    = get_scores(model_gip,    x, edge_full,  test_edges, n_circ)
        sc_no_gip = get_scores(model_no_gip, x, edge_nogip, test_edges, n_circ)
        y_true    = test_labels.cpu().numpy()

        auc_gip    = roc_auc_score(y_true, sc_gip)
        auc_no_gip = roc_auc_score(y_true, sc_no_gip)
        overall_gip.append(auc_gip)
        overall_no_gip.append(auc_no_gip)
        print(f"\n  Overall  with GIP: AUC={auc_gip:.4f}")
        print(f"  Overall w/o  GIP: AUC={auc_no_gip:.4f}")

        # Per-node AUC
        circ_indices = test_edges[0].cpu().numpy()
        auc_gip_node    = per_node_auc(circ_indices, sc_gip,    y_true)
        auc_no_gip_node = per_node_auc(circ_indices, sc_no_gip, y_true)

        common_nodes = set(auc_gip_node) & set(auc_no_gip_node)
        for idx in common_nodes:
            delta = auc_gip_node[idx] - auc_no_gip_node[idx]
            ql    = circ_q.get(idx, "Q4 (high)")
            delta_by_quartile[ql].append(delta)

    # ── Aggregate across folds ─────────────────────────────────────────────────
    print("\n\n" + "=" * 56)
    print("AGGREGATED RESULTS")
    print("=" * 56)

    mean_gip    = np.mean(overall_gip)
    mean_no_gip = np.mean(overall_no_gip)
    print(f"\nMean overall AUC with GIP    : {mean_gip:.4f}")
    print(f"Mean overall AUC without GIP : {mean_no_gip:.4f}")
    print(f"Overall ΔAUC                 : {mean_gip - mean_no_gip:+.4f}")

    quartile_rows = []
    for ql in Q_LABELS:
        deltas = delta_by_quartile.get(ql, [])
        row = {
            "quartile"       : ql,
            "n_nodes"        : len(deltas),
            "mean_delta_auc" : float(np.mean(deltas)) if deltas else float("nan"),
            "std_delta_auc"  : float(np.std(deltas))  if deltas else float("nan"),
        }
        quartile_rows.append(row)
        print(f"  {ql:12s}  n={len(deltas):4d}  "
              f"mean_ΔAUC={row['mean_delta_auc']:+.4f}  "
              f"std={row['std_delta_auc']:.4f}")

    q_df = pd.DataFrame(quartile_rows)
    q_df.to_csv(RESULTS / "gip_quartile_auc.csv", index=False)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos  = np.arange(len(Q_LABELS))
    vals   = [q_df[q_df.quartile == ql]["mean_delta_auc"].values[0] for ql in Q_LABELS]
    stds   = [q_df[q_df.quartile == ql]["std_delta_auc"].values[0]  for ql in Q_LABELS]
    colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in vals]

    bars = ax.bar(x_pos, vals, color=colors, alpha=0.85, yerr=stds,
                  capsize=5, error_kw={"elinewidth": 1.2})
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(Q_LABELS, fontsize=10)
    ax.set_ylabel("Mean ΔAUC per node (with GIP − without GIP)")
    ax.set_title("GIP edge contribution by circRNA degree quartile\n"
                 "(Q1 = fewest biological connections)")
    ax.spines[["top","right"]].set_visible(False)

    for bar, val, std in zip(bars, vals, stds):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (std + 0.005 if val >= 0 else -(std + 0.008)),
                    f"{val:+.4f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=9)

    plt.tight_layout()
    fig.savefig(RESULTS / "gip_quartile_auc.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved → {RESULTS / 'gip_quartile_auc.png'}")

    # ── Text report ────────────────────────────────────────────────────────────
    q1_delta = q_df[q_df.quartile == "Q1 (low)"]["mean_delta_auc"].values[0]
    q4_delta = q_df[q_df.quartile == "Q4 (high)"]["mean_delta_auc"].values[0]

    if not np.isnan(q1_delta) and not np.isnan(q4_delta):
        if q1_delta > q4_delta:
            observation = (
                f"GIP edges benefit LOW-degree circRNA nodes most.\n"
                f"  Q1 ΔAUC = {q1_delta:+.4f}  vs  Q4 ΔAUC = {q4_delta:+.4f}\n"
                f"  This confirms GIP similarity edges compensate for sparse\n"
                f"  biological interaction data in low-connectivity circRNAs."
            )
        else:
            observation = (
                f"GIP edges do NOT preferentially benefit low-degree nodes.\n"
                f"  Q1 ΔAUC = {q1_delta:+.4f}  vs  Q4 ΔAUC = {q4_delta:+.4f}\n"
                f"  The GIP signal may be redundant with existing graph structure\n"
                f"  for sparsely-connected circRNAs in this dataset."
            )
    else:
        observation = "Insufficient nodes with both label classes to compare quartiles."

    report = "\n".join([
        "=" * 62,
        "GIP EDGE IMPORTANCE — ANALYSIS REPORT",
        "=" * 62,
        "",
        "Methodology note",
        "  Interpretability is performed on the shared GraphSAGE encoder",
        "  backbone using a dot-product link predictor — not the full weighted",
        "  fusion model. This is justified because: (1) the encoder is the",
        "  component that learns graph structure from GIP edges; (2) the fusion",
        "  predictor combines pre-computed embeddings and cannot be ablated at",
        "  the edge level. Results capture encoder-level GIP contribution.",
        "",
        "Overall performance (mean across 5 folds)",
        f"  With GIP    : AUC = {mean_gip:.4f}",
        f"  Without GIP : AUC = {mean_no_gip:.4f}",
        f"  ΔAUC        : {mean_gip - mean_no_gip:+.4f}",
        "",
        "Per-quartile delta AUC (with GIP − without GIP)",
        q_df.to_string(index=False),
        "",
        "Key observation",
        observation,
        "",
        f"Outputs saved to: {RESULTS}",
        "=" * 62,
    ])

    print("\n" + report)
    with open(RESULTS / "gip_importance.txt", "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {RESULTS / 'gip_importance.txt'}")


if __name__ == "__main__":
    main()