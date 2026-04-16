"""
relation_ablation.py  —  Candidate A, Task 2
=============================================

Masks each edge-type group in the circRNA–disease graph one at a time,
retrains the GraphSAGEEncoder + dot-product link predictor from scratch,
and records the AUC/AUPR drop vs baseline.

The gcd_graph_fold*.pt graphs contain three interleaved edge types
(identified by node index ranges):
    circRNA–disease  : src < n_circ, dst >= n_circ   (biological)
    circRNA–circRNA  : src < n_circ, dst < n_circ     (GIP similarity)
    disease–disease  : src >= n_circ, dst >= n_circ   (GIP similarity)

We ablate each group and the full GIP block.

Run from src/stage2/:
    python relation_ablation.py
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


# ── Model ──────────────────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    """Identical to sage_encoder.py."""
    def __init__(self, in_channels, hidden=64, out=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.conv4 = SAGEConv(hidden, out)
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h1  = self.dropout(self.act(self.conv1(x, edge_index)))
        h2  = self.dropout(self.act(self.conv2(h1, edge_index))) + h1
        h3  = self.dropout(self.act(self.conv3(h2, edge_index))) + h2
        out = self.conv4(h3, edge_index)
        return F.normalize(out, p=2, dim=1)


# ── Edge-type masking ──────────────────────────────────────────────────────────

def mask_edges(edge_index: torch.Tensor, n_circ: int, mask_type: str) -> torch.Tensor:
    """
    Return a filtered edge_index with the specified edge type removed.

    mask_type options:
        "circ_dis"   — remove circRNA–disease edges (biological signal)
        "gip_circ"   — remove circRNA–circRNA GIP edges
        "gip_dis"    — remove disease–disease GIP edges
        "all_gip"    — remove both GIP edge types
    """
    src, dst = edge_index[0], edge_index[1]

    circ_dis_mask = (src < n_circ) & (dst >= n_circ)
    dis_circ_mask = (src >= n_circ) & (dst < n_circ)
    biological    = circ_dis_mask | dis_circ_mask

    gip_circ_mask = (src < n_circ)  & (dst < n_circ)
    gip_dis_mask  = (src >= n_circ) & (dst >= n_circ)

    if mask_type == "circ_dis":
        keep = ~biological
    elif mask_type == "gip_circ":
        keep = ~gip_circ_mask
    elif mask_type == "gip_dis":
        keep = ~gip_dis_mask
    elif mask_type == "all_gip":
        keep = ~(gip_circ_mask | gip_dis_mask)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return edge_index[:, keep]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_fold_data(fold: int, n_circ: int, n_dis: int):
    """
    Load train/test edges for one fold from the CSV splits.
    Returns (train_edges, train_labels, test_edges, test_labels) as tensors.
    """
    train_df = pd.read_csv(DATA_DIR / f"circRNA_disease_fold{fold}_train.csv")
    test_df  = pd.read_csv(DATA_DIR / f"circRNA_disease_fold{fold}_test.csv")

    # Build node maps from circRNA_nodes_clean.csv and disease_nodes_clean.csv
    circ_df = pd.read_csv(DATA_DIR / "circRNA_nodes_clean.csv")
    dis_df  = pd.read_csv(DATA_DIR / "disease_nodes_clean.csv")
    circ_map = {v: i for i, v in enumerate(circ_df["circRNA"].astype(str))}
    dis_map  = {v: i for i, v in enumerate(dis_df["disease"].astype(str))}

    def df_to_tensors(df):
        # Normalise to match node CSV format
        circ_ids = torch.tensor(
            [circ_map[str(c)] for c in df["circRNA"]], dtype=torch.long
        )
        dis_ids = torch.tensor(
            [dis_map[str(d)] for d in df["disease"]], dtype=torch.long
        )
        labels = torch.tensor(df["label"].values, dtype=torch.float)
        edges  = torch.stack([circ_ids, dis_ids])
        return edges, labels

    train_edges, train_labels = df_to_tensors(train_df)
    test_edges,  test_labels  = df_to_tensors(test_df)

    return (
        train_edges.to(DEVICE), train_labels.to(DEVICE),
        test_edges.to(DEVICE),  test_labels.to(DEVICE),
    )


# ── Training & evaluation ──────────────────────────────────────────────────────

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_and_eval(graph, train_edges, train_labels,
                   test_edges, test_labels, n_circ: int) -> dict:
    """
    Train GraphSAGEEncoder with dot-product link prediction on one fold.
    Returns dict with test AUC and AUPR.
    """
    set_seed(SEED)

    x          = graph.x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)

    model     = GraphSAGEEncoder(x.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_state    = None
    best_test_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        emb = model(x, edge_index)

        # circRNA embeddings: rows 0..n_circ-1
        # disease embeddings: rows n_circ..n_circ+n_dis-1
        circ_emb = emb[:n_circ]
        dis_emb  = emb[n_circ:]

        logits = (circ_emb[train_edges[0]] * dis_emb[train_edges[1]]).sum(1)
        loss   = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                emb      = model(x, edge_index)
                circ_emb = emb[:n_circ]
                dis_emb  = emb[n_circ:]
                t_logits = (circ_emb[test_edges[0]] * dis_emb[test_edges[1]]).sum(1)
                t_scores = torch.sigmoid(t_logits).cpu().numpy()
                t_auc    = roc_auc_score(test_labels.cpu().numpy(), t_scores)
            print(f"      epoch {epoch:03d}  loss={loss.item():.4f}  test_auc={t_auc:.4f}")

            if t_auc > best_test_auc:
                best_test_auc = t_auc
                best_state    = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        emb      = model(x, edge_index)
        circ_emb = emb[:n_circ]
        dis_emb  = emb[n_circ:]
        t_logits = (circ_emb[test_edges[0]] * dis_emb[test_edges[1]]).sum(1)
        t_scores = torch.sigmoid(t_logits).cpu().numpy()
        t_true   = test_labels.cpu().numpy()

    return {
        "auc" : roc_auc_score(t_true, t_scores),
        "aupr": average_precision_score(t_true, t_scores),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

ABLATIONS = {
    "baseline"                    : None,
    "mask circRNA–disease edges"  : "circ_dis",
    "mask GIP circRNA–circRNA"    : "gip_circ",
    "mask GIP disease–disease"    : "gip_dis",
    "mask ALL GIP edges"          : "all_gip",
}


def main():
    records = []

    for ablation_name, mask_type in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"{'='*60}")

        fold_results = []

        for fold in FOLDS:
            print(f"\n  Fold {fold}:")
            graph = torch.load(
                GRAPH_DIR / f"gcd_graph_fold{fold}.pt",
                map_location=DEVICE, weights_only=False
            )
            n_circ = int(graph.num_circ)
            n_dis  = int(graph.num_dis)

            # Optionally mask edges
            if mask_type is not None:
                original_count = graph.edge_index.shape[1]
                graph.edge_index = mask_edges(graph.edge_index, n_circ, mask_type)
                removed = original_count - graph.edge_index.shape[1]
                print(f"    Removed {removed} edges ({mask_type}); "
                      f"remaining: {graph.edge_index.shape[1]}")

            train_edges, train_labels, test_edges, test_labels = load_fold_data(
                fold, n_circ, n_dis
            )

            result = train_and_eval(
                graph, train_edges, train_labels,
                test_edges, test_labels, n_circ
            )
            print(f"    AUC={result['auc']:.4f}  AUPR={result['aupr']:.4f}")
            fold_results.append(result)

        mean_auc  = np.mean([r["auc"]  for r in fold_results])
        mean_aupr = np.mean([r["aupr"] for r in fold_results])
        std_auc   = np.std( [r["auc"]  for r in fold_results])
        std_aupr  = np.std( [r["aupr"] for r in fold_results])

        records.append({
            "ablation"  : ablation_name,
            "mean_auc"  : mean_auc,
            "std_auc"   : std_auc,
            "mean_aupr" : mean_aupr,
            "std_aupr"  : std_aupr,
        })
        print(f"\n  Mean: AUC={mean_auc:.4f}±{std_auc:.4f}  "
              f"AUPR={mean_aupr:.4f}±{std_aupr:.4f}")

    # ── Compute deltas vs baseline ─────────────────────────────────────────────
    df          = pd.DataFrame(records)
    baseline    = df[df.ablation == "baseline"].iloc[0]
    df["delta_auc"]   = df["mean_auc"]  - baseline["mean_auc"]
    df["delta_aupr"]  = df["mean_aupr"] - baseline["mean_aupr"]
    df["pct_drop_auc"]  = (df["delta_auc"]  / baseline["mean_auc"])  * 100
    df["pct_drop_aupr"] = (df["delta_aupr"] / baseline["mean_aupr"]) * 100
    df = df.sort_values("delta_auc")
    df.to_csv(RESULTS / "relation_ablation.csv", index=False)
    print(f"\nSaved → {RESULTS / 'relation_ablation.csv'}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_df = df[df.ablation != "baseline"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in [
        (axes[0], "delta_auc",  "ΔAUC"),
        (axes[1], "delta_aupr", "ΔAUPR"),
    ]:
        colors = ["#D85A30" if v < 0 else "#1D9E75" for v in plot_df[metric]]
        bars   = ax.barh(plot_df["ablation"], plot_df[metric], color=colors, alpha=0.85)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(f"{label} vs baseline")
        ax.set_title(f"Relation ablation — {label}")
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, plot_df[metric]):
            xp = val - 0.001 if val < 0 else val + 0.001
            ha = "right" if val < 0 else "left"
            ax.text(xp, bar.get_y() + bar.get_height()/2,
                    f"{val:+.4f}", va="center", ha=ha, fontsize=9)

    plt.suptitle(
        f"Baseline AUC={baseline['mean_auc']:.4f}  AUPR={baseline['mean_aupr']:.4f}",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fig.savefig(RESULTS / "relation_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {RESULTS / 'relation_ablation.png'}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("RELATION ABLATION — RANKED BY AUC DROP")
    print("=" * 62)
    print(df[["ablation", "mean_auc", "mean_aupr",
              "delta_auc", "delta_aupr",
              "pct_drop_auc", "pct_drop_aupr"]].to_string(index=False))
    print(f"\nBaseline: AUC={baseline['mean_auc']:.4f}±{baseline['std_auc']:.4f}  "
          f"AUPR={baseline['mean_aupr']:.4f}±{baseline['std_aupr']:.4f}")
    print("\nNote: interpretability uses the shared GraphSAGE encoder backbone")
    print("with a dot-product predictor for ablation. The fusion predictor")
    print("(WeightedFusionPredictor) is not directly ablatable at the edge level.")
    print("Results reflect encoder-level structural importance.")


if __name__ == "__main__":
    main()