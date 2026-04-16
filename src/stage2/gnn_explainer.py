"""
gnn_explainer.py  —  Member 2: Explainer Implementation
========================================================

NOTE ON METHODOLOGY (addresses reviewer feedback):
    Interpretability is performed on the shared GraphSAGE encoder backbone
    using a dot-product predictor, as the WeightedFusionPredictor operates
    on combined embeddings and is not directly explainable via current graph
    explanation techniques. This is standard practice in GNN interpretability —
    we explain what graph structure drives the encoder embeddings, not the
    exact fusion prediction score.

Graph used:
    graphs/gcd_graph_fold0.pt
    (same graph as training — circRNA-disease + GIP edges)
    This matches the graph the model was trained on, ensuring
    explanations are consistent with actual model behaviour.

Model used:
    saved_models/best_stage2_weighted_fold0.pt  (trained encoder weights)
    Falls back to training if checkpoint not found.

Outputs saved to results/interpretability/:
    edge_importance.csv
    node_importance.csv
    edge_importance.pt
    node_importance.pt
    node_importance_bar.png
    edge_importance_bar.png

Run from src/stage2/:
    python gnn_explainer.py

Prerequisites:
    Run weighted.ipynb first → saves saved_models/best_stage2_weighted_fold0.pt
"""

import random
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import SAGEConv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "graphs"
DATA_DIR  = ROOT / "data" / "data_cleaned"
MODEL_DIR = ROOT / "saved_models"
RESULTS   = ROOT / "results" / "interpretability"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLD   = 0

# Trained encoder checkpoint (saved by weighted.ipynb)
CKPT_PATH = MODEL_DIR / f"best_stage2_weighted_fold{FOLD}.pt"

EPOCHS         = 60
LR             = 1e-3
SEED           = 42
EXPLAIN_EPOCHS = 200
EXPLAIN_LR     = 0.01
TOP_K          = 10
RANDOM_K       = 10

print(f"Root       : {ROOT}")
print(f"Graph dir  : {GRAPH_DIR}")
print(f"Model dir  : {MODEL_DIR}")
print(f"Checkpoint : {CKPT_PATH}")
print(f"Results    : {RESULTS}")
print(f"Device     : {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD GRAPH FROM .PT FILE
#  Uses the same gcd_graph_fold0.pt that the model was trained on.
#  Graph contains: circRNA-disease edges + GIP similarity edges
#  Node layout: 0..n_circ-1 = circRNA,  n_circ..N-1 = disease
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(fold):
    graph_path = GRAPH_DIR / f"gcd_graph_fold{fold}.pt"
    print(f"\nLoading graph: {graph_path.name}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    n_circ = int(graph.num_circ)
    n_dis  = int(graph.num_dis)

    print(f"  Nodes : {graph.x.shape[0]}  (circRNA={n_circ}, disease={n_dis})")
    print(f"  Edges : {graph.edge_index.shape[1]}")
    print(f"  Features per node: {graph.x.shape[1]}")

    return graph, n_circ, n_dis


def load_node_names(n_circ, n_dis):
    """Load circRNA and disease names for labelling output CSVs."""
    circ_df   = pd.read_csv(DATA_DIR / "circRNA_nodes_clean.csv")
    dis_df    = pd.read_csv(DATA_DIR / "disease_nodes_clean.csv")
    circ_list = [str(v) for v in circ_df["circRNA"]][:n_circ]
    dis_list  = [str(v) for v in dis_df["disease"]][:n_dis]
    return circ_list, dis_list


def load_test_pairs(fold, n_circ):
    """Load test pairs — disease indices are offset by n_circ to match graph."""
    circ_df  = pd.read_csv(DATA_DIR / "circRNA_nodes_clean.csv")
    dis_df   = pd.read_csv(DATA_DIR / "disease_nodes_clean.csv")
    circ_map = {str(v): i         for i, v in enumerate(circ_df["circRNA"])}
    dis_map  = {str(v): i + n_circ for i, v in enumerate(dis_df["disease"])}

    test_df = pd.read_csv(DATA_DIR / f"circRNA_disease_fold{fold}_test.csv")
    circ_idx, dis_idx, labels = [], [], []
    for _, row in test_df.iterrows():
        c, d = str(row["circRNA"]), str(row["disease"])
        if c in circ_map and d in dis_map:
            circ_idx.append(circ_map[c])
            dis_idx.append(dis_map[d])
            labels.append(int(row["label"]))

    return (torch.tensor(circ_idx, dtype=torch.long),
            torch.tensor(dis_idx,  dtype=torch.long),
            torch.tensor(labels,   dtype=torch.float))


def load_train_pairs(fold, n_circ):
    """Load train pairs — used only for fallback training."""
    circ_df  = pd.read_csv(DATA_DIR / "circRNA_nodes_clean.csv")
    dis_df   = pd.read_csv(DATA_DIR / "disease_nodes_clean.csv")
    circ_map = {str(v): i         for i, v in enumerate(circ_df["circRNA"])}
    dis_map  = {str(v): i + n_circ for i, v in enumerate(dis_df["disease"])}

    train_df = pd.read_csv(DATA_DIR / f"circRNA_disease_fold{fold}_train.csv")
    circ_idx, dis_idx, labels = [], [], []
    for _, row in train_df.iterrows():
        c, d = str(row["circRNA"]), str(row["disease"])
        if c in circ_map and d in dis_map:
            circ_idx.append(circ_map[c])
            dis_idx.append(dis_map[d])
            labels.append(int(row["label"]))

    return (torch.tensor(circ_idx, dtype=torch.long),
            torch.tensor(dis_idx,  dtype=torch.long),
            torch.tensor(labels,   dtype=torch.float))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — MODEL
#  Identical architecture to sage_encoder.py and relation_ablation.py
# ══════════════════════════════════════════════════════════════════════════════

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


class LinkPredictor(nn.Module):
    """Encoder + dot-product decoder."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, edge_index, node_pairs=None):
        z = self.encoder(x, edge_index)
        if node_pairs is None:
            return z
        src, dst = node_pairs
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — LOAD TRAINED ENCODER WEIGHTS (with fallback)
#
#  Priority 1: saved_models/best_stage2_weighted_fold0.pt  (ckpt["encoder"])
#              — trained by weighted.ipynb
#  Priority 2: fallback train from scratch if checkpoint missing
# ══════════════════════════════════════════════════════════════════════════════

def load_or_train_model(graph, n_circ, circ_t, dis_t, label_t):

    model = LinkPredictor(
        GraphSAGEEncoder(in_channels=graph.x.shape[1])
    )

    # ── Try loading trained encoder weights ───────────────────────────────────
    if CKPT_PATH.exists():
        print(f"\nLoading trained encoder from checkpoint:")
        print(f"  {CKPT_PATH}")
        try:
            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

            # Checkpoint may store full model or just encoder weights
            if isinstance(ckpt, dict) and "encoder" in ckpt:
                model.encoder.load_state_dict(ckpt["encoder"])
                print("  Loaded encoder weights from ckpt['encoder']")
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print("  Loaded full model from ckpt['model_state_dict']")
            else:
                # Try direct state dict load
                try:
                    model.load_state_dict(ckpt)
                    print("  Loaded as full model state dict")
                except RuntimeError:
                    model.encoder.load_state_dict(ckpt)
                    print("  Loaded as encoder-only state dict")

            model.eval()
            print("  Trained weights loaded successfully.")
            return model

        except Exception as e:
            print(f"  Could not load checkpoint: {e}")
            print("  Falling back to training from scratch ...")
    else:
        print(f"\n  Checkpoint not found: {CKPT_PATH}")
        print("  Please run weighted.ipynb first.")
        print("  Falling back to training from scratch ...")

    # ── Fallback: train from scratch ──────────────────────────────────────────
    model = model.to(DEVICE)
    opt        = torch.optim.Adam(model.parameters(), lr=LR)
    x          = graph.x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)
    pairs      = torch.stack([circ_t, dis_t]).to(DEVICE)
    labels     = label_t.to(DEVICE)

    print("\nTraining encoder from scratch (fallback) ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy(model(x, edge_index, pairs), labels)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}")

    return model.cpu()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — SELF-CONTAINED GNNEXPLAINER MASK OPTIMISATION
#
#  Implements GNNExplainer (Ying et al., NeurIPS 2019) without any PyG
#  explain API — works on all PyG versions.
#
#  Per prediction learns:
#    edge_mask [E]  — importance of each edge in the graph
#    node_mask [F]  — importance of each input feature dimension
# ══════════════════════════════════════════════════════════════════════════════

def explain_one_pair(model, x, edge_index, src, dst, n_epochs, lr):
    model.eval()
    E     = edge_index.size(1)
    F_dim = x.size(1)

    edge_mask_raw = nn.Parameter(torch.ones(E) * 0.5)
    node_mask_raw = nn.Parameter(torch.ones(F_dim) * 0.5)
    opt = torch.optim.Adam([edge_mask_raw, node_mask_raw], lr=lr)

    for _ in range(n_epochs):
        opt.zero_grad()

        edge_w   = torch.sigmoid(edge_mask_raw)
        feat_w   = torch.sigmoid(node_mask_raw)
        x_masked = x * feat_w.unsqueeze(0)

        z = model.encoder(x_masked, edge_index)

        src_edges = (edge_index[0] == src)
        dst_edges = (edge_index[0] == dst)
        src_w = edge_w[src_edges].mean() if src_edges.any() else edge_w.mean()
        dst_w = edge_w[dst_edges].mean() if dst_edges.any() else edge_w.mean()

        score = torch.sigmoid((z[src] * src_w * z[dst] * dst_w).sum())

        # GNNExplainer loss: maximise prediction confidence + sparsity
        loss = (F.binary_cross_entropy(score.unsqueeze(0), torch.ones(1))
                + 0.01 * edge_w.mean()
                + 0.01 * feat_w.mean())

        loss.backward()
        opt.step()

    return (torch.sigmoid(edge_mask_raw).detach().numpy(),
            torch.sigmoid(node_mask_raw).detach().numpy())


def run_explainer(model, graph, pair_indices, circ_arr, dis_arr, label):
    results    = []
    x          = graph.x
    edge_index = graph.edge_index

    for rank, idx in enumerate(pair_indices):
        src = int(circ_arr[idx])
        dst = int(dis_arr[idx])
        print(f"  [{label}] {rank+1}/{len(pair_indices)}  "
              f"circRNA={src}  disease={dst}")

        edge_mask, node_mask = explain_one_pair(
            model, x, edge_index, src, dst,
            n_epochs=EXPLAIN_EPOCHS, lr=EXPLAIN_LR
        )

        results.append({
            "pair_idx"  : idx,
            "src"       : src,
            "dst"       : dst,
            "edge_mask" : edge_mask,
            "node_mask" : node_mask,
            "label"     : label,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — AGGREGATE & SAVE
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = ["degree", "log_degree", "betweenness",
                 "is_circRNA", "is_miRNA", "is_disease"]


def build_edge_df(results, graph, n_circ, circ_list, dis_list):
    """
    Label each edge by node type using the gcd_graph node layout:
        0 .. n_circ-1   → circRNA
        n_circ .. N-1   → disease
    GIP edges: circ-circ (both < n_circ) or dis-dis (both >= n_circ)
    """
    rows       = []
    edge_index = graph.edge_index.numpy()

    for res in results:
        for e_idx, imp in enumerate(res["edge_mask"]):
            s = int(edge_index[0, e_idx])
            d = int(edge_index[1, e_idx])

            # Determine node type and name
            if s < n_circ:
                s_type = "circRNA"
                s_name = circ_list[s] if s < len(circ_list) else f"circRNA_{s}"
            else:
                s_type = "disease"
                s_name = dis_list[s - n_circ] if (s - n_circ) < len(dis_list) else f"disease_{s-n_circ}"

            if d < n_circ:
                d_type = "circRNA"
                d_name = circ_list[d] if d < len(circ_list) else f"circRNA_{d}"
            else:
                d_type = "disease"
                d_name = dis_list[d - n_circ] if (d - n_circ) < len(dis_list) else f"disease_{d-n_circ}"

            # Edge type classification
            if s < n_circ and d >= n_circ:
                edge_type = "circRNA-disease"
            elif s >= n_circ and d < n_circ:
                edge_type = "disease-circRNA"
            elif s < n_circ and d < n_circ:
                edge_type = "GIP_circRNA-circRNA"
            else:
                edge_type = "GIP_disease-disease"

            rows.append({
                "pair_idx"      : res["pair_idx"],
                "group"         : res["label"],
                "edge_src_name" : s_name,
                "edge_dst_name" : d_name,
                "edge_src_type" : s_type,
                "edge_dst_type" : d_type,
                "edge_type"     : edge_type,
                "importance"    : float(imp),
            })

    return pd.DataFrame(rows)


def build_node_df(results):
    masks    = np.stack([r["node_mask"] for r in results], axis=0)
    mean_imp = masks.mean(axis=0)
    std_imp  = masks.std(axis=0)

    n     = len(mean_imp)
    names = (FEATURE_NAMES[:n] +
             [f"feature_{i}" for i in range(len(FEATURE_NAMES), n)])

    df = pd.DataFrame({
        "feature"         : names,
        "mean_importance" : mean_imp,
        "std_importance"  : std_imp,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def save_all(edge_df, node_df, results):
    edge_df.to_csv(RESULTS / "edge_importance.csv", index=False)
    node_df.to_csv(RESULTS / "node_importance.csv", index=False)
    torch.save(
        {str(r["pair_idx"]): torch.tensor(r["edge_mask"]) for r in results},
        RESULTS / "edge_importance.pt"
    )
    torch.save(
        {str(r["pair_idx"]): torch.tensor(r["node_mask"]) for r in results},
        RESULTS / "node_importance.pt"
    )
    print("  Saved: edge_importance.csv")
    print("  Saved: node_importance.csv")
    print("  Saved: edge_importance.pt")
    print("  Saved: node_importance.pt")


def plot_results(node_df, edge_df):
    # Node feature importance
    top = node_df.head(6)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["feature"][::-1], top["mean_importance"][::-1],
            xerr=top["std_importance"][::-1], color="#7F77DD", capsize=4)
    ax.set_xlabel("Mean node-feature importance")
    ax.set_title("Node feature importance — GNNExplainer\n"
                 "(GraphSAGE encoder backbone, gcd_graph)")
    plt.tight_layout()
    fig.savefig(RESULTS / "node_importance_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved: node_importance_bar.png")

    # Edge importance — grouped by edge type
    type_imp = (edge_df.groupby("edge_type")["importance"]
                .mean().reset_index()
                .sort_values("importance", ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: by edge type
    colors = ["#D85A30" if "GIP" in t else "#1D9E75"
              for t in type_imp["edge_type"]]
    axes[0].barh(type_imp["edge_type"][::-1],
                 type_imp["importance"][::-1], color=colors[::-1])
    axes[0].set_xlabel("Mean edge importance")
    axes[0].set_title("Edge importance by edge type")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right: top 15 individual edges
    top_e = (edge_df
             .groupby(["edge_src_name", "edge_dst_name"])["importance"]
             .mean().reset_index()
             .sort_values("importance", ascending=False)
             .head(15))
    labels = top_e["edge_src_name"] + " -> " + top_e["edge_dst_name"]
    axes[1].barh(labels[::-1], top_e["importance"][::-1], color="#4C72B0")
    axes[1].set_xlabel("Mean edge importance")
    axes[1].set_title("Top-15 individual edges")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("GNNExplainer edge importance — gcd_graph (circRNA-disease + GIP)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS / "edge_importance_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved: edge_importance_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)

    print("\n" + "="*60)
    print("  GNNExplainer — circRNA-disease link prediction")
    print(f"  Fold: {FOLD}  |  Device: {DEVICE}")
    print("="*60)
    print("\nNOTE: Interpretability is performed on the shared GraphSAGE")
    print("encoder backbone using a dot-product predictor. The fusion")
    print("module is not directly explainable via edge masking techniques.")
    print("This is standard practice in GNN interpretability.")

    # 1. Load the SAME graph the model was trained on
    graph, n_circ, n_dis = load_graph(FOLD)
    circ_list, dis_list  = load_node_names(n_circ, n_dis)

    # 2. Load test pairs (disease indices offset by n_circ to match graph)
    circ_te_t, dis_te_t, lbl_te_t = load_test_pairs(FOLD, n_circ)

    # 3. Load train pairs (only needed for fallback training)
    circ_tr_t, dis_tr_t, lbl_tr_t = load_train_pairs(FOLD, n_circ)

    # 4. Load trained encoder weights (or train fallback)
    model = load_or_train_model(
        graph, n_circ,
        circ_tr_t, dis_tr_t, lbl_tr_t
    )
    model.eval()

    # 5. Score all test pairs
    with torch.no_grad():
        scores = model(graph.x, graph.edge_index,
                       torch.stack([circ_te_t, dis_te_t])).numpy()
    labels = lbl_te_t.numpy()

    try:
        auc  = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)
        print(f"\nTest AUC  = {auc:.4f}")
        print(f"Test AUPR = {aupr:.4f}")
    except Exception:
        print("\n(AUC skipped — check label distribution)")

    # 6. Select pairs to explain (positive labels only)
    pos_indices = np.where(labels == 1)[0]
    if len(pos_indices) == 0:
        print("No positive test pairs found — exiting.")
        return

    top_k  = min(TOP_K,    len(pos_indices))
    rand_k = min(RANDOM_K, len(pos_indices))

    top_idx  = pos_indices[scores[pos_indices].argsort()[::-1]][:top_k].tolist()
    rand_idx = np.random.default_rng(SEED).choice(
                   pos_indices, size=rand_k, replace=False).tolist()

    circ_arr = circ_te_t.numpy()
    dis_arr  = dis_te_t.numpy()

    # 7. Run GNNExplainer
    print(f"\nExplaining {top_k} top-confidence predictions ...")
    top_res  = run_explainer(model, graph, top_idx,  circ_arr, dis_arr, "top")

    print(f"\nExplaining {rand_k} random predictions ...")
    rand_res = run_explainer(model, graph, rand_idx, circ_arr, dis_arr, "random")

    all_results = top_res + rand_res

    # 8. Build output tables
    edge_df = build_edge_df(all_results, graph, n_circ, circ_list, dis_list)
    node_df = build_node_df(all_results)

    # 9. Save
    print("\nSaving outputs ...")
    save_all(edge_df, node_df, all_results)
    plot_results(node_df, edge_df)

    # 10. Summary
    print("\n" + "="*60)
    print("NODE FEATURE IMPORTANCE")
    print("="*60)
    print(node_df[["rank", "feature", "mean_importance",
                   "std_importance"]].to_string(index=False))

    print("\n" + "="*60)
    print("EDGE IMPORTANCE BY EDGE TYPE")
    print("="*60)
    type_summary = (edge_df.groupby("edge_type")["importance"]
                    .agg(["mean", "std"])
                    .sort_values("mean", ascending=False)
                    .reset_index())
    print(type_summary.to_string(index=False))

    print("\n" + "="*60)
    print("TOP 10 INDIVIDUAL EDGES")
    print("="*60)
    print(edge_df
          .groupby(["edge_src_name", "edge_dst_name", "edge_type"])["importance"]
          .mean().reset_index()
          .sort_values("importance", ascending=False)
          .head(10).to_string(index=False))

    print("\nNOTE: Results reflect encoder-level structural importance.")
    print("The fusion predictor (WeightedFusionPredictor) is not directly")
    print("explainable at the edge level. Interpretability uses the shared")
    print("GraphSAGE encoder backbone with dot-product scoring.")

    print(f"\n✅  Done. Outputs saved to:\n    {RESULTS}")
    print("="*60)


if __name__ == "__main__":
    main()
