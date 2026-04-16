"""
embedding_analysis.py  —  Candidate A, Task 1
==============================================

Layer-wise embedding analysis using TRAINED weights from the best Stage 2
weighted fusion model (saved_models/best_stage2_weighted_fold0.pt).

NOTE ON METHODOLOGY:
    Interpretability is performed on the shared GraphSAGE encoder backbone
    using trained encoder weights. The WeightedFusionPredictor operates on
    combined embeddings and is not directly explainable via layer-wise CKA
    analysis — so we analyse the encoder, which is the component that learns
    graph structure. This is standard practice in GNN interpretability.

The encoder has 4 layers:
    Input(6) → H1(64) → H2(64, +residual) → H3(64, +residual) → Out(64, L2-norm)

Run from src/stage2/:
    python embedding_analysis.py

Prerequisites:
    Run weighted.ipynb first → saves saved_models/best_stage2_weighted_fold0.pt
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.nn import SAGEConv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
GRAPH_DIR  = ROOT / "graphs"
MODEL_DIR  = ROOT / "saved_models"
RESULTS    = ROOT / "results" / "interpretability"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
FOLD       = 0

print(f"Root      : {ROOT}")
print(f"Model dir : {MODEL_DIR}")
print(f"Results   : {RESULTS}")
print(f"Device    : {DEVICE}")

CKPT_PATH = MODEL_DIR / f"best_stage2_weighted_fold{FOLD}.pt"


# ── Instrumented encoder (identical weights to sage_encoder.py) ───────────────

class GraphSAGEEncoderInterp(nn.Module):
    """
    Same architecture as GraphSAGEEncoder in sage_encoder.py.
    Adds return_intermediates=True to expose per-layer outputs.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1   = SAGEConv(in_channels,     hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, hidden_channels)
        self.conv3   = SAGEConv(hidden_channels, hidden_channels)
        self.conv4   = SAGEConv(hidden_channels, out_channels)
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, return_intermediates: bool = False):
        h1  = self.dropout(self.act(self.conv1(x,  edge_index)))
        h2  = self.dropout(self.act(self.conv2(h1, edge_index))) + h1
        h3  = self.dropout(self.act(self.conv3(h2, edge_index))) + h2
        out = F.normalize(self.conv4(h3, edge_index), p=2, dim=1)

        if return_intermediates:
            return out, {
                "layer1": h1.detach().cpu().numpy(),
                "layer2": h2.detach().cpu().numpy(),
                "layer3": h3.detach().cpu().numpy(),
                "layer4": out.detach().cpu().numpy(),
            }
        return out


# ── Analysis helpers ───────────────────────────────────────────────────────────

def sample_cosine_sims(matrix: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(matrix), size=min(len(matrix), n), replace=False)
    sub   = matrix[idx].astype(np.float32)
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
    sub   = sub / norms
    sim   = sub @ sub.T
    return sim[np.triu_indices_from(sim, k=1)]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X   = X - X.mean(0, keepdims=True)
    Y   = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (den + 1e-10))


def mean_pairwise_l2(matrix: np.ndarray, n: int = 500, seed: int = 0) -> float:
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(matrix), size=min(len(matrix), n), replace=False)
    sub  = matrix[idx].astype(np.float32)
    diff = sub[:, None, :] - sub[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return float(dist[np.triu_indices_from(dist, k=1)].mean())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():

    # 1. Check checkpoint exists
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found: {CKPT_PATH}\n"
            f"Please run weighted.ipynb first to train the model.\n"
            f"It saves to: saved_models/best_stage2_weighted_fold{FOLD}.pt"
        )

    # 2. Load checkpoint and graph
    print(f"\nLoading checkpoint: {CKPT_PATH.name}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    graph_path = GRAPH_DIR / f"gcd_graph_fold{FOLD}.pt"
    print(f"Loading graph    : {graph_path.name}")
    graph = torch.load(graph_path, map_location=DEVICE, weights_only=False)

    x          = graph.x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)
    n_circ     = int(graph.num_circ)
    n_dis      = int(graph.num_dis)

    print(f"  Nodes : {x.shape[0]}  (circRNA={n_circ}, disease={n_dis})")
    print(f"  Edges : {edge_index.shape[1]}")
    print(f"  Features per node: {x.shape[1]}")

    # 3. Build model and load TRAINED encoder weights
    model = GraphSAGEEncoderInterp(
        in_channels=x.shape[1],
        hidden_channels=64,
        out_channels=64,
        dropout=0.0,   # disable dropout at eval time for deterministic output
    ).to(DEVICE)

    model.load_state_dict(ckpt["encoder"])   # ← loads trained weights
    model.eval()
    print("  Trained encoder weights loaded successfully.\n")

    # 4. Forward pass with intermediates
    with torch.no_grad():
        _, intermediates = model(x, edge_index, return_intermediates=True)

    LAYERS = ["layer1", "layer2", "layer3", "layer4"]
    COLORS = ["#AFA9EC", "#5DCAA5", "#EF9F27", "#D85A30"]

    embs_circ = {l: v[:n_circ] for l, v in intermediates.items()}
    embs_dis  = {l: v[n_circ:] for l, v in intermediates.items()}

    for l, v in intermediates.items():
        print(f"  {l}: shape={v.shape}")

    # ── 5. Cosine similarity distributions ────────────────────────────────────
    print("\nComputing cosine similarity distributions...")
    sim_stats = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, (ntype, embs) in zip(axes, [("circRNA", embs_circ), ("disease", embs_dis)]):
        for layer, color in zip(LAYERS, COLORS):
            sims = sample_cosine_sims(embs[layer])
            ax.hist(sims, bins=60, alpha=0.5, color=color, label=layer, density=True)
            sim_stats.append({
                "node_type": ntype, "layer": layer,
                "mean": float(sims.mean()), "std": float(sims.std()),
                "median": float(np.median(sims)),
                "q10": float(np.percentile(sims, 10)),
                "q90": float(np.percentile(sims, 90)),
            })
        ax.set_xlabel("Pairwise cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title(f"{ntype} nodes — similarity across layers (trained weights)")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(RESULTS / "cosine_similarity_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved cosine_similarity_distributions.png")
    sim_df = pd.DataFrame(sim_stats)
    sim_df.to_csv(RESULTS / "cosine_similarity_stats.csv", index=False)

    # ── 6. CKA between layers ─────────────────────────────────────────────────
    print("\nComputing CKA scores...")
    cka_records = []
    pairs = [
        ("layer1", "layer2"), ("layer2", "layer3"),
        ("layer3", "layer4"), ("layer1", "layer4"),
    ]

    for ntype, embs in [("circRNA", embs_circ), ("disease", embs_dis)]:
        for la, lb in pairs:
            score = linear_cka(embs[la], embs[lb])
            cka_records.append({"node_type": ntype, "pair": f"{la}→{lb}", "CKA": score})
            print(f"  {ntype:10s}  CKA({la},{lb}) = {score:.4f}")

    cka_df = pd.DataFrame(cka_records)
    cka_df.to_csv(RESULTS / "cka_scores.csv", index=False)

    main_pairs = ["layer1→layer2", "layer2→layer3", "layer3→layer4"]
    fig, ax    = plt.subplots(figsize=(8, 4))
    x_pos      = np.arange(len(main_pairs))
    width      = 0.35

    for i, (ntype, color) in enumerate([("circRNA", "#7F77DD"), ("disease", "#1D9E75")]):
        vals = [
            cka_df[(cka_df.node_type == ntype) & (cka_df.pair == p)]["CKA"].values[0]
            for p in main_pairs
        ]
        ax.bar(x_pos + i * width, vals, width, label=ntype, color=color, alpha=0.85)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(["L1→L2", "L2→L3", "L3→L4"], fontsize=10)
    ax.set_ylabel("Linear CKA")
    ax.set_ylim(0, 1.05)
    ax.set_title("CKA between consecutive layers — trained encoder\n"
                 "(higher = less representation change)")
    ax.legend()
    ax.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS / "cka_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved cka_bar.png")

    # ── 7. Oversmoothing check ─────────────────────────────────────────────────
    print("\nOversmoothing check...")
    smooth_records = []
    fig, ax = plt.subplots(figsize=(6, 4))

    for ntype, embs, color in [("circRNA", embs_circ, "#7F77DD"),
                                ("disease",  embs_dis,  "#1D9E75")]:
        dists = [mean_pairwise_l2(embs[l]) for l in LAYERS]
        ax.plot([1, 2, 3, 4], dists, marker="o", color=color, label=ntype)
        for l, d in zip(LAYERS, dists):
            smooth_records.append({"node_type": ntype, "layer": l, "mean_l2": d})
            print(f"  {ntype:10s}  {l}: mean_L2={d:.4f}")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Layer 1", "Layer 2", "Layer 3", "Layer 4"])
    ax.set_ylabel("Mean pairwise L2 distance")
    ax.set_title("Oversmoothing check — trained encoder")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS / "oversmoothing_check.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved oversmoothing_check.png")
    pd.DataFrame(smooth_records).to_csv(RESULTS / "oversmoothing_stats.csv", index=False)

    # ── 8. Embedding norms ────────────────────────────────────────────────────
    print("\nEmbedding norm distributions...")
    norm_records = []
    fig, axes   = plt.subplots(1, 2, figsize=(13, 4))

    for ax, (ntype, embs) in zip(axes, [("circRNA", embs_circ), ("disease", embs_dis)]):
        for layer, color in zip(LAYERS, COLORS):
            norms = np.linalg.norm(embs[layer], axis=1)
            ax.hist(norms, bins=40, alpha=0.5, color=color, label=layer, density=True)
            norm_records.append({
                "node_type": ntype, "layer": layer,
                "mean_norm": float(norms.mean()), "std_norm": float(norms.std()),
            })
        ax.set_xlabel("Embedding L2 norm")
        ax.set_ylabel("Density")
        ax.set_title(f"{ntype} — embedding norms per layer\n"
                     "(layer4 = 1.0 by L2 normalisation)")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(RESULTS / "embedding_norms.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved embedding_norms.png")
    pd.DataFrame(norm_records).to_csv(RESULTS / "embedding_norms.csv", index=False)

    # ── 9. Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("EMBEDDING ANALYSIS — SUMMARY  (trained encoder weights)")
    print("=" * 62)
    print(f"Checkpoint : {CKPT_PATH.name}")
    print(f"Graph      : gcd_graph_fold{FOLD}.pt")

    print("\n[Cosine similarity: mean ± std per layer]")
    pivot = sim_df.pivot_table(index=["node_type", "layer"], values=["mean", "std"])
    print(pivot.to_string())

    print("\n[CKA scores]")
    print(cka_df.to_string(index=False))

    print("\n[Interpretation]")
    for ntype in ["circRNA", "disease"]:
        sub = cka_df[
            (cka_df.node_type == ntype) &
            (cka_df.pair.isin(["layer1→layer2", "layer2→layer3", "layer3→layer4"]))
        ]
        for _, row in sub.iterrows():
            flag = "⚠ (similar — layer may be redundant)" if row.CKA > 0.95 else "✓"
            print(f"  {ntype:10s}  {row['pair']}  CKA={row['CKA']:.4f}  {flag}")

    print("\n[Methodology note]")
    print("  Interpretability uses trained GraphSAGE encoder weights from the")
    print("  weighted fusion model. The WeightedFusionPredictor (alpha/beta weights)")
    print("  is not analysed here — it operates post-embedding and is better studied")
    print("  via the fusion weight inspection task (Candidate B).")
    print("  Residual connections (h2+=h1, h3+=h2) will inflate CKA between adjacent")
    print("  layers — this is expected behaviour, not a flaw.")

    print(f"\nAll outputs saved to: {RESULTS}")
    print("=" * 62)


if __name__ == "__main__":
    main()
