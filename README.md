# Interpretable Prediction of circRNA-Disease Links using Multi-Graph GNN Fusion

A bioinformatics-based machine learning project that predicts potential associations between circular RNAs (circRNAs) and diseases using Graph Neural Networks (GNNs). The model learns hidden biological relationships from circRNA, miRNA, and disease interaction data to identify possible disease associations.

---

## Features

* Heterogeneous biological graph construction
* Graph Neural Network based prediction
* circRNA–miRNA–disease interaction modeling
* K-Fold Cross Validation
* Performance evaluation using AUC and AUPR.
* Model interpretability and debugging support
* Built using PyTorch Geometric

---

## Tech Stack

* Python
* PyTorch
* PyTorch Geometric
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## Project Structure

```bash
circRNA-disease-gnn/
│
├── data/                # Dataset files
├── models/              # GNN model architectures
├── preprocessing/       # Data preprocessing scripts
├── interpretability/    # Model debugging and analysis
├── training/            # Training and validation
├── utils/               # Utility functions
├── results/             # Output results and metrics
├── main.py              # Main execution file
└── requirements.txt
```

---

## Methodology

1. Preprocess biological interaction datasets
2. Construct a heterogeneous graph of circRNAs, miRNAs, and diseases
3. Train a Graph Neural Network to learn node embeddings
4. Predict potential circRNA–disease associations
5. Evaluate performance using cross-validation and classification metrics

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/circRNA-disease-gnn.git
cd circRNA-disease-gnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Results

The model successfully learns biological interaction patterns and predicts potential circRNA–disease associations with strong classification performance. Evaluation includes:

* AUPR
* AUC

---

## Future Improvements

* Integration of larger biological datasets
* Advanced GNN architectures
* Explainable AI techniques
* Web-based visualization dashboard

---

## Author

**Sneha Sudheesh**
B.Tech Computer Science and Engineering
Government College of Engineering Kannur
