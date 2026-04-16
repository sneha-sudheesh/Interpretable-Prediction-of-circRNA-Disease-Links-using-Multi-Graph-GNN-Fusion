import pandas as pd
import os

os.makedirs("data_cleaned", exist_ok=True)

# Load files
circ_dis = pd.read_excel("data_raw/circRNA-disease.xlsx")
mir_dis = pd.read_excel("data_raw/miRNA-disease.xlsx")
circ_mir = pd.read_excel("data_raw/circRNA-miRNA.xlsx")
circ = pd.read_excel("data_raw/circRNA.xlsx")
mir = pd.read_excel("data_raw/miRNA.xlsx")
dis = pd.read_excel("data_raw/diseases.xlsx")

# Preview
for name, df in [("circRNA-disease", circ_dis),
                 ("miRNA-disease", mir_dis),
                 ("circRNA-miRNA", circ_mir),
                 ("circRNA", circ),
                 ("miRNA", mir),
                 ("disease", dis)]:
    print(f"\n{name} → {df.shape} rows, {df.columns.tolist()[:5]}")
