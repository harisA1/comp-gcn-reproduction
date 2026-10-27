"""Convert the authors' xlsx files into data/*.npz.

    python prepare_data.py /path/to/COMP-GNN_data

The datasets are not included in this repository; request them from the paper's authors.
"""
import os
import sys
import numpy as np
import pandas as pd


def main(src, out="data"):
    os.makedirs(out, exist_ok=True)
    for y in (2019, 2020, 2021):
        n = pd.read_excel(f"{src}/{y}_stock_network_fillna_nodes_data.xlsx")
        e = pd.read_excel(f"{src}/{y}_stock_network_edges_data.xlsx")
        X = n.filter(like="title_").values          # 12 months x (posi, nega, neutral)
        np.savez(f"{out}/stock_{y}.npz", X=X, E=e[["source", "target"]].values,
                 y=n["label"].values, D=np.array([3] * 12))
    for t in (1, 2, 3, 4):
        n = pd.read_excel(f"{src}/term{t}_student_network_nodes_data.xlsx")
        e = pd.read_excel(f"{src}/term{t}_student_network_edges_data.xlsx")
        cons = n[["食堂消费", "超市消费", "淋浴消费", "网络缴费"]].values   # cafeteria, supermarket, shower, internet
        shower = n[[f"shower_week{i}" for i in range(1, 17)]].values
        breakfast = n[[f"breakfast_week{i}" for i in range(1, 17)]].values
        internet = n[[f"week{i}" for i in range(1, 17)]].values
        # chengji_ave (grade average) is deliberately excluded: it leaks the label
        X = np.concatenate([internet, breakfast, shower, cons], 1)
        np.savez(f"{out}/student_term{t}.npz", X=X, E=e[["source", "target"]].values,
                 y=n["fail_not"].values.astype(int), D=np.array([16, 16, 16, 4]))
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1])
