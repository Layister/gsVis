#!/usr/bin/env python3
# build_global_markers.py
"""
global marker 构建流程：
- 从 PanglaoDB 加载 marker（organ → cell_type → genes）
- 基于所有样本 spot 的基因表达，过滤掉泛表达 / 基本不表达的基因
- 输出 global_markers.json，用于后续细胞类型推断
"""

import os
import json
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc


"""
panglao_to_broad = {
    "Acinar cells": "Acinar cell",
    "Adipocyte progenitor cells": "Adipocyte lineage cell",
    "Adipocytes": "Adipocyte lineage cell",
    "Adrenergic neurons": "Neuron",
    "Airway epithelial cells": "Epithelial cell",
    "Airway goblet cells": "Goblet cell",
    "Alpha cells": "Pancreatic endocrine cell",
    "Alveolar macrophages": "Monocyte/Macrophage",
    "Anterior pituitary gland cells": "Pituitary endocrine cell",
    "Astrocytes": "Astrocyte",
    "B cells": "B cell",
    "B cells memory": "B cell",
    "B cells naive": "B cell",
    "Basal cells": "Basal epithelial cell",
    "Basophils": "Basophil",
    "Bergmann glia": "Glial cell",
    "Beta cells": "Pancreatic endocrine cell",
    "Cajal-Retzius cells": "Neuron",
    "Cardiac stem and precursor cells": "Stem/Progenitor cell",
    "Cardiomyocytes": "Cardiomyocyte",
    "Cholangiocytes": "Cholangiocyte",
    "Cholinergic neurons": "Neuron",
    "Chondrocytes": "Chondrocyte",
    "Choroid plexus cells": "Choroid plexus cell",
    "Chromaffin cells": "Endocrine/secretory cell",
    "Ciliated cells": "Epithelial cell",
    "Clara cells": "Epithelial cell",
    "Crypt cells": "Intestinal epithelial cell",
    "Delta cells": "Pancreatic endocrine cell",
    "Dendritic cells": "Dendritic cell",
    "Distal tubule cells": "Renal tubular epithelial cell",
    "Dopaminergic neurons": "Neuron",
    "Ductal cells": "Ductal cell",
    "Embryonic stem cells": "Stem/Progenitor cell",
    "Endothelial cells": "Endothelial cell",
    "Endothelial cells (aorta)": "Endothelial cell",
    "Enteric glia cells": "Glial cell",
    "Enteric neurons": "Neuron",
    "Enterochromaffin cells": "Enteroendocrine cell",
    "Enterocytes": "Enterocyte",
    "Enteroendocrine cells": "Enteroendocrine cell",
    "Eosinophils": "Eosinophil",
    "Ependymal cells": "Ependymal cell",
    "Epiblast cells": "Stem/Progenitor cell",
    "Epithelial cells": "Epithelial cell",
    "Epsilon cells": "Pancreatic endocrine cell",
    "Erythroblasts": "Erythroid lineage cell",
    "Erythroid-like and erythroid precursor cells": "Erythroid lineage cell",
    "Fibroblasts": "Fibroblast",
    "Follicular cells": "Endocrine/secretory cell",
    "Foveolar cells": "Gastric foveolar cell",
    "GABAergic neurons": "Neuron",
    "Gamma (PP) cells": "Pancreatic endocrine cell",
    "Gamma delta T cells": "T cell",
    "Gastric chief cells": "Endocrine/secretory cell",
    "Germ cells": "Germ cell",
    "Glomus cells": "Endocrine/secretory cell",
    "Glutaminergic neurons": "Neuron",
    "Glycinergic neurons": "Neuron",
    "Goblet cells": "Goblet cell",
    "Granulosa cells": "Granulosa cell",
    "Hematopoietic stem cells": "Hematopoietic stem/progenitor cell",
    "Hepatic stellate cells": "Stellate cell",
    "Hepatoblasts": "Hepatic progenitor cell",
    "Hepatocytes": "Hepatocyte",
    "Immature neurons": "Neuron",
    "Intercalated cells": "Renal tubular epithelial cell",
    "Interneurons": "Neuron",
    "Ionocytes": "Epithelial cell",
    "Keratinocytes": "Epithelial cell",
    "Kupffer cells": "Monocyte/Macrophage",
    "Langerhans cells": "Dendritic cell",
    "Leydig cells": "Endocrine/secretory cell",
    "Loop of Henle cells": "Renal tubular epithelial cell",
    "Luminal epithelial cells": "Luminal epithelial cell",
    "Luteal cells": "Endocrine/secretory cell",
    "Macrophages": "Monocyte/Macrophage",
    "Mammary epithelial cells": "Epithelial cell",
    "Mast cells": "Mast cell",
    "Megakaryocytes": "Megakaryocyte/Platelet",
    "Melanocytes": "Melanocyte",
    "Meningeal cells": "Meningeal cell",
    "Merkel cells": "Epithelial cell",
    "Mesangial cells": "Mesangial cell",
    "Mesothelial cells": "Mesothelial cell",
    "Microfold cells": "Microfold (M) cell",
    "Microglia": "Glial cell",
    "Monocytes": "Monocyte/Macrophage",
    "Motor neurons": "Neuron",
    "Myeloid-derived suppressor cells": "Monocyte/Macrophage",
    "Myoblasts": "Muscle/Pericyte cell",
    "Myocytes": "Muscle/Pericyte cell",
    "Myoepithelial cells": "Epithelial cell",
    "Myofibroblasts": "Fibroblast",
    "Müller cells": "Glial cell",
    "NK cells": "NK cell",
    "Natural killer T cells": "T cell",
    "Neural stem/precursor cells": "Stem/Progenitor cell",
    "Neuroblasts": "Neuron",
    "Neuroendocrine cells": "Endocrine/secretory cell",
    "Neurons": "Neuron",
    "Neutrophils": "Neutrophil",
    "Noradrenergic neurons": "Neuron",
    "Nuocytes": "Innate lymphoid cell",
    "Olfactory epithelial cells": "Epithelial cell",
    "Oligodendrocyte progenitor cells": "Glial cell",
    "Oligodendrocytes": "Glial cell",
    "Osteoblasts": "Osteo-lineage cell",
    "Osteoclast precursor cells": "Osteo-lineage cell",
    "Osteoclasts": "Osteo-lineage cell",
    "Osteocytes": "Osteo-lineage cell",
    "Oxyphil cells": "Parathyroid cell",
    "Pancreatic progenitor cells": "Pancreatic progenitor cell",
    "Pancreatic stellate cells": "Stellate cell",
    "Paneth cells": "Paneth cell",
    "Parathyroid chief cells": "Parathyroid cell",
    "Parietal cells": "Gastric parietal cell",
    "Peri-islet Schwann cells": "Glial cell",
    "Pericytes": "Muscle/Pericyte cell",
    "Peritubular myoid cells": "Muscle/Pericyte cell",
    "Photoreceptor cells": "Photoreceptor cell",
    "Pinealocytes": "Endocrine/secretory cell",
    "Plasma cells": "Plasma cell",
    "Plasmacytoid dendritic cells": "Dendritic cell",
    "Platelets": "Megakaryocyte/Platelet",
    "Pluripotent stem cells": "Stem/Progenitor cell",
    "Podocytes": "Podocyte",
    "Principal cells": "Renal principal cell",
    "Proximal tubule cells": "Renal tubular epithelial cell",
    "Pulmonary alveolar type I cells": "Alveolar epithelial cell",
    "Pulmonary alveolar type II cells": "Alveolar epithelial cell",
    "Purkinje neurons": "Neuron",
    "Pyramidal cells": "Neuron",
    "Radial glia cells": "Glial cell",
    "Red pulp macrophages": "Monocyte/Macrophage",
    "Reticulocytes": "Erythroid lineage cell",
    "Retinal ganglion cells": "Neuron",
    "Retinal progenitor cells": "Retinal progenitor cell",
    "Satellite cells": "Muscle stem/progenitor cell",
    "Satellite glial cells": "Glial cell",
    "Schwann cells": "Glial cell",
    "Sebocytes": "Sebaceous gland cell",
    "Serotonergic neurons": "Neuron",
    "Sertoli cells": "Endocrine/secretory cell",
    "Smooth muscle cells": "Muscle/Pericyte cell",
    "Stromal cells": "Fibroblast",
    "T cells": "T cell",
    "T cytotoxic cells": "CD8 T cell",
    "T follicular helper cells": "CD4 T cell",
    "T helper cells": "CD4 T cell",
    "T memory cells": "T cell",
    "T regulatory cells": "Treg",
    "Tanycytes": "Tanycyte",
    "Taste receptor cells": "Taste receptor cell",
    "Thymocytes": "T cell",
    "Transient cells": "Epithelial cell",
    "Trichocytes": "Hair follicle epithelial cell",
    "Trigeminal neurons": "Neuron",
    "Trophoblast cells": "Trophoblast cell",
    "Trophoblast progenitor cells": "Trophoblast progenitor cell",
    "Trophoblast stem cells": "Trophoblast stem cell",
    "Tuft cells": "Tuft cell",
    "Urothelial cells": "Epithelial cell"
}
"""


# ========= 1. 加载 PanglaoDB 数据============
def load_cellmarker(xlsx_path: str):

    df = pd.read_excel(xlsx_path)

    # 1. Human only + canonical markers
    df = df[df["species"].astype(str).str.contains("Hs", na=False)]
    df = df[df["canonical marker"] == 1.0]

    # 2. 统一 gene symbol
    df["official gene symbol"] = df["official gene symbol"].astype(str).str.upper()

    # 3. organ 清洗
    if "organ" not in df.columns:
        raise ValueError("PanglaoDB file must contain 'organ' column.")
    df["organ"] = df["organ"].astype(str).str.strip().str.replace(" ", "_")

    # 4. 构建 organ → celltype → markers
    marker_dict = {}
    for organ, g1 in df.groupby("organ"):
        organ_dict = {}
        for celltype, g2 in g1.groupby("cell type"):
            genes = sorted(set(g2["official gene symbol"].tolist()))
            organ_dict[celltype] = genes
        marker_dict[organ] = organ_dict

    print(f"[INFO] 加载 PanglaoDB 成功：{len(marker_dict)} 个器官，"
          f"{sum(len(v) for v in marker_dict.values())} 个细胞类型")
    return marker_dict


# ========= 2. 读取多个 cancer adata，整合为一个大矩阵 =========
def load_all_adata(adata_paths):
    adatas = []
    for p in adata_paths:
        print(f"[INFO] Loading {p}")
        adata = sc.read_h5ad(p)

        # 加前缀确保 obs_names 唯一
        sample_name = os.path.basename(p).replace("_add_latent.h5ad", "")
        adata.obs_names = [f"{sample_name}_{x}" for x in adata.obs_names]

        adatas.append(adata)

    print("[INFO] Concatenating all adata...")
    ad_all = anndata.concat(
        adatas, join="outer", merge="same",
        label="batch", index_unique=None
    )
    return ad_all


# ========= 3. 基于全体 expression 过滤掉泛表达/稀有 marker =========
def refine_markers_by_expression(
    marker_dict,
    ad_all,
    min_pos_cells=100,
    max_fraction=None,
    method="iqr",
    iqr_mult=1.5,
    mad_mult=3.0,
    entropy_high_quantile=0.9,
):
    """
    更鲁棒地基于表达过滤 PanglaoDB markers。

    步骤：
    1. 使用 min_pos_cells 去掉几乎不表达的基因；
    2. 可选使用 max_fraction 做一个全局的上界；
    3. 在此基础上，用 IQR / MAD / entropy 自动识别「泛表达」和极端 outlier 基因。

    ----
    method : {"iqr", "mad", "entropy", "none"}
        选择用于进一步识别 outlier 的方法：
        - "iqr": 基于检测比例的 IQR，去掉过高/过低的 fraction
        - "mad": 在 logit(fraction) 空间用 MAD 去掉极端高/低 fraction
        - "entropy": 基于表达分布的 Shannon 熵去掉在所有细胞中较均匀的基因
        - "none": 仅使用 min_pos_cells / max_fraction，不做额外鲁棒统计过滤
    """
    X = ad_all.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    n_cells, n_genes = X.shape

    # 0/1 检测矩阵：是否表达
    expr_binary = (X > 0).astype(float)
    pos_counts = expr_binary.sum(axis=0)  # 每个基因在多少个细胞中表达
    frac = pos_counts / float(n_cells)    # 检测比例 in [0,1]

    # ---- Step 1: 基础过滤：至少在 min_pos_cells 中表达，且表达比例不多于max_fraction ----
    base_mask = pos_counts >= float(min_pos_cells)
    base_mask &= (frac <= float(max_fraction))

    if not np.any(base_mask):
        raise RuntimeError(
            f"No genes pass basic filters (min_pos_cells={min_pos_cells}, "
            f"max_fraction={max_fraction}). Try relaxing thresholds."
        )

    method = (method or "none").lower()

    # 如果只想要旧逻辑，可以 method="none"
    if method in ("none", "off"):
        keep_mask = base_mask

    elif method == "iqr":
        # 在通过基础过滤的基因上计算 IQR
        frac_valid = frac[base_mask]
        q1, q3 = np.quantile(frac_valid, [0.25, 0.75])
        iqr = q3 - q1
        # 允许的数据范围 [q1 - k*IQR, q3 + k*IQR]，再夹到 [0,1]
        low_cut = max(0.0, q1 - iqr_mult * iqr)
        high_cut = min(1.0, q3 + iqr_mult * iqr)

        keep_mask = base_mask & (frac >= low_cut) & (frac <= high_cut)

    elif method == "mad":
        # 在 logit(fraction) 空间计算 MAD，更适合右偏分布
        frac_valid = frac[base_mask]
        eps = 1e-6
        p_clip = np.clip(frac_valid, eps, 1.0 - eps)
        z_valid = np.log(p_clip / (1.0 - p_clip))  # logit(p)

        med = np.median(z_valid)
        mad = np.median(np.abs(z_valid - med)) + 1e-8  # 防止除零

        p_all = np.clip(frac, eps, 1.0 - eps)
        z_all = np.log(p_all / (1.0 - p_all))

        z_low = med - mad_mult * mad
        z_high = med + mad_mult * mad

        keep_mask = base_mask & (z_all >= z_low) & (z_all <= z_high)

    elif method == "entropy":
        # 基于表达量的 Shannon 熵：
        # 对每个基因，归一化其在所有细胞中的表达分布，然后计算熵。
        X_pos = np.maximum(X, 0.0)          # 简单忽略负值
        gene_sum = X_pos.sum(axis=0)        # 每个基因的总表达量

        # 只对同时通过 base_mask 且总表达>0 的基因计算熵
        valid_idx = np.where(base_mask & (gene_sum > 0))[0]
        if valid_idx.size == 0:
            raise RuntimeError(
                "No genes with non-zero expression after base filters for entropy method."
            )

        sub_X = X_pos[:, valid_idx]                     # (n_cells, n_valid_genes)
        sub_sum = gene_sum[valid_idx]                   # (n_valid_genes,)
        prob = sub_X / (sub_sum + 1e-12)                # 概率分布：每个基因在各细胞上的相对贡献
        # 熵：对每个基因沿细胞维度求和
        entropy = -(prob * np.log2(prob + 1e-12)).sum(axis=0)   # (n_valid_genes,)
        entropy_norm = entropy / np.log2(float(n_cells))        # 归一化到 [0,1]

        # 构造一个和所有基因对齐的熵向量
        entropy_full = np.zeros(n_genes, dtype=float)
        entropy_full[valid_idx] = entropy_norm

        # 把熵最高的一部分基因视为「在所有细胞中较均匀表达」的非特异性基因
        high_cut = np.quantile(entropy_norm, entropy_high_quantile)
        keep_mask = base_mask & (entropy_full < high_cut)

    else:
        raise ValueError(f"Unknown method='{method}'")

    valid_genes = set(np.asarray(ad_all.var_names)[keep_mask])

    print(
        "[INFO] Expression-based filtering: "
        f"{n_genes} genes → {base_mask.sum()} after basic filters "
        f"(min_pos_cells={min_pos_cells}, max_fraction={max_fraction}) → "
        f"{len(valid_genes)} genes after method='{method}'."
    )

    # -------- organ-aware refine --------
    refined = {}
    for organ, organ_dict in marker_dict.items():
        organ_refined = {}
        for celltype, genes in organ_dict.items():
            keep = [g for g in genes if g in valid_genes]
            if keep:
                organ_refined[celltype] = sorted(keep)

        if organ_refined:
            refined[organ] = organ_refined

    print(
        f"[INFO] Refinement done: {sum(len(v) for v in refined.values())} 个细胞类型，"
        f"分布在 {len(refined)} 个器官中"
    )
    return refined


# ========= 4. 主程序============
def main():
    data_dir = "/home/wuyang/hest-data/process/"
    species = "Homo sapiens"

    adata_paths = [
        os.path.join(data_dir, species, "COAD", "TENX89_adata.h5ad"),
        os.path.join(data_dir, species, "COAD", "TENX90_adata.h5ad"),
        os.path.join(data_dir, species, "COAD", "TENX91_adata.h5ad"),
        os.path.join(data_dir, species, "COAD", "TENX92_adata.h5ad"),

        os.path.join(data_dir, species, "EPM", "NCBI629_adata.h5ad"),
        os.path.join(data_dir, species, "EPM", "NCBI630_adata.h5ad"),
        os.path.join(data_dir, species, "EPM", "NCBI631_adata.h5ad"),
        os.path.join(data_dir, species, "EPM", "NCBI632_adata.h5ad"),
        os.path.join(data_dir, species, "EPM", "NCBI633_adata.h5ad"),

        os.path.join(data_dir, species, "IDC", "NCBI681_adata.h5ad"),
        os.path.join(data_dir, species, "IDC", "NCBI682_adata.h5ad"),
        os.path.join(data_dir, species, "IDC", "NCBI683_adata.h5ad"),
        os.path.join(data_dir, species, "IDC", "NCBI684_adata.h5ad"),
        os.path.join(data_dir, species, "IDC", "TENX13_adata.h5ad"),
        os.path.join(data_dir, species, "IDC", "TENX14_adata.h5ad"),

        os.path.join(data_dir, species, "PAAD", "NCBI569_adata.h5ad"),
        os.path.join(data_dir, species, "PAAD", "NCBI570_adata.h5ad"),
        os.path.join(data_dir, species, "PAAD", "NCBI571_adata.h5ad"),
        os.path.join(data_dir, species, "PAAD", "NCBI572_adata.h5ad"),

        os.path.join(data_dir, species, "PRAD", "INT25_adata.h5ad"),
        os.path.join(data_dir, species, "PRAD", "INT26_adata.h5ad"),
        os.path.join(data_dir, species, "PRAD", "INT27_adata.h5ad"),
        os.path.join(data_dir, species, "PRAD", "INT28_adata.h5ad"),
        os.path.join(data_dir, species, "PRAD", "TENX40_adata.h5ad"),
        os.path.join(data_dir, species, "PRAD", "TENX46_adata.h5ad"),

        os.path.join(data_dir, species, "READ", "ZEN36_adata.h5ad"),
        os.path.join(data_dir, species, "READ", "ZEN40_adata.h5ad"),
        os.path.join(data_dir, species, "READ", "ZEN48_adata.h5ad"),
        os.path.join(data_dir, species, "READ", "ZEN49_adata.h5ad"),
    ]

    cell_marker_path = "/home/wuyang/hest-data/PanglaoDB_Cell_marker.xlsx"
    output_dir = "./global_ref"

    parser = argparse.ArgumentParser(description="Global marker builder")
    parser.add_argument("--cell_marker_path", default=cell_marker_path)
    parser.add_argument("--adata_list", default=adata_paths)
    parser.add_argument("--output", default=output_dir)
    parser.add_argument("--min_pos_cells", type=float, default=50)
    parser.add_argument("--max_fraction", type=float, default=0.5)
    args = parser.parse_args()

    # -------- 加载 PanglaoDB markers --------
    panglao = load_cellmarker(args.cell_marker_path)

    # -------- 合并所有 adata --------
    adata_all = load_all_adata(args.adata_list)

    # -------- refine markers --------
    refined = refine_markers_by_expression(
        panglao,
        adata_all,
        min_pos_cells=args.min_pos_cells,
        max_fraction=args.max_fraction,
        method="mad",  # 或 "iqr" / "entropy" / "none"
    )

    os.makedirs(args.output, exist_ok=True)

    # -------- 保存 global_markers.json --------
    json_path = os.path.join(args.output, "global_markers.json")
    with open(json_path, "w") as f:
        json.dump(refined, f, indent=2)
    print(f"[INFO] Saved organ-aware global markers → {json_path}")

    # -------- 保存 marker_summary.csv --------
    summary_path = os.path.join(args.output, "marker_summary.csv")
    summary_list = []
    for organ, organ_dict in refined.items():
        for celltype, genes in organ_dict.items():
            summary_list.append({
                "organ": organ,
                "cell_type": celltype,
                "n_markers": len(genes)
            })

    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values("n_markers", ascending=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Saved organ-aware marker summary → {summary_path}")


if __name__ == "__main__":
    main()
