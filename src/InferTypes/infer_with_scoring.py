#!/usr/bin/env python3
"""
Unified scoring runner: AUCell + PAGE + ssGSEA with QC and robust fusion.

- 读取 organ-aware 的 global_markers.json: {organ: {cell_type: [genes]}}
- 根据 cancer_type 选择相关 organ，再自动加上 Immune_system / Vasculature / Connective_tissue 等通用组织（可关闭）
- 使用 rank-based AUCell、PAGE、ssGSEA 计算打分
- 动态加权融合三种方法，输出 presence-like likelihood（稀疏、不强制和为 1）
- 输出：
    - aucell_results.json
    - page_results.json
    - ssgsea_results.json
    - mixture_fused.json
    - qc_report.json（可选）
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp


# --------------------------------------------------------------------------- #
# 配置：癌种 -> 主要组织
# --------------------------------------------------------------------------- #

MIN_MARKERS_PER_CT = 2

CANCER_TO_ORGANS = {
    # 消化系统肿瘤
    "COAD": ["GI_tract"],  # 结肠癌：胃肠道
    "READ": ["GI_tract"],  # 直肠癌：胃肠道
    "EPM": ["GI_tract"],  # 肠息肉：胃肠道
    "PAAD": ["Pancreas", "GI_tract", "Liver"],  # 胰腺癌：胰腺 + 胃肠道 + 肝
    "ESCC": ["GI_tract"],  # 食管鳞状细胞癌：胃肠道
    "ESCA": ["GI_tract"],  # 食管腺癌：胃肠道
    "STAD": ["GI_tract"],  # 胃癌：胃肠道
    "LIHC": ["Liver", "GI_tract"],  # 肝癌：肝 + 胃肠道

    # 乳腺肿瘤
    "IDC": ["Mammary_gland"],  # 浸润性导管癌：乳腺
    "ILC": ["Mammary_gland"],  # 浸润性小叶癌：乳腺
    "DCIS": ["Mammary_gland"],  # 导管原位癌：乳腺
    "BRCA": ["Mammary_gland"],  # 乳腺癌（通用）：乳腺

    # 呼吸系统肿瘤
    "LUAD": ["Lungs"],  # 肺腺癌：肺
    "LUSC": ["Lungs"],  # 肺鳞癌：肺
    "SCLC": ["Lungs"],  # 小细胞肺癌：肺

    # 泌尿系统肿瘤
    "KIRC": ["Kidney"],  # 肾透明细胞癌：肾
    "KIRP": ["Kidney"],  # 肾乳头状细胞癌：肾
    "KICH": ["Kidney"],  # 肾嫌色细胞癌：肾
    "BLCA": ["Urinary_bladder"],  # 膀胱癌：膀胱

    # 生殖系统肿瘤
    "PRAD": ["Reproductive"],  # 前列腺癌：生殖系统
    "OV": ["Reproductive"],  # 卵巢癌：生殖系统
    "CESC": ["Reproductive"],  # 宫颈鳞癌：生殖系统

    # 其他常见肿瘤
    "SKCM": ["Skin"],  # 皮肤黑色素瘤：皮肤
    "GBM": ["Brain"],  # 胶质母细胞瘤：脑
    "LGG": ["Brain"],  # 低级别胶质瘤：脑
    "THCA": ["Thyroid"],  # 甲状腺癌：甲状腺
    "HNSC": ["Epithelium"],  # 头颈鳞癌：上皮
    "ACC": ["Adrenal_glands"],  # 肾上腺皮质癌：肾上腺
    "PCPG": ["Adrenal_glands"],  # 副神经节瘤：肾上腺

    # 血液肿瘤
    "LAML": ["Blood", "Bone"],  # 急性髓系白血病：血液 + 骨
    "ALL": ["Blood", "Bone"],  # 急性淋巴细胞白血病：血液 + 骨
    "CLL": ["Blood", "Bone"],  # 慢性淋巴细胞白血病：血液 + 骨
    "MM": ["Blood", "Bone"],  # 多发性骨髓瘤：血液 + 骨

    # 肉瘤
    "SARC": ["Connective_tissue", "Skeletal_muscle", "Bone"],  # 肉瘤：结缔组织 + 骨骼肌 + 骨
    "LMS": ["Smooth_muscle"],  # 平滑肌肉瘤：平滑肌
    "OS": ["Bone"],  # 骨肉瘤：骨
    "CHS": ["Connective_tissue"],  # 软骨肉瘤：结缔组织
}

COMMON_ORGANS = ["Immune_system", "Vasculature", "Connective_tissue"]


# --------------------------------------------------------------------------- #
# Marker utilities
# --------------------------------------------------------------------------- #

def load_markers(
    path: str,
    cancer_type: str,
    marker_organs: Optional[str] = None,
    include_common_organs: bool = True,
) -> Dict[str, Iterable[str]]:
    """
    Load organ-aware marker JSON and flatten to cell_type -> genes.

    参数
    ----
    path:
        global_markers.json 路径（由 build_global_markers.py 生成）
    cancer_type:
        当前分析的癌种（例如 "COAD", "PAAD"），用于自动选择 organ。
    marker_organs:
        手动指定 organ 列表，逗号分隔，例如 "GI_tract,Pancreas"。
        如果提供，则优先使用，不再根据 cancer_type。
    include_common_organs:
        若为 True，则如果 JSON 中存在 Immune_system / Vasculature / Connective_tissue，
        会自动加入这些组织的 marker。

    返回
    ----
    markers_flat : dict
        cell_type_with_organ -> [UPPERCASE_GENE_SYMBOLS]
        为避免 organ 间同名 cell_type 冲突，cell_type 名字会附上 organ，例如：
        "Endothelial cells (Vasculature)"。
    """
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("Marker JSON must be a non-empty dict: {organ: {cell_type: [genes]}}")

    organ_to_ct = data  # organ -> cell_type -> genes

    # 1) 选 organ
    selected_organs: set = set()

    if marker_organs:
        # 手动指定 organ
        selected_organs = {o.strip() for o in marker_organs.split(",") if o.strip()}
    else:
        ct = cancer_type.upper()
        selected_organs = set(CANCER_TO_ORGANS.get(ct, []))

    # 2) 加通用 organ（如果存在）
    if include_common_organs:
        for organ in COMMON_ORGANS:
            if organ in organ_to_ct:
                selected_organs.add(organ)

    # 3) 如果还没选上任何 organ，就退化为所有 organ
    if not selected_organs:
        selected_organs = set(organ_to_ct.keys())

    # 4) 展平为 cell_type -> genes，并统一基因名为大写
    markers_flat: Dict[str, Iterable[str]] = {}
    for organ in sorted(selected_organs):
        organ_dict = organ_to_ct.get(organ, {})
        for cell_type, genes in organ_dict.items():
            if not genes:
                continue
            ct_key = f"{cell_type} ({organ})"
            markers_flat[ct_key] = [str(g).upper() for g in genes]

    print(
        f"[INFO] Loaded organ-aware markers from {path}: "
        f"{len(selected_organs)} organs, {len(markers_flat)} cell types"
    )
    return markers_flat


def compute_marker_coverage(
    markers: Mapping[str, Iterable[str]], gene_universe: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    """Return per-cell-type marker coverage against the expression gene set."""
    gene_set = set(gene_universe)
    coverage = {}
    for ct, genes in markers.items():
        genes = list(genes)
        total = len(genes)
        if total == 0:
            coverage[ct] = {"present": 0, "total": 0, "ratio": 0.0}
            continue
        present = sum(1 for g in genes if g in gene_set)
        coverage[ct] = {
            "present": present,
            "total": total,
            "missing": total - present,
            "ratio": round(present / total, 4),
        }
    return coverage


def build_marker_index(
    markers: Mapping[str, Iterable[str]],
    gene_names: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    将 marker 字典转换为 gene index 形式，方便高效计算。

    参数
    ----
    markers:
        cell_type -> [gene symbols (UPPER)]
    gene_names:
        表达矩阵中基因名（UPPER）

    返回
    ----
    marker_index:
        cell_type -> np.ndarray[int]  (在 gene_names 中的列索引)
    """
    gene_index = {g: i for i, g in enumerate(gene_names)}
    marker_index: Dict[str, np.ndarray] = {}
    n_mapped = 0

    for ct, genes in markers.items():
        idxs = [gene_index[g] for g in genes if g in gene_index]
        arr = np.asarray(sorted(set(idxs)), dtype=np.int32) if idxs else np.asarray([], dtype=np.int32)

        if arr.size < MIN_MARKERS_PER_CT:
            continue

        marker_index[ct] = arr
        n_mapped += arr.size

    print(
        f"[INFO] Built marker index: {len(marker_index)} cell types, "
        f"{n_mapped} marker genes present in data"
    )
    return marker_index


# --------------------------------------------------------------------------- #
# AUCell
# --------------------------------------------------------------------------- #

def compute_aucell_scores(
    X: np.ndarray,
    obs_names: Iterable[str],
    gene_names: np.ndarray,
    marker_index: Mapping[str, np.ndarray],
    top_frac: float = 0.05,
    auc_max_rank: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Rank-based AUCell-like scores.

    对每个 spot：
      1. 取表达最高的 top_k 基因（top_frac 或 auc_max_rank 控制）；
      2. 在这 top_k 基因内部按表达量从高到低排序；
      3. 对每个 cell type，用 marker 在这个 rank 列表中的相对位置计算一个 AUC-like 指标：
         - 所有 marker 都在 top_k 顶部 → 接近 1；
         - 很少 marker 进入 top_k 或都在靠后位置 → 接近 0。
    """
    X = np.asarray(X)
    n_spots, n_genes = X.shape
    obs_names = list(obs_names)

    if auc_max_rank is not None and auc_max_rank > 0:
        top_k = min(n_genes, int(auc_max_rank))
    else:
        top_k = max(1, min(n_genes, int(round(n_genes * float(top_frac)))))

    print(f"[AUCell] Rank-based mode, using top {top_k} / {n_genes} genes per spot")

    results: Dict[str, Dict[str, float]] = {}

    for i, spot in enumerate(obs_names):
        row = X[i, :]

        # 1) 取 top_k 基因索引（未排序）
        if top_k < n_genes:
            # 取最大的 top_k 个
            top_idx = np.argpartition(row, -top_k)[-top_k:]
        else:
            top_idx = np.arange(n_genes, dtype=np.int32)

        # 2) 在 top_k 内按表达量从高到低排序
        top_vals = row[top_idx]
        order = np.argsort(top_vals)[::-1]  # descending
        ranked_idx = top_idx[order]         # 基因索引，长度为 top_k

        # 3) 构建 gene_idx -> rank 的查找表（0 = 最高表达）
        rank_pos = np.full(n_genes, -1, dtype=np.int32)
        rank_pos[ranked_idx] = np.arange(top_k, dtype=np.int32)

        # 4) 对每个 cell type 计算 AUC-like 分数
        spot_scores: Dict[str, float] = {}
        for ct, idxs in marker_index.items():
            if idxs.size == 0:
                spot_scores[ct] = 0.0
                continue

            pos = rank_pos[idxs]
            pos = pos[pos >= 0]  # 只保留真正落在 top_k 内的 marker
            if pos.size == 0:
                spot_scores[ct] = 0.0
                continue

            # 最高表达基因 pos=0，贡献最大；越靠后贡献越小
            contrib = top_k - pos.astype(float)  # 1..top_k
            auc_like = contrib.sum() / (top_k * float(len(idxs)))
            spot_scores[ct] = float(auc_like)

        results[spot] = spot_scores

    return results


# --------------------------------------------------------------------------- #
# PAGE
# --------------------------------------------------------------------------- #

def compute_page_scores(
    X: np.ndarray,
    obs_names: Iterable[str],
    marker_index: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    PAGE z-score：对每个 cell type，在每个 spot 上计算：
        z = (mean(expr_markers) - mean(expr_all_genes)) / std(expr_all_genes)
    """
    X = np.asarray(X)
    n_spots, _ = X.shape
    obs_names = list(obs_names)

    mean_all = X.mean(axis=1)
    std_all = X.std(axis=1) + 1e-8

    results: Dict[str, Dict[str, float]] = {}

    for i, spot in enumerate(obs_names):
        row = X[i, :]
        mu = float(mean_all[i])
        sigma = float(std_all[i])

        ct_scores: Dict[str, float] = {}
        for ct, idxs in marker_index.items():
            if idxs.size == 0:
                ct_scores[ct] = 0.0
                continue
            vals = row[idxs]
            if vals.size == 0:
                ct_scores[ct] = 0.0
                continue
            mean_marker = float(vals.mean())
            z = (mean_marker - mu) / sigma
            ct_scores[ct] = z

        results[spot] = ct_scores

    return results


# --------------------------------------------------------------------------- #
# ssGSEA
# --------------------------------------------------------------------------- #

def compute_ssgsea_scores(
    X: np.ndarray,
    obs_names: Iterable[str],
    gene_names: np.ndarray,
    markers: Mapping[str, Iterable[str]],
    n_threads: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    使用 gseapy.ssgsea 进行 ssGSEA 计算。

    - 先对每个 spot 做 rank transform（表达越高 rank 越大）
    - 然后构成 gene x spot 的 DataFrame 交给 gseapy
    - 结果是 NES 分数（Name=spot, Term=cell_type）
    """
    if gp is None:
        raise ImportError("gseapy is required for ssGSEA scoring but not installed.")

    X = np.asarray(X)
    n_obs, n_genes = X.shape
    obs_names = list(obs_names)

    # Rank per spot so high expression gets high rank (noise reduction).
    order = np.argsort(X, axis=1)
    ranks = np.empty_like(X, dtype=float)
    ranks[np.arange(n_obs)[:, None], order] = np.arange(1, n_genes + 1, dtype=float)

    expr_df = pd.DataFrame(
        ranks.T,
        index=list(gene_names),
        columns=obs_names,
    )

    enr = gp.ssgsea(
        data=expr_df,
        gene_sets=markers,
        outdir=None,
        sample_norm_method=None,
        min_size=1,
        max_size=5000,
        threads=int(n_threads),
        verbose=False,
    )

    df = enr.res2d  # tidy: Name(sample) x Term(cell type)
    pivot = df.pivot(index="Name", columns="Term", values="NES")
    pivot = pivot.reindex(index=obs_names)
    pivot = pivot.fillna(0.0).infer_objects(copy=False)

    return {
        spot: {ct: float(pivot.loc[spot, ct]) for ct in pivot.columns}
        for spot in pivot.index
    }


# --------------------------------------------------------------------------- #
# Fusion + presence
# --------------------------------------------------------------------------- #

def minmax_normalize(score_dict: Mapping[str, float]) -> Dict[str, float]:
    """Min-max normalize per spot over cell types."""
    if not score_dict:
        return {}
    vals = np.array(list(score_dict.values()), dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if np.isclose(vmin, vmax):
        return {ct: 0.0 for ct in score_dict.keys()}
    norm = (vals - vmin) / (vmax - vmin)
    return {ct: float(v) for ct, v in zip(score_dict.keys(), norm)}


def softmax(score_dict: Mapping[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not score_dict:
        return {}
    vals = np.array(list(score_dict.values()), dtype=float) / max(temperature, 1e-8)
    vals = vals - vals.max()
    exps = np.exp(vals)
    probs = exps / (exps.sum() + 1e-12)
    return {ct: float(p) for ct, p in zip(score_dict.keys(), probs)}


def _method_quality(norm_scores: Mapping[str, float]) -> float:
    """
    评估某个方法在当前 spot 上的“质量”，返回 [0,1]：

      - frac_nonzero: 有多少比例的 cell type 得到了非零分
      - peakedness = 1 - entropy:
            如果分布很平（大家差不多高），信息量小；
            如果有明显几个 cell type 高，其它低，则 entropy 小、peakedness 高。
    """
    if not norm_scores:
        return 0.0

    vals = np.array(list(norm_scores.values()), dtype=float)
    if vals.size == 0:
        return 0.0

    frac_nonzero = float((vals > 0).mean())

    s = float(vals.sum())
    if s <= 0.0:
        peaked = 0.0
    else:
        p = vals / s
        ent = float(-(p * np.log(p + 1e-12)).sum())
        ent_norm = ent / (np.log(len(p)) + 1e-12)  # in [0,1]
        peaked = 1.0 - ent_norm

    return float(0.5 * frac_nonzero + 0.5 * peaked)


def presence_likelihood_across_spots(
    all_fused_raw: Mapping[str, Mapping[str, float]],
    sparsity_quantile: float = 0.7,
    sharpness: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Adaptive presence-like scores across spots.

    思路：
      - 对所有 spot 的 fused_raw 统计每个 cell type 的得分分布；
      - 对每个 cell type 计算一个全局 quantile 阈值 q_ct；
      - 某个 spot 上的得分 v_ct > q_ct 时：
            presence_ct = ((v_ct - q_ct) / (max_ct - q_ct))^sharpness；
        否则为 0；
      - 这样实现了 per-cell-type adaptive quantile，而不是 per-spot 同一个 q。
    """
    if not all_fused_raw:
        return {}

    q = float(np.clip(sparsity_quantile, 0.0, 1.0))

    # 1) 汇总每个 cell type 在所有 spots 上的得分
    ct_values: Dict[str, list] = defaultdict(list)
    for fused in all_fused_raw.values():
        for ct, v in fused.items():
            ct_values[ct].append(float(v))

    per_ct_q = {}
    per_ct_max = {}
    for ct, vals in ct_values.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            continue
        per_ct_q[ct] = float(np.quantile(arr, q))
        per_ct_max[ct] = float(arr.max())

    # 2) 对每个 spot 计算 presence
    presence_all: Dict[str, Dict[str, float]] = {}
    for spot, fused in all_fused_raw.items():
        if not fused:
            presence_all[spot] = {}
            continue
        spot_presence = {}
        for ct, v in fused.items():
            vmax = per_ct_max.get(ct, 0.0)
            vq = per_ct_q.get(ct, 0.0)
            if vmax <= 0.0 or np.isclose(vmax, vq):
                score = 0.0
            else:
                base = max(0.0, float(v) - vq) / (vmax - vq + 1e-8)
                score = base ** sharpness if sharpness != 1.0 else base
            if score > 0.0:
                spot_presence[ct] = float(score)
        presence_all[spot] = spot_presence

    return presence_all


def fuse_scores(
    aucell_scores: Mapping[str, Mapping[str, float]],
    ssgsea_scores: Mapping[str, Mapping[str, float]],
    page_scores: Mapping[str, Mapping[str, float]],
    w_aucell: float = 0.4,
    w_ssgsea: float = 0.35,
    w_page: float = 0.25,
    sparsity_quantile: float = 0.6,
    sharpness: float = 1.5,
    min_presence: float = 0.1,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    """
    Fusion logic:
      1) Min-max normalize per method per spot.
      2) Compute dynamic weights per spot = base_weight * (0.5 + 0.5*quality),
         where quality reflects non-zero coverage + distribution peakedness.
      3) Weighted sum → fused_raw；再跨 spot 做 per-cell-type adaptive presence。
    返回 (fused_results, weight_log).
    """
    spots = set(aucell_scores.keys()) | set(ssgsea_scores.keys()) | set(page_scores.keys())
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    weight_log = {"aucell": [], "ssgsea": [], "page": []}
    raw_scores_by_spot: Dict[str, Dict[str, float]] = {}

    for spot in spots:
        a_raw = aucell_scores.get(spot, {})
        s_raw = ssgsea_scores.get(spot, {})
        p_raw = page_scores.get(spot, {})

        # Normalize per method
        a_norm = minmax_normalize(a_raw) if a_raw else {}
        s_norm = minmax_normalize(s_raw) if s_raw else {}
        p_norm = minmax_normalize(p_raw) if p_raw else {}

        # Dynamic weights (0 if method missing)
        wA = w_aucell * (0.5 + 0.5 * _method_quality(a_norm)) if a_norm else 0.0
        wS = w_ssgsea * (0.5 + 0.5 * _method_quality(s_norm)) if s_norm else 0.0
        wP = w_page * (0.5 + 0.5 * _method_quality(p_norm)) if p_norm else 0.0

        weight_log["aucell"].append(wA)
        weight_log["ssgsea"].append(wS)
        weight_log["page"].append(wP)

        active = [w for w in (wA, wS, wP) if w > 0]
        if not active:
            raw_scores_by_spot[spot] = {}
            continue

        w_sum = sum(active)
        wA = wA / w_sum if wA > 0 else 0.0
        wS = wS / w_sum if wS > 0 else 0.0
        wP = wP / w_sum if wP > 0 else 0.0

        cts = set(a_norm.keys()) | set(s_norm.keys()) | set(p_norm.keys())
        fused_raw = {}
        for ct in cts:
            fused_raw[ct] = float(
                wA * a_norm.get(ct, 0.0)
                + wS * s_norm.get(ct, 0.0)
                + wP * p_norm.get(ct, 0.0)
            )

        raw_scores_by_spot[spot] = fused_raw

    # 第二步：跨 spot 做 per-cell-type adaptive presence
    presence_all = presence_likelihood_across_spots(
        raw_scores_by_spot,
        sparsity_quantile=sparsity_quantile,
        sharpness=sharpness,
    )

    for spot, presence in presence_all.items():
        presence_filtered = {ct: v for ct, v in presence.items() if v >= min_presence}
        results[spot] = {
            "presence": presence_filtered,
        }

    return results, weight_log


# --------------------------------------------------------------------------- #
# QC helpers
# --------------------------------------------------------------------------- #

def summarize_weights(weight_log: Mapping[str, Iterable[float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for method, vals in weight_log.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            summary[method] = {"mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        summary[method] = {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def method_nonzero_stats(scores: Mapping[str, Mapping[str, float]]):
    if not scores:
        return {"spots": 0, "nonzero_fraction_mean": 0.0}
    nonzero = []
    for spot_scores in scores.values():
        if not spot_scores:
            nonzero.append(0.0)
            continue
        vals = np.array(list(spot_scores.values()), dtype=float)
        nonzero.append(float((vals != 0).mean()))
    arr = np.array(nonzero, dtype=float)
    return {"spots": len(scores), "nonzero_fraction_mean": float(arr.mean())}


# --------------------------------------------------------------------------- #
# Main & CLI
# --------------------------------------------------------------------------- #

def parse_args():
    cancer_type = "READ"
    id = "ZEN49"

    adata_path = os.path.join("/home/wuyang/hest-data/process/Homo sapiens", cancer_type, f"{id}_adata.h5ad")
    markers_path = "./global_ref/global_markers.json"
    output_path = f"../../output/HEST/Homo sapiens/{cancer_type}/{id}/cell_types"


    parser = argparse.ArgumentParser(
        description="Compute AUCell, PAGE, ssGSEA scores and fused presence likelihoods with QC."
    )

    parser.add_argument("--cancer_type", default=cancer_type,
                        help="Cancer type code, e.g. COAD / READ / IDC / PAAD.")
    parser.add_argument("--sample_id", default=id,
                        help="Sample ID used in default paths (e.g. TENX89).")
    parser.add_argument("--adata", default=adata_path,
                        help="Input .h5ad file. If not set, built from cancer_type & sample_id.")
    parser.add_argument("--markers", default=markers_path,
                        help="Organ-aware marker JSON (organ -> cell_type -> genes) from build_global_markers.py")
    parser.add_argument("--output_dir", default=output_path,
                        help="Directory for JSON outputs. If not set, built from cancer_type & sample_id.")

    parser.add_argument(
        "--marker_organs",
        type=str,
        default=None,
        help="Comma-separated organ names to select from marker JSON, "
             "e.g. 'GI_tract,Pancreas'. If omitted, auto-selected from cancer_type.",
    )
    parser.add_argument(
        "--omit_common_organs",
        action="store_true",
        help="If set, do NOT auto-add Immune_system/Vasculature/Connective_tissue.",
    )

    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.1,
        help="Top fraction of genes per spot for AUCell (rank-based).",
    )
    parser.add_argument("--ssgsea_threads", type=int, default=40, help="Threads for ssGSEA.")

    parser.add_argument("--w_aucell", type=float, default=0.4, help="Base weight for AUCell in fusion.")
    parser.add_argument("--w_ssgsea", type=float, default=0.35, help="Base weight for ssGSEA in fusion.")
    parser.add_argument("--w_page", type=float, default=0.25, help="Base weight for PAGE in fusion.")

    parser.add_argument(
        "--presence_quantile",
        type=float,
        default=0.8,
        help="Per-cell-type quantile threshold (0-1) across spots; scores below are shrunk to 0.",
    )
    parser.add_argument(
        "--presence_sharpness",
        type=float,
        default=1.0,
        help="Exponent (>1 makes presence scores peakier) applied after thresholding.",
    )
    parser.add_argument(
        "--presence_min_value",
        type=float,
        default=0.1,
        help="Minimum presence value kept in output JSON (after fusion).",
    )
    parser.add_argument(
        "--qc_json",
        default=output_path+"/qc_report.json",
        help="Optional QC report JSON path. If not set, saved in output_dir/qc_report.json",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading AnnData: {args.adata}")
    adata = sc.read_h5ad(args.adata)

    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = np.asarray(X)
    n_spots, n_genes = X.shape
    print(f"[INFO] AnnData loaded: {n_spots} spots x {n_genes} genes")

    gene_names = np.array([str(g).upper() for g in adata.var_names])
    obs_names = np.array(adata.obs_names)

    print(f"[INFO] Loading markers: {args.markers}")
    markers = load_markers(
        args.markers,
        cancer_type=args.cancer_type,
        marker_organs=args.marker_organs,
        include_common_organs=not args.omit_common_organs,
    )
    marker_index = build_marker_index(markers, gene_names)

    coverage = compute_marker_coverage(markers, gene_names)
    avg_cov = np.mean([v["ratio"] for v in coverage.values()]) if coverage else 0.0
    print(f"[QC] Mean marker coverage vs data genes: {avg_cov:.3f}")

    # ------------------------------------------------------------------ #
    # AUCell
    # ------------------------------------------------------------------ #
    print("[INFO] Running AUCell (rank-based)...")
    aucell_scores = compute_aucell_scores(
        X,
        obs_names,
        gene_names,
        marker_index,
        top_frac=args.top_frac,
    )
    aucell_path = os.path.join(args.output_dir, "aucell_results.json")
    with open(aucell_path, "w") as f:
        json.dump(aucell_scores, f, indent=2)
    print(f"[INFO] Saved AUCell results → {aucell_path}")

    # ------------------------------------------------------------------ #
    # PAGE
    # ------------------------------------------------------------------ #
    print("[INFO] Running PAGE...")
    page_scores = compute_page_scores(X, obs_names, marker_index)
    page_path = os.path.join(args.output_dir, "page_results.json")
    with open(page_path, "w") as f:
        json.dump(page_scores, f, indent=2)
    print(f"[INFO] Saved PAGE results → {page_path}")

    # ------------------------------------------------------------------ #
    # ssGSEA
    # ------------------------------------------------------------------ #
    print("[INFO] Running ssGSEA...")
    ssgsea_scores = compute_ssgsea_scores(
        X,
        obs_names,
        gene_names,
        markers,
        n_threads=args.ssgsea_threads,
    )
    ssgsea_path = os.path.join(args.output_dir, "ssgsea_results.json")
    with open(ssgsea_path, "w") as f:
        json.dump(ssgsea_scores, f, indent=2)
    print(f"[INFO] Saved ssGSEA results → {ssgsea_path}")

    # ------------------------------------------------------------------ #
    # Fusion
    # ------------------------------------------------------------------ #
    print("[INFO] Fusing scores...")
    fused, weight_log = fuse_scores(
        aucell_scores,
        ssgsea_scores,
        page_scores,
        w_aucell=args.w_aucell,
        w_ssgsea=args.w_ssgsea,
        w_page=args.w_page,
        sparsity_quantile=args.presence_quantile,
        sharpness=args.presence_sharpness,
        min_presence=args.presence_min_value,
    )

    fused_path = os.path.join(args.output_dir, "mixture_fused.json")
    with open(fused_path, "w") as f:
        json.dump(fused, f, indent=2)
    print(f"[INFO] Saved fused results → {fused_path}")

    # ------------------------------------------------------------------ #
    # QC report
    # ------------------------------------------------------------------ #
    if args.qc_json:
        qc_report = {
            "marker_coverage": coverage,
            "coverage_mean": float(avg_cov),
            "method_nonzero": {
                "aucell": method_nonzero_stats(aucell_scores),
                "ssgsea": method_nonzero_stats(ssgsea_scores),
                "page": method_nonzero_stats(page_scores),
            },
            "fusion_weights": summarize_weights(weight_log),
            "fusion_base_weights": {
                "aucell": args.w_aucell,
                "ssgsea": args.w_ssgsea,
                "page": args.w_page,
            },
        }
        os.makedirs(os.path.dirname(args.qc_json) or ".", exist_ok=True)
        with open(args.qc_json, "w") as f:
            json.dump(qc_report, f, indent=2)
        print(f"[INFO] Saved QC report → {args.qc_json}")


if __name__ == "__main__":
    main()
