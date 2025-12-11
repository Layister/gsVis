"""数据加载：表达归一化 + 细胞类型标准化 + 聚类结构加载"""

import os
import json
import ast
import numpy as np
from collections import defaultdict
import re

from analysis_config import AnalysisConfig


# ---------------------------------------------------------------
# 表达 0-1 归一化（跨 domain 可比）
# ---------------------------------------------------------------
def normalize_expression_dict(expr_dict):
    """对表达字典做 0–1 归一化"""
    if not expr_dict:
        return {}

    values = np.array(list(expr_dict.values()))
    if values.max() == values.min():
        return {g: 0.0 for g in expr_dict}

    norm_values = (values - values.min()) / (values.max() - values.min())
    return dict(zip(expr_dict.keys(), norm_values))


# ---------------------------------------------------------------
# 细胞类型标准化（使用置信度但不估计比例，名称标准化）
# ---------------------------------------------------------------
def normalize_cell_type_data(cell_type_data):
    """
    处理格式：
    {
        "spot_id": {
            "presence": { cell_type_name: score, ... }
        }
    }
    presence 表示存在可能性，不是比例。我们在 spot 内做归一化，
    使其成为“相对贡献”，方便后面聚合。
    """
    normalized = {}

    for spot_id, item in cell_type_data.items():
        presence = item.get("presence", {})
        if not presence:
            normalized[spot_id] = {}
            continue

        # 总和（用于归一化）
        tot = sum(presence.values())
        if tot <= 0:
            normalized[spot_id] = {}
            continue

        # 转成比例，以便 cluster 层汇总
        # 并且处理细胞类型名，去掉括号内组织信息
        std_prop = {}
        for raw_name, v in presence.items():
            base = raw_name.split("(")[0].strip()
            std_name = (
                base.replace("+", "_")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .upper()
            )
            std_prop[std_name] = std_prop.get(std_name, 0) + (v / tot)

        normalized[spot_id] = std_prop

    return normalized



# ---------------------------------------------------------------
# 富集术语清洗（用于 cluster 注释）
# ---------------------------------------------------------------
def standardize_enrichment_terms(term_list):
    """标准化 cluster['core_enrichment'] 的术语（仅用于注释）"""
    cleaned = []

    for item in term_list:
        if not isinstance(item, dict) or "term" not in item:
            continue

        term = item["term"]

        # 去掉 GO:XXXXX
        term = re.sub(r"GO:\d+\s*", "", term)

        cleaned_item = item.copy()
        cleaned_item["term"] = term
        cleaned.append(cleaned_item)

    return cleaned


# ---------------------------------------------------------------
# 路径解析癌症类型
# ---------------------------------------------------------------
def extract_cancer_type_from_path(path):
    parts = path.split("/")

    # 优先匹配配置中的 key
    for p in parts:
        if p in AnalysisConfig.cancer_type_mapping:
            return p, AnalysisConfig.cancer_type_mapping[p]

    # 次选：四位大写字符串
    for p in parts:
        if len(p) == 4 and p.isupper():
            return p, p

    return "unknown", "未知癌症"


# ---------------------------------------------------------------
# 主函数：加载 domain/cluster/cell type 数据
# ---------------------------------------------------------------
def load_multi_sample_data(sample_paths):
    sample_data = []

    for idx, path in enumerate(sample_paths):
        print(f"📂 加载样本 {idx}: {path}")

        if not os.path.exists(path):
            print(f"⚠️ 路径不存在: {path}")
            continue

        # 文件路径
        cluster_path = os.path.join(path, "tumor_analysis_results", "tables", "community_detection_statistics.json")
        domain_path = os.path.join(path, "spot_domain_features.json")
        celltype_path = os.path.join(path, "cell_types", "mixture_fused.json")

        missing = [p for p in [cluster_path, domain_path, celltype_path] if not os.path.exists(p)]
        if missing:
            print(f"⚠️ 缺少文件，跳过样本: {missing}")
            continue

        try:
            clusters_raw = json.load(open(cluster_path))
            domain_raw = json.load(open(domain_path))
            celltype_raw = json.load(open(celltype_path))

            clusters_raw = clusters_raw.get("cluster_biology", [])
            if not clusters_raw:
                print(f"⚠️ 样本 {idx} 没有有效聚类数据")
                continue

            # -------------------------
            # 1. 清洗聚类数据
            # -------------------------
            cleaned_clusters = []
            for c in clusters_raw:
                # core_genes 标准化
                cg = c.get("core_genes", [])
                if isinstance(cg, str):
                    try:
                        cg = ast.literal_eval(cg)
                    except:
                        cg = []
                if not isinstance(cg, list):
                    cg = []
                c["core_genes"] = cg

                # domains 标准化
                dms = c.get("domains", [])
                if isinstance(dms, str):
                    try:
                        dms = ast.literal_eval(dms)
                    except:
                        dms = []
                if not isinstance(dms, list):
                    dms = []
                c["domains"] = dms

                # core_enrichment 清洗
                c["core_enrichment"] = standardize_enrichment_terms(
                    c.get("core_enrichment", [])
                )

                cleaned_clusters.append(c)

            # -------------------------
            # 2. 归一化 domain 表达
            # -------------------------
            domain_norm = {}
            for did, info in domain_raw.items():
                expr = info.get("gene_avg_expr_domain", {})
                norm_expr = normalize_expression_dict(expr)
                info["gene_avg_expr_norm"] = norm_expr
                domain_norm[did] = info

            # -------------------------
            # 3. 归一化 cell types
            # -------------------------
            celltype_norm = normalize_cell_type_data(celltype_raw)

            # -------------------------
            # 4. 解析癌症类型
            # -------------------------
            cancer_abbr, cancer_name = extract_cancer_type_from_path(path)


            sample_data.append({
                "sample_id": f"sample_{idx}",
                "cancer_type": cancer_abbr,
                "cancer_name": cancer_name,
                "clusters": cleaned_clusters,
                "domain_features_norm": domain_norm,
                "cell_type_data_norm": celltype_norm,
                "path": path,
            })

            print(f"✅ 样本 {idx} ({cancer_name}) 加载成功，共 {len(cleaned_clusters)} 个聚类")

        except Exception as e:
            print(f"❌ 加载样本 {idx} 出错: {e}")
            continue

    print(f"🎉 完成加载，共 {len(sample_data)} 个样本")
    return sample_data


# ---------------------------------------------------------------
# 分组函数
# ---------------------------------------------------------------
def group_samples_by_cancer(sample_data):
    groups = defaultdict(list)
    for s in sample_data:
        groups[s["cancer_type"]].append(s)
    return groups
