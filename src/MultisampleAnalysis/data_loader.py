"""数据加载和去噪处理"""


import os
import json
import ast
import re
import numpy as np
from collections import defaultdict

from analysis_config import AnalysisConfig


class DataDenoiser:
    """数据去噪处理器"""

    @staticmethod
    def remove_noise_genes(gene_list):
        """移除噪声基因"""
        if not gene_list:
            return []

        cleaned_genes = []
        noise_count = 0

        for gene in gene_list:
            is_noise = False
            for category, noise_patterns in AnalysisConfig.noise_genes.items():
                for pattern in noise_patterns:
                    if gene.startswith(pattern):
                        is_noise = True
                        noise_count += 1
                        break
                if is_noise:
                    break

            if not is_noise:
                cleaned_genes.append(gene)

        print(f"去噪: {len(gene_list)} -> {len(cleaned_genes)} 个基因 (移除 {noise_count} 个噪声基因)")
        return cleaned_genes

    @staticmethod
    def denoise_expression_data(expression_dict):
        """去噪表达数据"""
        if not expression_dict:
            return {}

        cleaned_expr = {}
        noise_count = 0

        for gene, expr in expression_dict.items():
            is_noise = False
            for category, noise_patterns in AnalysisConfig.noise_genes.items():
                for pattern in noise_patterns:
                    if gene.startswith(pattern):
                        is_noise = True
                        noise_count += 1
                        break
                if is_noise:
                    break

            if not is_noise:
                cleaned_expr[gene] = expr

        print(f"表达数据去噪: {len(expression_dict)} -> {len(cleaned_expr)} 个基因")
        return cleaned_expr


def normalize_expression_dict(expr_dict):
    """
    新增：对表达字典做0-1归一化（跨样本可比）
    expr_dict: {gene: expr_value, ...}
    """
    if not expr_dict:
        return {}

    # 提取表达值并归一化
    genes = list(expr_dict.keys())
    values = np.array(list(expr_dict.values()))
    if values.max() == values.min():
        return {g: 0.0 for g in genes}  # 避免除0

    norm_values = (values - values.min()) / (values.max() - values.min())
    return dict(zip(genes, norm_values))


def normalize_cell_type_data(cell_type_data):
    """
    新增：细胞类型数据标准化（比例归一化+命名统一）
    cell_type_data: 原始cell_type_inference_results.json加载的字典
    """
    normalized_cell_type = {}

    # 遍历每个spot的细胞类型数据
    for spot_id, spot_data in cell_type_data.items():
        if "confidence" not in spot_data:
            normalized_cell_type[spot_id] = {}
            continue

        # 提取细胞类型置信度并计算比例（总和=1）
        cell_conf = spot_data["confidence"]
        total = sum(cell_conf.values())
        if total == 0:
            cell_proportion = {k: 0.0 for k in cell_conf.keys()}
        else:
            cell_proportion = {k: v / total for k, v in cell_conf.items()}

        # 统一细胞类型命名（消除跨样本差异）
        standardized_proportion = {}
        for cell_type, prop in cell_proportion.items():
            # 标准化命名：CD8+T → CD8_T，T cell → T_cell等
            std_type = cell_type.replace("+", "_").replace(" ", "_").replace("-", "_").upper()
            standardized_proportion[std_type] = prop

        normalized_cell_type[spot_id] = standardized_proportion

    return normalized_cell_type


def standardize_enrichment_terms(enrichment_list):
    """
    新增：富集通路术语标准化（跨样本可比）
    enrichment_list: cluster['core_enrichment'] 原始列表
    """
    standardized = []
    for item in enrichment_list:
        if not isinstance(item, dict) or "term" not in item:
            continue

        # 标准化术语（移除GO编号、统一中英文/格式）
        term = item["term"]
        # 移除GO:XXXXXXX 前缀
        term = re.sub(r"GO:\d+ ", "", term)
        # 统一关键术语
        term = term.replace("apoptosis", "细胞凋亡")
        term = term.replace("cell cycle", "细胞周期")
        term = term.replace("immune response", "免疫应答")
        term = term.replace("angiogenesis", "血管生成")

        # 保留原有字段，仅替换标准化后的术语
        standardized_item = item.copy()
        standardized_item["term"] = term
        # 保留原有adj_pvalue（已<0.01，不筛选）
        standardized.append(standardized_item)

    return standardized


def extract_cancer_type_from_path(path):
    """从路径中动态提取癌症类型"""
    path_parts = path.split('/')

    for part in path_parts:
        if part in AnalysisConfig.cancer_type_mapping:
            return part, AnalysisConfig.cancer_type_mapping[part]

    for part in path_parts:
        if len(part) == 4 and part.isupper():
            return part, part

    return "unknown", "未知癌症类型"


def load_multi_sample_data(sample_paths):
    """加载多个样本的聚类结果、细胞类型数据和预计算富集通路"""
    sample_data = []

    for idx, path in enumerate(sample_paths):
        print(f"正在加载样本 {idx}: {path}")

        if not os.path.exists(path):
            print(f"警告: 路径不存在: {path}")
            continue

        # 构建文件路径
        biology_path = os.path.join(path, "tumor_analysis_results", "tables", "community_detection_statistics.json")
        gene_sets_path = os.path.join(path, "spot_domain_features.json")
        cell_type_path = os.path.join(path, "cell_type_inference", "cell_type_inference_results.json")

        # 检查所有必要文件是否存在
        missing_files = []
        if not os.path.exists(biology_path):
            missing_files.append(biology_path)
        if not os.path.exists(gene_sets_path):
            missing_files.append(gene_sets_path)
        if not os.path.exists(cell_type_path):
            missing_files.append(cell_type_path)
        if missing_files:
            print(f"警告: 以下文件不存在，跳过样本: {missing_files}")
            continue

        try:
            # 加载数据
            with open(biology_path, 'r') as f:
                biology_data = json.load(f)
            with open(gene_sets_path, 'r') as f:
                domain_features = json.load(f)
            with open(cell_type_path, 'r') as f:
                cell_type_data = json.load(f)

            # 提取聚类数据
            raw_clusters = biology_data.get("cluster_biology", [])
            if not raw_clusters:
                print(f"警告: 样本 {idx} 中没有有效的聚类数据")
                continue

            # 清理和转换聚类数据
            cleaned_clusters = []
            for cluster in raw_clusters:
                # 处理core_genes（确保为列表）
                if 'core_genes' in cluster and isinstance(cluster['core_genes'], str):
                    try:
                        cluster['core_genes'] = ast.literal_eval(cluster['core_genes'])
                    except:
                        cluster['core_genes'] = []
                if not isinstance(cluster.get('core_genes', []), list):
                    cluster['core_genes'] = []

                # 处理domains（确保为列表）
                if 'domains' in cluster and isinstance(cluster['domains'], str):
                    try:
                        cluster['domains'] = ast.literal_eval(cluster['domains'])
                    except:
                        cluster['domains'] = []
                if not isinstance(cluster.get('domains', []), list):
                    cluster['domains'] = []

                # 保留预计算的富集通路信息
                cluster['core_enrichment'] = standardize_enrichment_terms(cluster.get('core_enrichment', []))

                cleaned_clusters.append(cluster)


            # 对每个domain的表达数据做0-1归一化
            normalized_domain_features = {}
            for domain_id, domain_info in domain_features.items():
                if "gene_avg_expr_domain" in domain_info:
                    # 先去噪，再归一化
                    denoised_expr = DataDenoiser.denoise_expression_data(domain_info["gene_avg_expr_domain"])
                    normalized_expr = normalize_expression_dict(denoised_expr)
                    domain_info["gene_avg_expr_norm"] = normalized_expr  # 新增归一化表达字段
                normalized_domain_features[domain_id] = domain_info

            normalized_cell_type = normalize_cell_type_data(cell_type_data)

            # 提取癌症类型
            cancer_abbr, cancer_name = extract_cancer_type_from_path(path)

            sample_data.append({
                "sample_id": f"sample_{idx}",
                "cancer_type": cancer_abbr,
                "cancer_name": cancer_name,
                "clusters": cleaned_clusters,
                "domain_features": domain_features,
                "domain_features_norm": normalized_domain_features,  # 归一化表达
                "cell_type_data": cell_type_data,
                "cell_type_data_norm": normalized_cell_type,  # 标准化细胞类型
                "path": path
            })
            print(f"成功加载样本 {idx} ({cancer_name}): {len(cleaned_clusters)} 个聚类")

        except Exception as e:
            print(f"加载样本 {idx} 时出错: {str(e)}")
            continue

    print(f"成功加载 {len(sample_data)} 个样本数据")
    return sample_data


def group_samples_by_cancer(sample_data):
    """按癌症类型分组样本"""
    cancer_groups = defaultdict(list)
    for sample in sample_data:
        cancer_groups[sample["cancer_type"]].append(sample)
    return cancer_groups