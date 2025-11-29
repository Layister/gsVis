import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from src.Utils.DataProcess import read_data
from typing import Dict, List, Tuple, Any

warnings.filterwarnings('ignore')


def load_cell_type_markers_from_excel(file_path: str) -> Dict[str, List[str]]:
    """从Excel加载细胞类型标记基因"""
    markers = {}
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if 'cell_name' not in df.columns or 'Symbol' not in df.columns:
            continue  # 跳过缺少必要列的工作表
        # 去重并过滤空值
        cell_markers = df.dropna(subset=['cell_name', 'Symbol']).groupby('cell_name')['Symbol'].unique()
        for cell_type, genes in cell_markers.items():
            markers[cell_type] = list(set(genes))  # 确保基因唯一
    return markers


def gsea_enrichment(
        feature_genes: List[str],
        cell_type_markers: Dict[str, List[str]],
        background_genes: List[str]
) -> Dict[str, Tuple[float, int, int]]:
    """执行超几何检验，返回每个细胞类型的富集结果（p值、重叠基因数、总标记基因数）"""
    M = len(background_genes)  # 背景基因总数
    background_set = set(background_genes)
    feature_set = set(feature_genes)
    enrichment_results = {}

    for cell_type, markers in cell_type_markers.items():
        # 仅保留在背景基因中的标记基因（避免无意义的匹配）
        valid_markers = [g for g in markers if g in background_set]
        n = len(valid_markers)  # 有效标记基因数
        if n == 0:
            continue  # 无有效标记基因，跳过

        N = len(feature_genes)  # 特征基因总数
        k = len(feature_set & set(valid_markers))  # 重叠基因数

        # 超几何检验（下尾检验：P(X >= k)）
        p_value = hypergeom.sf(k - 1, M, n, N)  # sf = 1 - cdf
        enrichment_results[cell_type] = (p_value, k, n)

    return enrichment_results


def calculate_confidence_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """将富集得分转换为置信度（概率分布，使用softmax归一化）"""
    if not scores:
        return {}
    # 提取得分并防止数值溢出（减去最大值）
    score_values = np.array(list(scores.values()))
    exp_scores = np.exp(score_values - np.max(score_values))
    confidences = exp_scores / np.sum(exp_scores)
    return {cell_type: float(conf) for cell_type, conf in zip(scores.keys(), confidences)}


def infer_domain_cell_types(
        domain_features: Dict[str, Any],  # 微域特征数据（来自JSON）
        cell_type_markers: Dict[str, List[str]],
        background_genes: List[str],
        fdr_threshold: float = 0.05,
        top_k: int = 5  # 每个微域保留的top细胞类型数
) -> Dict[str, Dict[str, Any]]:
    """
    细胞类型推断：结合表达倍数变化计算富集得分，输出置信度
    """
    results = {}
    background_set = set(background_genes)

    for domain_id, domain_data in domain_features.items():
        feature_genes = domain_data['feature_genes']
        # 从数据中直接获取该微域的基因表达倍数变化（已预处理）
        fold_changes = domain_data['fold_changes']  # 格式：{gene: log2(FC)}

        # 1. 计算富集结果（p值、重叠基因数等）
        enrichment = gsea_enrichment(
            feature_genes=feature_genes,
            cell_type_markers=cell_type_markers,
            background_genes=background_genes
        )

        if not enrichment:
            results[domain_id] = {'inferred_types': [], 'confidence': {}, 'details': {}}
            continue

        # 2. 整理富集结果为DataFrame，计算FDR校正
        enrich_df = pd.DataFrame(
            enrichment.values(),
            index=enrichment.keys(),
            columns=['p_value', 'overlap_genes', 'total_markers']
        )
        # FDR校正（Benjamini-Hochberg方法）
        enrich_df['fdr'] = multipletests(enrich_df['p_value'], method='fdr_bh')[1]

        # 3. 计算重叠基因的平均表达倍数变化（核心改进）
        enrich_df['avg_fold_change'] = 0.0
        for cell_type in enrich_df.index:
            # 获取该细胞类型的有效标记基因
            valid_markers = [g for g in cell_type_markers[cell_type] if g in background_set]
            # 计算与特征基因的重叠
            overlap_genes = list(set(feature_genes) & set(valid_markers))
            if overlap_genes:
                # 提取重叠基因的fold change（过滤不在fold_changes中的基因）
                fc_values = [fold_changes[g] for g in overlap_genes if g in fold_changes]
                if fc_values:
                    enrich_df.loc[cell_type, 'avg_fold_change'] = np.mean(fc_values)

        # 4. 改进的富集得分：整合重叠比例、FDR和表达倍数（核心公式）
        # 公式：(重叠比例) * (tanh(平均FC) + 1) * (-log10(FDR))
        # 注：+1避免负FC降低权重，突出高特异性基因（FC>0）的贡献
        enrich_df['enrichment_score'] = (
                (enrich_df['overlap_genes'] / np.sqrt(enrich_df['total_markers'])) *  # 重叠比例
                (np.tanh(enrich_df['avg_fold_change']) + 1) *  # 表达特异性权重
                (-np.log10(enrich_df['fdr'] + 1e-10))  # 统计显著性
        )

        # 5. 筛选显著结果并排序
        significant = enrich_df[enrich_df['fdr'] < fdr_threshold].sort_values(
            by='enrichment_score', ascending=False
        )

        # 6. 计算置信度（多细胞类型概率分布）
        top_types = significant.head(top_k)
        confidence = calculate_confidence_scores(
            top_types['enrichment_score'].to_dict()
        )

        # 7. 保存结果
        results[domain_id] = {
            'inferred_types': top_types.index.tolist(),  # 推断的细胞类型（按得分排序）
            'confidence': confidence,  # 每种类型的置信度（概率）
            'details': {
                cell_type: {
                    'fdr': float(top_types.loc[cell_type, 'fdr']),
                    'overlap_genes': int(top_types.loc[cell_type, 'overlap_genes']),
                    'avg_fold_change': float(top_types.loc[cell_type, 'avg_fold_change']),
                    'enrichment_score': float(top_types.loc[cell_type, 'enrichment_score'])
                } for cell_type in top_types.index
            }
        }

    return results


def visualize_cell_type_confidence(
        inference_results: Dict[str, Dict[str, Any]],
        output_dir: str,
        top_n_cell_types: int = 10
) -> None:
    """可视化细胞类型的置信度分布（改进版：展示平均置信度）"""
    # 统计所有微域中细胞类型的平均置信度
    cell_type_confidence = {}
    for domain_res in inference_results.values():
        for cell_type, conf in domain_res['confidence'].items():
            if cell_type not in cell_type_confidence:
                cell_type_confidence[cell_type] = []
            cell_type_confidence[cell_type].append(conf)

    # 计算平均置信度并排序
    if cell_type_confidence:
        avg_conf = {
            ct: np.mean(confs)
            for ct, confs in cell_type_confidence.items()
        }
        top_cell_types = sorted(avg_conf.items(), key=lambda x: x[1], reverse=True)[:top_n_cell_types]
        cell_types, confidences = zip(*top_cell_types) if top_cell_types else ([], [])

        # 绘制条形图
        plt.figure(figsize=(10, 6))
        plt.barh(cell_types, confidences, color='skyblue')
        plt.xlabel('Average Confidence Across Domains')
        plt.title(f'Top {top_n_cell_types} Cell Types by Average Confidence')
        plt.gca().invert_yaxis()  # 最高置信度在顶部
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cell_type_confidence.png'), dpi=300)
        plt.close()


def main():
    data_dir = "/home/wuyang/hest-data/process/"
    file_root = "../../output/HEST"

    species = "Homo sapiens"
    cancer_type = "PRAD"
    id = "TENX46"

    # 配置路径
    feather_path = os.path.join(data_dir, species, cancer_type, id, f"latent_to_gene/{id}_gene_marker_score.feather")
    h5ad_path = os.path.join(data_dir, species, cancer_type, id, f"find_latent_representations/{id}_add_latent.h5ad")
    genes_path = os.path.join(file_root, species, cancer_type, id) # 微域特征基因路径
    file_path = "/home/wuyang/hest-data/CellMarker_Cell_marker.xlsx" # 细胞类型特异性基因数据
    output_dir = genes_path + "/cell_type_inference"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载输入数据
    # 1.1 微域特征数据
    with open(genes_path + '/spot_domain_features.json', 'r') as f:
        domain_features = json.load(f)

    # 1.2 细胞类型标记基因（Excel）
    cell_markers = load_cell_type_markers_from_excel(file_path)

    # 2. 确定背景基因（所有表达基因或所有微域的特征基因集合）
    _, adata = read_data(feather_path, h5ad_path)
    background_genes = adata.var_names.tolist()
    # background_genes = list({
    #     gene for domain_data in domain_features.values()
    #     for gene in domain_data['feature_genes']
    # })

    # 3. 执行细胞类型推断
    inference_results = infer_domain_cell_types(
        domain_features=domain_features,
        cell_type_markers=cell_markers,
        background_genes=background_genes,
        fdr_threshold=0.05,
        top_k=5
    )

    # 4. 保存结果
    with open(os.path.join(output_dir, 'cell_type_inference_results.json'), 'w') as f:
        json.dump(inference_results, f, indent=2)

    # 5. 可视化置信度分布
    visualize_cell_type_confidence(
        inference_results=inference_results,
        output_dir=output_dir,
        top_n_cell_types=10
    )

    print(f"分析完成！结果保存至 {output_dir}")


if __name__ == "__main__":
    """
    注意点：
    1. 同一细胞类型在不同癌症中标记基因可能不同，后续的可以考虑加上癌症分辨
    2. 在分析过程中可以考虑分辨正常组织与癌症病变组织，在文件读取时设置选项
    """
    main()