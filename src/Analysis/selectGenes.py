
import os
import json
import scipy
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from src.Utils.GssGeneSelector import GssGeneSelector
from src.Utils.DataProcess import get_self_data_dir, get_hest_data_dir, set_chinese_font, read_data



def select_gene_func(mk_score_df, adata, output_dir=None,
                          min_expr_threshold=0.1, min_gss_threshold=0.5,
                          concentration_threshold = 90, corr_threshold=0.4,
                          entropy_threshold=0.2, morans_i_threshold=0.3):
    # 初始化选择器
    selector = GssGeneSelector(
        adata=adata,  # AnnData对象
        gss_df=mk_score_df,  # GSS分数矩阵
        output_dir=output_dir,
        min_expr_threshold=min_expr_threshold,
        min_gss_threshold=min_gss_threshold,
        concentration_threshold=concentration_threshold,
        corr_threshold=corr_threshold,
        entropy_threshold=entropy_threshold,
        morans_i_threshold=morans_i_threshold,
    )

    # 运行全流程
    selected_genes, selected_results = selector.run_pipeline()

    # 查看结果
    print(f"最终选择 {len(selected_genes)} 个基因")

    # 可视化Top基因的空间表达
    # if output_dir:
    #     sc.settings.figdir = output_dir
    #
    # for gene in selected_genes[:-1]:
    #     sc.pl.spatial(adata, color=gene, title=gene, save=f"{gene}_calibrated.png")

    return selected_genes, selected_results


def calculate_spatial_purity(adata, genes, cluster_key='annotation_type'):
    """
    计算基因在空间聚类中的表达纯度

    参数:
        adata: AnnData对象（需包含空间聚类标签）
        genes: 待评估基因列表
        cluster_key: 空间聚类标签的列名（默认为'annotation_type'）

    返回:
        包含每个基因纯度分数的DataFrame
    """
    purity_scores = []

    # 检查聚类标签是否存在
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"AnnData中不存在聚类标签 '{cluster_key}'")

    for gene in genes:
        # 获取基因表达量（转换为一维数组）
        expr = adata[:, gene].X
        if scipy.sparse.issparse(expr):
            expr = expr.toarray().flatten()

        # 获取空间聚类标签
        clusters = adata.obs[cluster_key].values

        # 处理聚类标签为类别型数据
        if pd.api.types.is_categorical_dtype(clusters):
            clusters = clusters.codes

        # 计算每个聚类内的平均表达
        cluster_expr = {}
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            if np.sum(mask) > 0:  # 避免空聚类
                cluster_expr[cluster] = np.mean(expr[mask])

        # 计算纯度（最高表达聚类的贡献比例）
        if cluster_expr:
            max_expr = max(cluster_expr.values())
            total_expr = np.sum(list(cluster_expr.values()))
            purity = max_expr / total_expr if total_expr > 0 else 0
        else:
            purity = 0  # 无有效聚类时纯度为0

        types_num = len(adata.obs[cluster_key].values.unique())
        purity_scores.append({
            'gene': gene,
            'purity': purity,
            'max_cluster_expr': max_expr if cluster_expr else 0,
            'total_expr': total_expr,
            'total_cluster': types_num
        })

    return pd.DataFrame(purity_scores)


def tissue_type_specificity(adata, genes, tissue_type_key='annotation_type'):
    """
    验证基因在特定细胞类型中的特异性表达
    """
    # 提取指定基因的表达矩阵（细胞×基因）
    expr_matrix = adata[:, genes].X.toarray() if isinstance(adata.X, np.ndarray) else adata[:, genes].X.A

    # 构建数据框
    cell_types = adata.obs[tissue_type_key].values

    # 按组织类型和基因分组计算平均表达量
    specificity = []
    for i, gene in enumerate(genes):
        # 每个基因在所有细胞中的表达量
        gene_expr = expr_matrix[:, i]

        # 按组织类型分组计算平均表达
        type_expr = {}
        for tissue_type in np.unique(cell_types):
            # 提取该组织类型下所有细胞的表达量
            type_mask = (cell_types == tissue_type)
            type_expr[tissue_type] = np.mean(gene_expr[type_mask])

        # 转换为数组便于计算
        expr_values = np.array(list(type_expr.values()))
        tissue_types = list(type_expr.keys())

        # 计算特异性评分
        max_expr = np.max(expr_values)
        mean_expr = np.mean(expr_values)

        # 避免除以零（当所有类型表达量都为0时）
        if mean_expr < 1e-10:
            specificity_score = 0.0
        else:
            specificity_score = max_expr / mean_expr  # 核心公式：最高表达/平均表达

        # 记录最高表达的组织类型
        max_type = tissue_types[np.argmax(expr_values)]

        specificity.append({
            'gene': gene,
            'specificity_score': specificity_score,
            'max_expression': max_expr,
            'mean_expression': mean_expr,
            'max_expression_type': max_type,
            'tissue_type_count': len(tissue_types)  # 参与计算的组织类型数量
        })

    return pd.DataFrame(specificity)


def plot_gene_spatial(mk_score_df, adata, gene_name,
                      output_dir=None, visual_indicators = None,
                      cmap='viridis', size=0.8, alpha=0.8,
                      background_alpha=0.7, show=True):
    """使用 Scanpy 内置方法可视化特定基因表达并叠加病理切片"""
    # 基因存在性检查
    if gene_name not in adata.var_names:
        print(f"错误: 基因 '{gene_name}' 不在数据中")
        return

    # 检查空间坐标
    if 'spatial' not in adata.obsm:
        print("错误: 缺少空间坐标信息")
        return

    # 获取基因表达值
    if visual_indicators == "GSS" and gene_name in mk_score_df.index:
        print(f"基因 '{gene_name}' 使用GSS分数数据")
        obs_column_name = f"Marker_score_{gene_name}"
        adata.obs[obs_column_name] = mk_score_df.loc[gene_name].values
        color = obs_column_name
    else:
        print(f"基因 '{gene_name}' 使用原始基因表达值")
        color = gene_name  # 直接使用 adata.var_names 中的基因

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 使用 Scanpy 的空间可视化函数
    try:
        fig = sc.pl.spatial(
            adata,
            cmap=cmap,
            color=color,
            # size=size,
            # alpha=alpha,
            alpha_img=background_alpha,
            img_key='downscaled_fullres', # 'hires'
            title=f'{gene_name}空间表达分布({visual_indicators})',
            return_fig = True,
            frameon=True,
            show=show,
        )

        fig.savefig(os.path.join(output_dir, f"{color}_calibrated.png"), dpi=400)
        plt.close(fig)

    except Exception as e:
        print(f"可视化失败: {e}")


def plot_multiple_genes(mk_score_df, adata, gene_names,
                        output_dir=None, visual_indicators = None,
                        cmap='viridis', size=0.8, alpha=0.8,
                        background_alpha=0.7, show=False):
    """
    可视化多个基因在空间上的表达分布

    参数:
    - mk_score_df: 标记分数DataFrame
    - adata: AnnData对象
    - gene_names: 要可视化的基因名称列表
    - output_dir: 图像保存目录
    - cmap: 颜色映射
    - size: 点大小
    - alpha: 透明度
    """
    for gene_name in gene_names:
        plot_gene_spatial(
            mk_score_df, adata, gene_name,
            output_dir=output_dir,
            visual_indicators=visual_indicators,
            cmap=cmap,
            size=size,
            alpha=alpha,
            background_alpha=background_alpha,
            show=show  # 批量绘制时不显示，提高效率
        )

    print(f"已完成 {len(gene_names)} 个基因的可视化")


def analyze_single_sample(output_dir, feather_path, h5ad_path):
    """处理单个样本：基因筛选、可视化、评估"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    mk_score_df, adata = read_data(feather_path, h5ad_path)

    # 2. 基因筛选
    selected_genes, _ = select_gene_func(
        mk_score_df, adata,
        output_dir=output_dir,
        min_expr_threshold=0.001,
        min_gss_threshold=0.1,
        concentration_threshold=70,  # 表达离散度和集中性阈值 [0, 100]
        corr_threshold=0.6,  # GSS-表达量相关性阈值 [0, 1]
        entropy_threshold=0.6,  # GSS的标准化信息熵阈值 [0, 1]
        morans_i_threshold=0.6,  # 空间自相关性阈值 [0, 1]
    )
    print(selected_genes)

    # 3. 可视化空间分布
    plot_multiple_genes(
        mk_score_df, adata, selected_genes,
        output_dir=output_dir,
        visual_indicators="Expr",  # ["GSS", "Expr"]
        cmap='viridis',
        size=1.0,
        alpha=0.6
    )

    # 4. 评估：空间纯度、组织特异性等
    # # 各基因在空间聚类中的表达纯度
    # purity_scores = calculate_spatial_purity(adata, selected_genes)
    # purity_scores.to_csv(output_dir + "purity_scores.csv", index=False, sep='\t')
    # print(f"各基因在空间聚类中的表达纯度结果已保存!!!")
    #
    # # 基因在特定细胞类型中的特异性表达
    # specificity = tissue_type_specificity(adata, selected_genes)
    # specificity.to_csv(output_dir + "specificity.csv", index=False, sep='\t')
    # print(f"基因在特定细胞类型中的特异性表达结果已保存!!!")


def batch_analysis(select_n, json_path, data_dir, output_root):
    """
    批量分析JSON中所有样本
    :param select_n: 至多分析的样本数量
    :param json_path: JSON文件路径（存储物种、癌症、样本映射）
    :param data_dir: 数据存放路径
    :param output_root: 所有结果的根输出目录
    """
    # 1. 读取JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2. 遍历物种
    n = 5 # 每种癌症选取的样本数
    method = 'selectGenes'
    species = "Homo sapiens"
    species_data = data.get(species, {})
    cancer_types = species_data.get("cancer_types", {})

    # 3. 遍历每个癌症类型
    i = 1
    for cancer_type, sample_ids in cancer_types.items():
        # 4. 遍历该癌症类型下的每个样本
        ids_to_query = sample_ids[: min(n, len(sample_ids))]
        for id in ids_to_query:
            # 构建文件路径
            if i <= select_n:
                output_dir = os.path.join(output_root, species, cancer_type, id, method)
                feather_path = os.path.join(data_dir, species, cancer_type, id,
                                            f"latent_to_gene/{id}_gene_marker_score.feather")
                h5ad_path = os.path.join(data_dir, species, cancer_type, id,
                                         f"find_latent_representations/{id}_add_latent.h5ad")

                # 5. 调用单样本分析函数
                print(f"------------开始分析：物种={species}, 癌症类型={cancer_type}, 样本={id}------------")
                analyze_single_sample(output_dir, feather_path, h5ad_path)
                print(f"完成分析：{output_dir}")
            else:
                if i > select_n:
                    break

            i += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="统计学方式进行批量分析")
    parser.add_argument("--select-n", type=str,
                        default=29,
                        help="至多分析样本")
    parser.add_argument("--json", type=str,
                        default="/Users/wuyang/Documents/MyPaper/3/dataset/cancer_samples.json",
                        help="样本映射JSON路径")
    parser.add_argument("--data-dir", type=str,
                        default="/Users/wuyang/Documents/MyPaper/3/dataset/HEST-data/",
                        help="数据根目录")
    parser.add_argument("--output-root", type=str,
                        default="/Users/wuyang/Documents/MyPaper/3/gsVis/output/HEST",
                        help="结果输出根目录")

    # 解析命令行参数
    args = parser.parse_args()
    # 设置中文字体
    set_chinese_font()

    # 进行批量分析
    batch_analysis(
        select_n = args.select_n,
        json_path = args.json,
        data_dir = args.data_dir,
        output_root = args.output_root
    )
