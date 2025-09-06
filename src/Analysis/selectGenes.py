import os
import argparse
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from src.Utils.GssGeneSelector import GssGeneSelector
from src.Utils.DataProcess import get_self_data_dir, get_hest_data_dir, set_chinese_font, read_data


def select_gene_traditional(mk_score_df, adata, output_dir=None,
                            select_n=30, expr_weight=0.8):
    """
    基于GSS值和表达量选择 n 个基因

    参数:
    - mk_score_df: 标记分数DataFrame (GSS值)
    - adata: AnnData对象
    - select_n: 选择的基因数量
    - expr_weight: 表达量在综合评分中的权重(0-1)

    返回:
    - top_genes: 选择的基因名称列表
    - gene_scores: 选择的基因及其分数的DataFrame
    """

    # 计算每个基因非零值的平均GSS分数
    mean_gss = (mk_score_df.replace(0, pd.NA).mean(axis=1, skipna=True))

    # 计算每个基因的平均表达量
    if scipy.sparse.issparse(adata.X):
        mean_expr = np.array(adata.X.mean(axis=0)).flatten()
    else:
        mean_expr = np.mean(adata.X, axis=0)

    # 转换为Series，索引为基因名
    mean_expr = pd.Series(mean_expr, index=adata.var_names)

    # 标准化GSS值和表达量
    gss_norm = (mean_gss - mean_gss.min()) / (mean_gss.max() - mean_gss.min())
    expr_norm = (mean_expr - mean_expr.min()) / (mean_expr.max() - mean_expr.min())

    # 计算加权综合评分
    scores = gss_norm * expr_weight + expr_norm * (1 - expr_weight)
    score_name = '综合评分'

    # 获取分数最高的前n个基因
    top_genes_idx = scores.argsort()[::-1][:select_n]
    top_genes = scores.index[top_genes_idx]

    # 创建包含基因及其分数的DataFrame
    gene_scores = pd.DataFrame({
        '基因名称': top_genes,
        score_name: scores[top_genes].values,
        'GSS值': mean_gss[top_genes].values,
        '表达量': mean_expr[top_genes].values
    })

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    gene_scores.to_csv(output_dir + "selected_genes.csv", index=False)
    print(f"验证结果已保存至 {output_dir}")

    return top_genes, gene_scores


def select_gene_statistic(mk_score_df, adata, output_dir=None,
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


def plot_expression_heatmap(mk_score_df, adata, select_n=50, method='gss_expr', expr_weight=0.5,
                            downsample_ratio=0.1, output_dir=None, show=True):
    """
    绘制根据不同准则选择的top n个基因的热图

    参数:
    - mk_score_df: 标记分数DataFrame
    - adata: AnnData对象
    - select_n: 选择的基因数量
    - method: 基因选择方法，可选值: 'gss', 'expr', 'gss_expr'
    - expr_weight: 表达量在综合评分中的权重(0-1)
    - downsample_ratio: 细胞降采样比例
    - output_dir: 图像保存目录
    - show: 是否显示图像
    """
    # 对细胞进行降采样
    n_cells = adata.shape[0]
    n_downsample = int(n_cells * downsample_ratio)
    downsample_idx = np.random.choice(n_cells, n_downsample, replace=False)
    adata_downsampled = adata[downsample_idx, :].copy()

    # 选择top n个基因
    top_genes, gene_scores = select_gene_traditional(
        mk_score_df, adata_downsampled,
        select_n=select_n,
        method=method,
        expr_weight=expr_weight
    )

    # 打印筛选出的基因
    method_names = {
        'gss': 'GSS值最高',
        'expr': '表达量最高',
        'gss_expr': 'GSS值和表达量综合评分最高'
    }
    # print(f"筛选出的{method_names[method]}的基因:")
    # print(gene_scores.to_string(index=False))

    # 创建表达矩阵

    expr_matrix = pd.DataFrame(
        index=adata_downsampled.obs_names,
        columns=top_genes
    )

    for gene in top_genes:
        expr_matrix[gene] = adata_downsampled[:, gene].X.toarray().flatten() if scipy.sparse.issparse(
            adata_downsampled.X) else adata_downsampled[:, gene].X.flatten()

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 绘制热图
    sns.heatmap(
        expr_matrix,
        cmap='viridis',
        square=True,
        xticklabels=True,
        yticklabels=False,
        cbar_kws={'label': '表达量'},
        linewidths=0.5,  # 增加边框宽度
        annot_kws={"size": 6}  # 调整注释字体大小
    )

    # 设置标题
    plt.title(f'{method_names[method]}的前 {select_n} 个基因的热图')

    # 调整布局
    plt.tight_layout()

    # 保存图像（如果指定了输出目录）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'top_{select_n}_genes_heatmap_{method}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"热图已保存至: {output_path}")

        # 保存筛选出的基因
        gene_scores_path = os.path.join(output_dir, f'top_{select_n}_genes_{method}.csv')
        gene_scores.to_csv(gene_scores_path, index=False)
        print(f"筛选出的基因已保存至: {gene_scores_path}")

    # 显示图像
    if show:
        plt.show()

    # 关闭图形
    plt.close()


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


def main():

    #分析方式
    method = 'selectGenes'

    # Mus musculus
    # LIHB:NCBI627, MEL:ZEN81, PRAD:NCBI793, SKCM:NCBI689

    # Homo sapiens
    # ACYC:NCBI771, ALL:TENX134, BLCA:NCBI855, CESC:TENX50, COAD:TENX156,
    # COADREAD:TENX139, CSCC:NCBI770, EPM:NCBI641, GBM:TENX138, HCC:TENX120,
    # HGSOC:TENX142, IDC:TENX99, ILC :TENX96, LNET:TENX72, LUAD:TENX141

    # 本地数据地址
    work_dir = '/Users/wuyang/Documents/MyPaper/3/gsVis'
    sample_id = 'BRCA'
    sample_name = 'Human_Breast_Cancer'

    # HEST数据地址
    dataset = 'HEST'
    species = 'Homo sapiens'  # 'Mus musculus'
    cancer = 'ACYC'  # 'LIHB'
    id = 'NCBI771'  # 'NCBI627'

    # feather_path, h5ad_path, output_dir = (
    #     get_self_data_dir(method, work_dir, sample_name, sample_id))

    feather_path, h5ad_path, output_dir = (
        get_hest_data_dir(method, work_dir, dataset, species, cancer, id))

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='可视化基因在空间上的表达分布')
    parser.add_argument('--feather-path', default=feather_path, help='标记分数feather文件路径')
    parser.add_argument('--h5ad-path', default=h5ad_path, help='AnnData h5ad文件路径')
    parser.add_argument('--output-dir', default=output_dir, help='图像保存目录')

    parser.add_argument('--gene', help='要可视化的基因名称')
    parser.add_argument('--genes-file', help='包含多个基因名称的文件路径，每行一个基因')
    parser.add_argument('--select-n', type=int, default=10, help='可视化值得分析的n个基因')
    parser.add_argument('--method', choices=['statistic', 'gss_expr'], default='statistic',
                        help='基因选择方法: statistic=使用统计学方法筛选, gss_expr=基于GSS值和表达量的综合评分')
    parser.add_argument('--expr-weight', type=float, default=0.1,
                        help='表达量在综合评分中的权重(0-1)，仅在method=gss_expr时有效')
    parser.add_argument('--downsample-ratio', type=float, default=0.1, help='细胞降采样比例')
    parser.add_argument('--cmap', default='viridis', help='颜色映射方案 (默认: viridis)')
    parser.add_argument('--size', type=float, default=1.0, help='点大小 (默认: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.6, help='透明度 (默认: 0.7)')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置中文字体
    set_chinese_font()

    # 读取数据
    mk_score_df, adata = read_data(args.feather_path, args.h5ad_path)

    # 检查是否提供了基因名称
    if args.gene:
        # 可视化单个基因
        plot_gene_spatial(
            mk_score_df, adata, args.gene,
            output_dir=args.output_dir,
            cmap=args.cmap,
            size=args.size,
            alpha=args.alpha
        )
    elif args.genes_file:
        # 从文件读取多个基因
        with open(args.genes_file, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]

        if genes:
            print(f"将可视化 {len(genes)} 个基因")
            plot_multiple_genes(
                mk_score_df, adata, genes,
                output_dir=args.output_dir,
                cmap=args.cmap,
                size=args.size,
                alpha=args.alpha
            )
        else:
            print("错误: 基因文件为空")
    elif args.select_n != 0:
        # 可视化值得分析的n个基因
        print(f"将使用{args.method}方法来可视化基因！！！")

        # 绘制热图
        # plot_expression_heatmap(
        #     mk_score_df, adata,
        #     select_n=args.select_n,
        #     method=args.method,
        #     expr_weight=args.expr_weight,
        #     downsample_ratio=args.downsample_ratio,
        #     output_dir=args.output_dir
        # )

        if args.method == "statistic":
            # 通过统计学方式可视化基因
            selected_genes, _ = select_gene_statistic(
                mk_score_df, adata,
                output_dir=args.output_dir,
                min_expr_threshold=0.01,
                min_gss_threshold=0.1,
                concentration_threshold=90, # 表达离散度和集中性阈值
                corr_threshold=0.6, # GSS-表达量相关性阈值
                entropy_threshold=8.5, # GSS的信息熵阈值
                morans_i_threshold=0.5, # 空间自相关性阈值
            )
        elif args.method == "gss_expr":
            # 基于GSS值和表达量可视化基因
            selected_genes, _ = select_gene_traditional(
                mk_score_df, adata,
                output_dir=args.output_dir,
                select_n=args.select_n,
                expr_weight=args.expr_weight
            )
        else:
            raise ValueError(f"不支持的选择方法: {args.method}")

        print(selected_genes)

        # # 计算各基因在空间聚类中的表达纯度
        # purity_scores = calculate_spatial_purity(adata, selected_genes)
        # purity_scores.to_csv(args.output_dir + "purity_scores.csv", index=False, sep='\t')
        # print(f"各基因在空间聚类中的表达纯度结果已保存至 {args.output_dir}")
        #
        # # 验证基因在特定细胞类型中的特异性表达
        # specificity = tissue_type_specificity(adata, selected_genes)
        # specificity.to_csv(args.output_dir + "specificity.csv", index=False, sep='\t')
        # print(f"基因在特定细胞类型中的特异性表达结果已保存至 {args.output_dir}")

        plot_multiple_genes(
            mk_score_df, adata, selected_genes,
            output_dir=args.output_dir,
            visual_indicators = "GSS", # ["GSS", "Expr"]
            cmap=args.cmap,
            size=args.size,
            alpha=args.alpha
        )
    else:
        print("错误: 请提供--gene、--genes-file或--select-n参数等")
        parser.print_help()


if __name__ == "__main__":
    main()