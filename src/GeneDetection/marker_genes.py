
import numpy as np
import pandas as pd
from collections import Counter
import multiprocessing as mp
from functools import partial
from scipy.stats import rankdata
from gss_analysis import run_gss_analysis


def analyze_at_spot_domain_level(mk_score_df, adata, output_dir,
                                 top_genes_per_spot=100,
                                 fc_threshold=1.5,
                                 pval_threshold=0.05):
    """
    Spot微域特征基因识别
    """
    print(f"开始处理 {adata.n_obs} 个spots...")

    # 预计算全局统计量
    print("步骤1: 预先计算全局统计量...")
    expression_matrix, global_expression, global_means = precompute_global_stats(adata)
    spot_names = adata.obs_names.tolist()
    gene_names = adata.var_names.tolist()
    global_means_array = global_means.values
    spot_coordinates = extract_spot_coordinates(adata)

    # 预计算全局排名
    print("步骤2: 预计算全局基因排名...")
    global_ranks = precompute_global_ranks(expression_matrix)

    # 预计算GSS分布
    print("步骤3: 运行GSS分布分析...")
    gssInfo = run_gss_analysis(mk_score_df, adata, output_dir=f'{output_dir}/gss_output')
    genes_per_spot = gssInfo['spot_genes_dict']

    # 预计算邻居关系
    print("步骤4: 预计算邻居关系矩阵...")
    regional_neighbors = adata.uns.get("regional_neighbors", {})
    neighbor_mask = precompute_neighbor_masks(spot_names, regional_neighbors)

    # 主分析循环
    print("步骤5: 并行执行特征基因筛选...")
    n_jobs = 40
    shared_data = {
        'expression_matrix': expression_matrix,
        'global_ranks': global_ranks,
        'genes_per_spot': genes_per_spot,
        'gene_names': gene_names,
        'neighbor_mask': neighbor_mask,
        'spot_names': spot_names,
        'spot_coordinates': spot_coordinates,
        'fc_threshold': fc_threshold,
        'pval_threshold': pval_threshold,
        'top_genes_per_spot': top_genes_per_spot,
        'global_means_array': global_means_array
    }

    # 创建进程池
    with mp.Pool(processes=n_jobs) as pool:
        # 准备参数
        process_func = partial(process_single_spot, shared_data=shared_data)
        # 并行处理所有spots
        results = list(pool.map(process_func, range(len(spot_names))))

    # 整理结果
    spot_domain_features = {}

    # for i, spot_name in enumerate(spot_names):
    #     # 获取微域信息
    #     domain_indices = np.where(neighbor_mask[i])[0]
    #     domain_spots = [spot_names[idx] for idx in domain_indices]
    #
    #     # GSS候选基因筛选
    #     candidate_genes = get_candidate_genes(domain_spots, genes_per_spot, gene_names)
    #
    #     if not candidate_genes:
    #         spot_domain_features[spot_name] = {
    #             'domain_spots': domain_spots,
    #             'domain_size': len(domain_spots),
    #             'feature_genes': [],
    #             'num_feature_genes': 0,
    #             'candidate_genes_count': 0
    #         }
    #         continue
    #
    #     # 转换为基因索引
    #     candidate_gene_indices = [gene_names.index(gene) for gene in candidate_genes if gene in gene_names]
    #
    #     # 使用基于排名的快速统计检验
    #     feature_gene_indices = fast_rank_based_test(
    #         expression_matrix, global_ranks, domain_indices,
    #         candidate_gene_indices, fc_threshold, pval_threshold, top_genes_per_spot
    #     )
    #
    #     feature_genes = [gene_names[idx] for idx in feature_gene_indices]
    #
    #     # 存储结果
    #     spot_domain_features[spot_name] = {
    #         'domain_spots': domain_spots,
    #         'domain_size': len(domain_spots),
    #         'feature_genes': feature_genes,
    #         'num_feature_genes': len(feature_genes),
    #         'candidate_genes_count': len(candidate_genes)
    #     }
    #
    #     # if i % 500 == 0:
    #     print(f"已处理 {i}/{len(spot_names)} spots, 当前找到 {len(feature_genes)} 个特征基因")

    for spot_name, result in zip(spot_names, results):
        spot_domain_features[spot_name] = result

    print_analysis_summary(spot_domain_features)
    return spot_domain_features



def process_single_spot(spot_idx, shared_data):
    """处理单个spot的函数，用于多进程并行"""
    expression_matrix = shared_data['expression_matrix']
    global_ranks = shared_data['global_ranks']
    genes_per_spot = shared_data['genes_per_spot']
    gene_names = shared_data['gene_names']
    neighbor_mask = shared_data['neighbor_mask']
    spot_names = shared_data['spot_names']
    spot_coordinates = shared_data['spot_coordinates']
    fc_threshold = shared_data['fc_threshold']
    pval_threshold = shared_data['pval_threshold']
    top_genes_per_spot = shared_data['top_genes_per_spot']
    global_means_array = shared_data['global_means_array']


    domain_indices = np.where(neighbor_mask[spot_idx])[0]
    domain_spots = [spot_names[idx] for idx in domain_indices]
    current_spot_coord = spot_coordinates[spot_names[spot_idx]].tolist()

    candidate_genes = get_candidate_genes(domain_spots, genes_per_spot, gene_names)

    if not candidate_genes:
        return {
            'domain_spots': domain_spots,
            'domain_size': len(domain_spots),
            'coordinates': current_spot_coord,
            'feature_genes': [],
            'num_feature_genes': 0,
            'gene_avg_expr_domain': {},
            'gene_avg_expr_global': {},
            'fold_changes': {}
        }

    candidate_gene_indices = [gene_names.index(gene) for gene in candidate_genes if gene in gene_names]

    feature_gene_indices = fast_rank_based_test(
        expression_matrix, global_ranks, domain_indices,
        candidate_gene_indices, fc_threshold, pval_threshold, top_genes_per_spot
    )

    feature_genes = [gene_names[idx] for idx in feature_gene_indices]

    gene_avg_expr_domain = {}
    gene_avg_expr_global = {}
    fold_changes = {}

    if feature_gene_indices:
        domain_expr_subset = expression_matrix[domain_indices][:, feature_gene_indices]
        avg_exprs_domain = np.mean(domain_expr_subset, axis=0)

        avg_exprs_global = global_means_array[feature_gene_indices]

        fold_changes_array = avg_exprs_domain / (avg_exprs_global + 1e-8)

        for i, gene_idx in enumerate(feature_gene_indices):
            gene_name = gene_names[gene_idx]
            gene_avg_expr_domain[gene_name] = float(avg_exprs_domain[i])
            gene_avg_expr_global[gene_name] = float(avg_exprs_global[i])
            fold_changes[gene_name] = float(fold_changes_array[i])

    return {
        'domain_spots': domain_spots,
        'domain_size': len(domain_spots),
        'coordinates': current_spot_coord,
        'feature_genes': feature_genes,
        'num_feature_genes': len(feature_genes),
        'gene_avg_expr_domain': gene_avg_expr_domain,
        'gene_avg_expr_global': gene_avg_expr_global,
        'fold_changes': fold_changes
    }


def extract_spot_coordinates(adata):
    """
    从adata对象中提取spots的空间坐标
    """
    spot_coordinates = {}
    spot_names = adata.obs_names.tolist()

    if 'spatial' in adata.obsm:
        # 从obsm['spatial']获取坐标
        spatial_coords = adata.obsm['spatial']
        for i, spot in enumerate(spot_names):
            spot_coordinates[spot] = spatial_coords[i]
    else:
        # 如果找不到坐标信息，使用索引作为虚拟坐标
        print("警告: 未找到空间坐标信息，使用虚拟坐标")

    return spot_coordinates


def precompute_global_ranks(expression_matrix):
    """
    预计算每个基因在所有spots中的表达排名，加速后续的统计检验
    """
    n_genes = expression_matrix.shape[1]
    global_ranks = np.zeros_like(expression_matrix, dtype=float)

    # 对每个基因单独计算排名
    for gene_idx in range(n_genes):
        gene_expr = expression_matrix[:, gene_idx]
        # 使用scipy的rankdata，处理相同值的情况
        global_ranks[:, gene_idx] = rankdata(gene_expr)

    return global_ranks


def fast_rank_based_test(expression_matrix, global_ranks, domain_indices,
                         candidate_gene_indices, fc_threshold, pval_threshold, top_k):
    """
    基于预计算排名的快速统计检验，结合了Wilcoxon检验的思想
    """
    n_total = expression_matrix.shape[0]
    n_domain = len(domain_indices)
    n_other = n_total - n_domain

    scores = []

    for gene_idx in candidate_gene_indices:
        # 计算倍数变化
        domain_expr = expression_matrix[domain_indices, gene_idx]
        domain_mean = np.mean(domain_expr)
        global_mean = np.mean(expression_matrix[:, gene_idx])
        fold_change = domain_mean / (global_mean + 1e-8)

        if fold_change < fc_threshold:
            continue

        # 基于预计算排名的快速统计检验
        domain_ranks = global_ranks[domain_indices, gene_idx]
        domain_rank_sum = np.sum(domain_ranks)

        # 计算期望排名和（在零假设下）
        expected_rank_sum = n_domain * (n_total + 1) / 2

        # 计算排名和统计量（Wilcoxon秩和检验的核心）
        rank_statistic = domain_rank_sum - expected_rank_sum

        # 标准化统计量（近似正态分布）
        std_dev = np.sqrt(n_domain * n_other * (n_total + 1) / 12)
        if std_dev > 0:
            z_score = rank_statistic / std_dev
            # 计算双侧p值
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        else:
            p_value = 1.0

        # 使用评分逻辑：-log10(p值) × 倍数变化
        if p_value < pval_threshold:
            score = -np.log10(p_value + 1e-8) * fold_change
            scores.append((gene_idx, score))

    # 按评分排序
    scores.sort(key=lambda x: x[1], reverse=True)
    return [gene_idx for gene_idx, score in scores[:top_k]]


def precompute_neighbor_masks(spot_names, regional_neighbors):
    """预计算邻居关系的布尔矩阵"""
    n_spots = len(spot_names)
    spot_to_index = {spot: i for i, spot in enumerate(spot_names)}

    neighbor_mask = np.zeros((n_spots, n_spots), dtype=bool)

    for i, spot in enumerate(spot_names):
        if spot in regional_neighbors:
            neighbors = regional_neighbors[spot]
            for neighbor in neighbors:
                if neighbor in spot_to_index:
                    j = spot_to_index[neighbor]
                    neighbor_mask[i, j] = True
        neighbor_mask[i, i] = True

    return neighbor_mask


def precompute_global_stats(adata):
    """预先计算全局统计量"""
    from scipy.sparse import issparse
    if issparse(adata.X):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = adata.X

    spot_names = adata.obs_names.tolist()
    gene_names = adata.var_names.tolist()
    global_expression = pd.DataFrame(
        expression_matrix,
        index=spot_names,
        columns=gene_names
    )

    global_means = global_expression.mean()
    return expression_matrix, global_expression, global_means


def get_candidate_genes(domain_spots, genes_per_spot, gene_names):
    """基于GSS分析和频率统计筛选候选基因（没有考虑微域内的表达强度分布，后续可以考虑优化）"""
    domain_genes = []

    for domain_spot in domain_spots:
        if domain_spot in genes_per_spot:
            domain_genes.extend(genes_per_spot[domain_spot])

    gene_freq = Counter(domain_genes)
    min_freq = max(1, len(domain_spots) // 3)

    candidate_genes = [
        gene for gene, count in gene_freq.items()
        if count >= min_freq and gene in gene_names
    ]
    return candidate_genes


def print_analysis_summary(spot_domain_features):
    """输出分析摘要"""
    total_spots = len(spot_domain_features)
    total_genes = sum(data['num_feature_genes'] for data in spot_domain_features.values())
    spots_with_genes = sum(1 for data in spot_domain_features.values()
                           if data['num_feature_genes'] > 0)

    print(f"\n=== 分析摘要 ===")
    print(f"总spots数: {total_spots}")
    print(f"找到特征基因的spots数: {spots_with_genes} ({spots_with_genes / total_spots * 100:.1f}%)")
    print(f"平均每个spot的特征基因数: {total_genes / total_spots:.2f}")
    print(f"总特征基因数: {total_genes}")

    domain_sizes = [data['domain_size'] for data in spot_domain_features.values()]
    print(f"微域大小 - 均值: {np.mean(domain_sizes):.2f}, 范围: [{min(domain_sizes)}, {max(domain_sizes)}]")