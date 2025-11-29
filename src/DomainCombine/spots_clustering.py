
# -*- coding: utf-8 -*-
"""
肿瘤微域特征基因分析完整流程 - 社区检测聚类版
核心聚类模块：专注于聚类分析，结果保存为文件供富集分析使用
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from collections import defaultdict, Counter
import networkx as nx
import gseapy as gp
import warnings

warnings.filterwarnings('ignore')


# --------------------------
# 1. 参数设置 - 社区检测聚类版
# --------------------------
class Config:
    """分析参数配置"""
    file_root = "../../output/HEST"

    species = "Homo sapiens"
    cancer_type = "EPM"
    id = "NCBI629"

    file_dir = os.path.join(file_root, species, cancer_type, id)

    # 数据路径
    input_json_path = file_dir + "/spot_domain_features.json"  # 输入数据JSON文件
    output_dir = file_dir + "/tumor_analysis_results"  # 结果输出目录

    # 特征基因筛选参数
    fc_threshold = 2.0  # 差异倍数阈值
    fdr_threshold = 0.05  # FDR校正阈值
    skip_statistical_test = True  # 是否跳过统计检验（直接使用输入数据中的特征基因）
    predefined_feature_genes = []  # 若skip_statistical_test=True，可直接指定特征基因列表

    # 最小基因数参数
    min_genes_per_domain = 1  # 每个微域最少特征基因数

    # 图构建参数
    integrated_similarity_threshold = 0.3  # 综合相似度阈值
    min_spatial_sim = 0.2  # 最小空间相似度
    min_gene_sim = 0.1  # 最小基因相似度

    # 社区检测参数
    community_resolution = 0.8  # Louvain或Leiden算法分辨率参数，值越大社区越小
    min_community_size = 20  # 最小社区大小
    max_community_size = 500  # 最大社区大小

    # 质量过滤参数
    min_cluster_quality_score = 0.2  # 最小聚类质量评分
    min_internal_density = 0.1  # 最小内部连接密度
    min_gene_consistency = 0.1  # 最小基因一致性
    min_core_genes = 2  # 最小核心基因数
    min_final_gene_similarity = 0.2  # 最小基因相似度
    min_spot_coverage = 0.3  # 最小spot覆盖率
    min_final_cluster_size = 5  # 最小最终聚类大小

    # 基因频率参数
    min_gene_frequency = 0.5  # 核心基因最小频率

    # 富集分析参数
    enrich_padj_threshold = 0.01 # 显著性阈值

    # 本地GMT文件路径
    local_gmt_files = {
        "GO_BP": "/home/wuyang/hest-data/gseapy/GO_Biological_Process_2025.gmt",
        "GO_CC": "/home/wuyang/hest-data/gseapy/GO_Cellular_Component_2025.gmt",
        "GO_MF": "/home/wuyang/hest-data/gseapy/GO_Molecular_Function_2025.gmt",
        "KEGG": "/home/wuyang/hest-data/gseapy/KEGG_2021_Human.gmt",
        "Hallmark": "/home/wuyang/hest-data/gseapy/MSigDB_Hallmark_2020.gmt"
    }


# 创建输出目录
if not os.path.exists(Config.output_dir):
    os.makedirs(Config.output_dir)
    os.makedirs(os.path.join(Config.output_dir, "figures"))
    os.makedirs(os.path.join(Config.output_dir, "tables"))


# --------------------------
# 2. 数据读取与预处理
# --------------------------
def load_and_preprocess_data(json_path):
    """读取JSON数据并进行预处理"""
    print("正在读取数据...")
    # 读取JSON数据
    with open(json_path, "r") as f:
        data = json.load(f)

    domain_ids = list(data.keys())
    gene_expr_list = []
    coords_list = []

    for did in domain_ids:
        domain_data = data[did]

        # 提取表达数据
        expr_df = pd.DataFrame({
            "domain_id": did,
            "gene": list(domain_data["gene_avg_expr_domain"].keys()),
            "expr_domain": list(domain_data["gene_avg_expr_domain"].values()),
            "expr_global": list(domain_data["gene_avg_expr_global"].values()),
            "fold_change": list(domain_data["fold_changes"].values())
        })
        gene_expr_list.append(expr_df)

        # 提取空间坐标
        if "coordinates" in domain_data:
            x, y = domain_data["coordinates"]
        else:
            # 随机生成坐标用于示例，实际使用时应替换为真实坐标
            x, y = np.random.uniform(0, 100), np.random.uniform(0, 100)
        coords_list.append({"domain_id": did, "x": x, "y": y})

    # 合并数据
    all_expr = pd.concat(gene_expr_list, ignore_index=True)
    spot_coords = pd.DataFrame(coords_list)

    # 保存预处理数据
    all_expr.to_csv(os.path.join(Config.output_dir, "tables", "all_gene_expression.csv"), index=False)
    spot_coords.to_csv(os.path.join(Config.output_dir, "tables", "spot_coordinates.csv"), index=False)

    print(f"数据读取完成，共{len(domain_ids)}个微域，{len(all_expr['gene'].unique())}个基因")
    return all_expr, spot_coords, data


# --------------------------
# 3. 特征基因筛选
# --------------------------
def filter_feature_genes(all_expr, raw_data=None):
    """基于差异倍数和统计显著性筛选特征基因（支持跳过检验）"""
    print("正在筛选特征基因...")

    # 若配置跳过统计检验，直接使用预设基因或原始数据中的特征基因
    if Config.skip_statistical_test:
        if len(Config.predefined_feature_genes) > 0:
            # 使用用户指定的特征基因
            feature_genes = Config.predefined_feature_genes
            # 提取这些基因的统计信息（用于后续分析）
            gene_stats = all_expr[all_expr["gene"].isin(feature_genes)].groupby("gene").agg(
                mean_domain=("expr_domain", "mean"),
                mean_global=("expr_global", "mean"),
                fold_change=("fold_change", "mean")
            ).reset_index()
            gene_stats["p_val"] = 0.0  # 占位
            gene_stats["fdr"] = 0.0  # 占位
        else:
            # 从原始数据中提取每个微域的特征基因并取 union
            domain_feature_genes = []
            for did in raw_data:
                domain_feature_genes.extend(raw_data[did].get("feature_genes", []))
            feature_genes = list(set(domain_feature_genes))  # 去重
            # 提取统计信息
            gene_stats = all_expr[all_expr["gene"].isin(feature_genes)].groupby("gene").agg(
                mean_domain=("expr_domain", "mean"),
                mean_global=("expr_global", "mean"),
                fold_change=("fold_change", "mean")
            ).reset_index()
            gene_stats["p_val"] = 0.0
            gene_stats["fdr"] = 0.0

        feature_genes = [g for g in feature_genes if g in all_expr["gene"].unique()]  # 过滤不存在的基因
        gene_stats = gene_stats[gene_stats["gene"].isin(feature_genes)]
    else:
        # 原有逻辑：执行统计检验和筛选
        def calculate_statistics(group):
            control = group["expr_global"].values
            treatment = group["expr_domain"].values
            stat, p_val = mannwhitneyu(treatment, control, alternative='greater')
            return pd.Series({
                "p_val": p_val,
                "mean_domain": np.mean(treatment),
                "mean_global": np.mean(control)
            })

        gene_stats = all_expr.groupby("gene").apply(calculate_statistics).reset_index()
        gene_stats["fdr"] = multipletests(gene_stats["p_val"], method="fdr_bh")[1]
        gene_stats["fold_change"] = gene_stats["mean_domain"] / gene_stats["mean_global"].replace(0, 1e-10)
        # 应用筛选条件
        gene_stats = gene_stats[
            (gene_stats["fold_change"] >= Config.fc_threshold) &
            (gene_stats["fdr"] < Config.fdr_threshold)
            ].sort_values("fold_change", ascending=False)
        feature_genes = gene_stats["gene"].tolist()

    # 保存结果
    gene_stats.to_csv(os.path.join(Config.output_dir, "tables", "feature_genes.csv"), index=False)
    print(f"特征基因筛选完成，共{len(feature_genes)}个基因符合条件")
    return feature_genes, gene_stats


# --------------------------
# 4. 社区检测聚类核心函数
# --------------------------
def get_domain_feature_genes(raw_data, domain_id, feature_genes):
    """获取单个微域的特征基因（应用最小基因数过滤）"""
    if not Config.skip_statistical_test:
        domain_genes = raw_data[domain_id].get("feature_genes", [])
    else:
        domain_expr = raw_data[domain_id]["gene_avg_expr_domain"]
        domain_genes = [g for g in feature_genes if g in domain_expr]

    # 应用最小基因数过滤
    if len(domain_genes) < Config.min_genes_per_domain:
        return []
    return domain_genes


def calculate_gene_similarity(genes1, genes2):
    """计算两个基因集的Jaccard相似度"""
    if not genes1 or not genes2:
        return 0

    intersection = len(genes1.intersection(genes2))
    union = len(genes1.union(genes2))
    return intersection / union if union > 0 else 0


def calculate_spatial_overlap_similarity(raw_data, domain_ids):
    """基于spot重叠计算空间相似度"""
    print("  - 计算空间重叠相似度...")
    n_domains = len(domain_ids)
    overlap_similarity = np.zeros((n_domains, n_domains))

    # 构建微域-spot映射
    domain_spots = {}
    for i, did in enumerate(domain_ids):
        domain_spots[did] = set(raw_data[did]["domain_spots"])

    # 计算重叠相似度
    for i in range(n_domains):
        did_i = domain_ids[i]
        spots_i = domain_spots[did_i]

        for j in range(i, n_domains):
            did_j = domain_ids[j]
            spots_j = domain_spots[did_j]

            # Jaccard相似度：交集/并集
            intersection = len(spots_i.intersection(spots_j))
            union = len(spots_i.union(spots_j))

            similarity = intersection / union if union > 0 else 0
            overlap_similarity[i, j] = similarity
            overlap_similarity[j, i] = similarity

    # 确保对角线为1
    np.fill_diagonal(overlap_similarity, 1.0)

    print(f"    空间相似度范围: {overlap_similarity.min():.3f} - {overlap_similarity.max():.3f}")
    return overlap_similarity


def calculate_integrated_similarity(domain_i, domain_j, raw_data, domain_gene_sets):
    """
    计算两个微域的综合相似度
    """
    # 空间相似度：基于spot重叠
    spots_i = set(raw_data[domain_i]["domain_spots"])
    spots_j = set(raw_data[domain_j]["domain_spots"])
    intersection = len(spots_i.intersection(spots_j))
    union = len(spots_i.union(spots_j))
    spatial_sim = intersection / union if union > 0 else 0

    # 基因相似度：基于特征基因集
    if domain_i in domain_gene_sets and domain_j in domain_gene_sets:
        genes_i = domain_gene_sets[domain_i]
        genes_j = domain_gene_sets[domain_j]
        gene_sim = calculate_gene_similarity(genes_i, genes_j)
    else:
        gene_sim = 0

    # 综合评分：要求两者都达到最小阈值
    if spatial_sim < Config.min_spatial_sim or gene_sim < Config.min_gene_sim:
        return 0

    # 使用调和平均，对低值更敏感
    if spatial_sim + gene_sim > 0:
        integrated_sim = 2 * spatial_sim * gene_sim / (spatial_sim + gene_sim)
    else:
        integrated_sim = 0

    return integrated_sim


def calculate_cluster_internal_gene_similarity(cluster, domain_gene_sets):
    """计算聚类内部的平均基因相似度"""
    if len(cluster) < 2:
        return 1.0

    valid_domains = [d for d in cluster if d in domain_gene_sets and len(domain_gene_sets[d]) > 0]
    if len(valid_domains) < 2:
        return 0.0

    total_similarity = 0
    count = 0

    for i in range(len(valid_domains)):
        for j in range(i + 1, len(valid_domains)):
            genes_i = domain_gene_sets[valid_domains[i]]
            genes_j = domain_gene_sets[valid_domains[j]]
            similarity = calculate_gene_similarity(genes_i, genes_j)
            total_similarity += similarity
            count += 1

    return total_similarity / count if count > 0 else 0.0


def calculate_gene_frequency(cluster, domain_gene_sets):
    """计算聚类中每个基因的出现频率"""
    gene_counts = defaultdict(int)
    total_domains = len(cluster)

    for domain in cluster:
        if domain in domain_gene_sets:
            for gene in domain_gene_sets[domain]:
                gene_counts[gene] += 1

    gene_frequencies = {gene: count / total_domains for gene, count in gene_counts.items()}
    return gene_frequencies


def get_cluster_spots(cluster, raw_data):
    """获取聚类中所有微域的spot并集"""
    cluster_spots = set()
    for domain in cluster:
        if domain in raw_data:
            cluster_spots.update(raw_data[domain]["domain_spots"])
    return cluster_spots


def calculate_spatial_compactness(coordinates):
    """
    计算空间紧凑性：基于点集的凸包面积与最小包围圆的比例
    """
    try:
        from scipy.spatial import ConvexHull
        # 计算凸包面积
        hull = ConvexHull(coordinates)
        convex_area = hull.volume  # 在2D中，volume是面积

        # 计算点集范围
        x_range = np.ptp(coordinates[:, 0])
        y_range = np.ptp(coordinates[:, 1])
        bounding_area = x_range * y_range

        # 紧凑性 = 凸包面积 / 边界框面积
        if bounding_area > 0:
            compactness = convex_area / bounding_area
        else:
            compactness = 1.0

        return compactness
    except:
        return 1.0


# ==================== 社区检测核心函数 ====================
def louvain_community_detection(G, resolution=1.0, min_community_size=3):
    """
    使用Louvain算法进行社区检测

    参数:
    - G: 图对象
    - resolution: 分辨率参数，控制社区大小
    - min_community_size: 最小社区大小
    """
    try:
        import community as community_louvain

        print(f"  - 使用Louvain算法进行社区检测 (分辨率={resolution})...")

        # 计算分区
        partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)

        # 将分区转换为社区列表
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        # 转换为集合，并过滤最小大小
        community_sets = [set(nodes) for nodes in communities.values()
                          if len(nodes) >= min_community_size]

        print(f"    Louvain发现 {len(community_sets)} 个社区")

        # 分析社区大小分布
        community_sizes = [len(comm) for comm in community_sets]
        if community_sizes:
            print(
                f"    社区大小分布: 最小{min(community_sizes)}, 最大{max(community_sizes)}, 平均{np.mean(community_sizes):.1f}")

        return community_sets

    except ImportError:
        print("警告: 未安装python-louvain，使用连通分量作为替代")
        print("请安装: pip install python-louvain")
        return [c for c in nx.connected_components(G) if len(c) >= min_community_size]


def leiden_community_detection(G, resolution=1.0, min_community_size=3):
    """
    使用Leiden算法进行社区检测（更先进）
    """
    try:
        import leidenalg
        import igraph as ig

        print(f"  - 使用Leiden算法进行社区检测 (分辨率={resolution})...")

        # 将networkx图转换为igraph
        # 首先确保所有节点都有连续的整数ID
        nodes = list(G.nodes())
        node_to_id = {node: i for i, node in enumerate(nodes)}
        id_to_node = {i: node for i, node in enumerate(nodes)}

        # 创建边列表和权重
        edge_list = []
        weights = []
        for u, v, data in G.edges(data=True):
            edge_list.append((node_to_id[u], node_to_id[v]))
            weights.append(data.get('weight', 1.0))

        # 创建igraph图
        g_ig = ig.Graph(n=len(nodes), edges=edge_list, directed=False)

        # 设置边权重
        if weights:
            g_ig.es['weight'] = weights

        # 设置节点名称属性，便于后续映射
        g_ig.vs['name'] = nodes

        # 运行Leiden算法
        partition = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            weights=weights if weights else None,
            # resolution_parameter=resolution
        )

        # 转换为社区列表
        community_sets = []
        for community in partition:
            if len(community) >= min_community_size:
                # 获取原始节点名称
                community_nodes = [id_to_node[i] for i in community]
                community_sets.append(set(community_nodes))

        print(f"    Leiden发现 {len(community_sets)} 个社区")

        # 分析社区大小分布
        community_sizes = [len(comm) for comm in community_sets]
        if community_sizes:
            print(
                f"    社区大小分布: 最小{min(community_sizes)}, 最大{max(community_sizes)}, 平均{np.mean(community_sizes):.1f}")

        return community_sets

    except ImportError as e:
        print(f"警告: Leiden算法导入失败 - {str(e)}")
        print("回退到Louvain算法")
        return louvain_community_detection(G, resolution, min_community_size)
    except Exception as e:
        print(f"Leiden算法执行错误: {str(e)}")
        print("回退到Louvain算法")
        return louvain_community_detection(G, resolution, min_community_size)


def analyze_graph_connectivity(G, domain_ids):
    """
    分析图的连接性，帮助诊断问题
    """
    print("  - 分析图连接性...")

    # 计算度分布
    degrees = [deg for _, deg in G.degree()]

    print(f"    图连接性统计:")
    print(f"      节点数: {G.number_of_nodes()}")
    print(f"      边数: {G.number_of_edges()}")
    print(f"      平均度: {np.mean(degrees):.2f}")
    print(f"      最大度: {max(degrees)}")
    print(f"      最小度: {min(degrees)}")
    print(f"      连通分量数: {nx.number_connected_components(G)}")

    # 分析连通分量大小分布
    components = list(nx.connected_components(G))
    component_sizes = [len(comp) for comp in components]

    print(f"      连通分量大小分布:")
    size_counts = Counter(component_sizes)
    for size, count in sorted(size_counts.items()):
        print(f"        大小{size}: {count}个分量")

    return degrees, component_sizes


def community_detection_clustering(raw_data, feature_genes, spot_coords):
    """
    基于社区检测的综合聚类方法
    """
    print("正在进行基于社区检测的综合聚类分析...")

    domain_ids = list(raw_data.keys())
    n_domains = len(domain_ids)

    # 1. 预计算每个微域的特征基因集
    domain_gene_sets = {}
    valid_domains = 0
    for did in domain_ids:
        domain_genes = get_domain_feature_genes(raw_data, did, feature_genes)
        if len(domain_genes) >= Config.min_genes_per_domain:
            domain_gene_sets[did] = set(domain_genes)
            valid_domains += 1

    print(f"    有效微域数量: {valid_domains}/{n_domains} (要求≥{Config.min_genes_per_domain}个特征基因)")

    # 2. 构建图：节点为微域，边权重为综合相似度
    G = nx.Graph()
    for did in domain_ids:
        G.add_node(did)

    print("  - 构建综合相似度图...")

    # 预计算空间重叠相似度
    spatial_sim = calculate_spatial_overlap_similarity(raw_data, domain_ids)

    # 添加边：基于综合相似度
    edges_added = 0
    for i in range(n_domains):
        domain_i = domain_ids[i]
        if domain_i not in domain_gene_sets:
            continue

        for j in range(i + 1, n_domains):
            domain_j = domain_ids[j]
            if domain_j not in domain_gene_sets:
                continue

            # 计算空间相似度
            spatial_similarity = spatial_sim[i, j]

            # 计算基因相似度
            genes_i = domain_gene_sets[domain_i]
            genes_j = domain_gene_sets[domain_j]
            gene_similarity = calculate_gene_similarity(genes_i, genes_j)

            # 综合相似度评分
            if spatial_similarity > 0 and gene_similarity > 0:
                # 使用调和平均，要求两个相似度都不能为0
                integrated_similarity = 2 * (spatial_similarity * gene_similarity) / (
                        spatial_similarity + gene_similarity)

                # 添加边的条件：综合相似度达到阈值
                if integrated_similarity >= Config.integrated_similarity_threshold:
                    G.add_edge(domain_i, domain_j,
                               weight=integrated_similarity,
                               spatial_sim=spatial_similarity,
                               gene_sim=gene_similarity)
                    edges_added += 1

    print(f"    图构建完成: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")

    # 分析图连接性
    analyze_graph_connectivity(G, domain_ids)

    # 3. 使用社区检测算法进行聚类
    print("  - 执行社区检测...")

    # 方法1: 使用Louvain算法
    # initial_communities = louvain_community_detection(
    #     G,
    #     resolution=Config.community_resolution,
    #     min_community_size=Config.min_community_size
    # )

    # 方法2: 或者使用Leiden算法（更推荐）
    initial_communities = leiden_community_detection(
        G,
        resolution=Config.community_resolution,
        min_community_size=Config.min_community_size
    )

    # 4. 过滤过大社区
    filtered_communities = []
    for community in initial_communities:
        if len(community) <= Config.max_community_size:
            filtered_communities.append(community)
        else:
            print(f"    过滤过大社区: {len(community)}个节点")

    print(f"    大小过滤后: {len(filtered_communities)}个社区")

    # 5. 质量过滤
    final_clusters = filter_clusters_by_quality(filtered_communities, domain_gene_sets, raw_data, G)

    # 6. 准备结果
    return prepare_clustering_results(final_clusters, domain_gene_sets, raw_data, spot_coords, domain_ids)


def filter_clusters_by_quality(clusters, domain_gene_sets, raw_data, G):
    """
    基于生物学质量过滤聚类
    """
    print("  - 基于生物学质量过滤聚类...")

    high_quality_clusters = []

    for i, cluster in enumerate(clusters):
        # 计算内部连接密度
        internal_density = calculate_internal_density(cluster, G)

        # 检查核心基因
        gene_freq = calculate_gene_frequency(cluster, domain_gene_sets)
        core_genes = [g for g, f in gene_freq.items() if f >= Config.min_gene_frequency]

        # 检查基因一致性
        gene_sim = calculate_cluster_internal_gene_similarity(cluster, domain_gene_sets)

        # 计算综合质量评分
        cluster_score = 0.5 * internal_density + 0.5 * gene_sim

        # 调试信息
        print(f"    社区{i}: 大小{len(cluster)}, "
              f"基因一致性{gene_sim:.3f}, 密度{internal_density:.3f}, "
              f"核心基因{len(core_genes)}, 综合评分{cluster_score:.3f}")

        # 质量判断
        quality_conditions = [
            cluster_score >= Config.min_cluster_quality_score,
            internal_density >= Config.min_internal_density,
            gene_sim >= Config.min_gene_consistency,
            len(core_genes) >= Config.min_core_genes,
            len(cluster) >= Config.min_final_cluster_size
        ]

        if all(quality_conditions):
            high_quality_clusters.append(cluster)
            print(f"      通过质量检查")
        else:
            failed_conditions = []
            if cluster_score < Config.min_cluster_quality_score:
                failed_conditions.append(f"综合评分低({cluster_score:.3f}<{Config.min_cluster_quality_score})")
            if internal_density < Config.min_internal_density:
                failed_conditions.append(f"内部密度低({internal_density:.3f}<{Config.min_internal_density})")
            if gene_sim < Config.min_gene_consistency:
                failed_conditions.append(f"基因相似度低({gene_sim:.3f}<{Config.min_gene_consistency})")
            if len(core_genes) < Config.min_core_genes:
                failed_conditions.append(f"核心基因不足({len(core_genes)}<{Config.min_core_genes})")
            if len(cluster) < Config.min_final_cluster_size:
                failed_conditions.append(f"聚类大小小({len(cluster)}<{Config.min_final_cluster_size})")

            print(f"      过滤原因: {', '.join(failed_conditions)}")

    print(f"    质量过滤后: {len(high_quality_clusters)}个高质量聚类")
    return high_quality_clusters


def calculate_internal_density(cluster, G):
    """计算聚类的内部连接密度"""
    nodes = list(cluster)
    internal_edges = 0

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if G.has_edge(nodes[i], nodes[j]):
                internal_edges += 1

    total_possible_edges = len(nodes) * (len(nodes) - 1) / 2
    return internal_edges / total_possible_edges if total_possible_edges > 0 else 0


def convert_numpy_types(obj):
    """递归转换所有numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def prepare_clustering_results(final_clusters, domain_gene_sets, raw_data, spot_coords, domain_ids):
    """
    准备聚类结果
    """
    # 识别未聚类区域
    all_domains = set(domain_ids)
    clustered_domains = set().union(*final_clusters) if final_clusters else set()
    unclustered_domains = all_domains - clustered_domains

    # 计算每个聚类的生物学指标
    cluster_biology = []
    for cluster_id, cluster in enumerate(final_clusters):
        gene_similarity = calculate_cluster_internal_gene_similarity(cluster, domain_gene_sets)
        gene_frequencies = calculate_gene_frequency(cluster, domain_gene_sets)

        # 按频率排序，保存完整列表
        core_genes_sorted = sorted([(gene, freq) for gene, freq in gene_frequencies.items()
                                    if freq >= Config.min_gene_frequency],
                                   key=lambda x: x[1], reverse=True)
        core_gene_names = [gene for gene, freq in core_genes_sorted]

        cluster_genes = set()
        for domain in cluster:
            if domain in domain_gene_sets:
                cluster_genes.update(domain_gene_sets[domain])
        cluster_spots = get_cluster_spots(cluster, raw_data)

        # 计算空间紧凑性
        coordinates = []
        for domain in cluster:
            if domain in raw_data and "coordinates" in raw_data[domain]:
                coords = raw_data[domain]["coordinates"]
                coordinates.append(coords)

        if coordinates:
            coordinates = np.array(coordinates)
            spatial_compactness = calculate_spatial_compactness(coordinates)
            spatial_center = np.mean(coordinates, axis=0)
            spatial_range = np.ptp(coordinates, axis=0)
        else:
            spatial_compactness = 0
            spatial_center = [0, 0]
            spatial_range = [0, 0]

        cluster_biology.append({
            'cluster_id': int(cluster_id),
            'size': int(len(cluster)),
            'gene_similarity': float(gene_similarity),
            'core_gene_count': int(len(core_gene_names)),
            'total_gene_count': int(len(cluster_genes)),
            'spot_count': int(len(cluster_spots)),
            'spatial_compactness': float(spatial_compactness),
            'spatial_center_x': float(spatial_center[0]) if len(spatial_center) > 0 else 0.0,
            'spatial_center_y': float(spatial_center[1]) if len(spatial_center) > 1 else 0.0,
            'spatial_range_x': float(spatial_range[0]) if len(spatial_range) > 0 else 0.0,
            'spatial_range_y': float(spatial_range[1]) if len(spatial_range) > 1 else 0.0,
            'core_genes': core_gene_names,  # 保存完整排序的核心基因列表
            'domains': list(cluster)
        })

    # 准备聚类分配结果
    cluster_results = []
    for cluster_id, cluster in enumerate(final_clusters):
        for domain in cluster:
            cluster_results.append({
                "domain_id": domain,
                "cluster": int(cluster_id),
                "cluster_type": "Clustered"
            })

    for domain in unclustered_domains:
        cluster_results.append({
            "domain_id": domain,
            "cluster": -1,
            "cluster_type": "Unclustered"
        })

    cluster_df = pd.DataFrame(cluster_results).merge(spot_coords, on="domain_id", how="left")

    # 保存统计信息 - 确保所有数值都是Python原生类型
    coverage_rate = len(clustered_domains) / len(all_domains) * 100 if len(all_domains) > 0 else 0
    avg_cluster_size = np.mean([len(c) for c in final_clusters]) if final_clusters else 0

    cluster_stats = {
        "total_domains": int(len(all_domains)),
        "clustered_domains": int(len(clustered_domains)),
        "unclustered_domains": int(len(unclustered_domains)),
        "coverage_rate": float(coverage_rate),
        "final_clusters_count": int(len(final_clusters)),
        "average_cluster_size": float(avg_cluster_size),
        "cluster_biology": cluster_biology
    }

    # 递归转换所有numpy类型为Python原生类型
    cluster_stats = convert_numpy_types(cluster_stats)

    with open(os.path.join(Config.output_dir, "tables", "community_detection_statistics.json"), "w") as f:
        json.dump(cluster_stats, f, indent=2)

    cluster_df.to_csv(
        os.path.join(Config.output_dir, "tables", "community_detection_clusters.csv"),
        index=False
    )

    # 保存聚类生物学信息
    if cluster_biology:
        biology_df = pd.DataFrame(cluster_biology)
        biology_df.to_csv(
            os.path.join(Config.output_dir, "tables", "cluster_biology.csv"),
            index=False
        )

    print(f"社区检测聚类完成：共{len(final_clusters)}个最终聚类，覆盖率{coverage_rate:.1f}%")

    # 输出聚类质量统计
    if cluster_biology:
        avg_gene_sim = np.mean([c['gene_similarity'] for c in cluster_biology])
        min_gene_sim = np.min([c['gene_similarity'] for c in cluster_biology])
        max_gene_sim = np.max([c['gene_similarity'] for c in cluster_biology])
        avg_core_genes = np.mean([c['core_gene_count'] for c in cluster_biology])
        avg_compactness = np.mean([c['spatial_compactness'] for c in cluster_biology])

        print(f"聚类质量统计:")
        print(f"  基因相似度: 平均{avg_gene_sim:.3f}, 最小{min_gene_sim:.3f}, 最大{max_gene_sim:.3f}")
        print(f"  核心基因数: 平均{avg_core_genes:.1f}")
        print(f"  空间紧凑性: 平均{avg_compactness:.3f}")

    return cluster_df, len(final_clusters), unclustered_domains, domain_gene_sets


# --------------------------
# 5. 聚类结果可视化
# --------------------------
def visualize_community_detection(cluster_results, output_dir):
    """可视化社区检测聚类结果"""
    print("正在可视化社区检测聚类结果...")

    # 空间分布图
    plt.figure(figsize=(12, 10))

    # 按聚类分组
    clustered_data = cluster_results[cluster_results["cluster"] != -1]
    unclustered_data = cluster_results[cluster_results["cluster"] == -1]

    unique_clusters = sorted(clustered_data["cluster"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # 绘制聚类区域
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = clustered_data[clustered_data["cluster"] == cluster_id]
        plt.scatter(cluster_data["x"], cluster_data["y"],
                    c=[colors[i]], s=50, alpha=0.8,
                    label=f"Cluster {cluster_id} ({len(cluster_data)})")

    # 绘制未聚类区域
    if len(unclustered_data) > 0:
        plt.scatter(unclustered_data["x"], unclustered_data["y"],
                    c='gray', s=30, alpha=0.4,
                    label=f"Unclustered ({len(unclustered_data)})")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Community Detection: Spatial Distribution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "community_detection_spatial.png"), dpi=300)
    plt.close()

    # 聚类大小分布图
    plt.figure(figsize=(10, 6))
    clustered_only = cluster_results[cluster_results["cluster"] != -1]
    if len(clustered_only) > 0:
        cluster_sizes = clustered_only.groupby("cluster").size()
        plt.bar(range(len(cluster_sizes)), cluster_sizes.values, color='skyblue', alpha=0.7)
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Microdomains")
        plt.title("Cluster Size Distribution")
        plt.xticks(range(len(cluster_sizes)), cluster_sizes.index)
        for i, v in enumerate(cluster_sizes.values):
            plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "cluster_size_distribution.png"), dpi=300)
    plt.close()

    print("社区检测聚类可视化完成")


# --------------------------
# 6. 功能富集分析
# --------------------------
def run_cluster_enrichment(cluster_biology, output_dir):
    """
    对每个聚类的核心基因进行富集分析
    """
    print("正在进行功能富集分析...")

    # 1. 检查本地GMT文件
    valid_gmts = {}
    for name, path in Config.local_gmt_files.items():
        if os.path.exists(path):
            valid_gmts[name] = path

    if not valid_gmts:
        print("错误: 没有有效的GMT文件，跳过富集分析。")
        return cluster_biology

    all_clusters_results_list = []

    for cluster in cluster_biology:
        cluster_id = cluster['cluster_id']
        core_genes = cluster['core_genes']

        # 即使只有3个基因也尝试跑，但如果太少可能完全没结果
        if len(core_genes) < 2:
            cluster['top_pathways'] = "基因过少"
            cluster['top_enriched_pathways'] = []  # 添加空列表
            continue

        print(f"  - 分析聚类 {cluster_id} (基因数: {len(core_genes)})...")
        current_cluster_dfs = []

        try:
            for db_name, gmt_path in valid_gmts.items():
                enr = gp.enrich(
                    gene_list=core_genes,
                    gene_sets=gmt_path,
                    outdir=None,
                    verbose=False
                )

                res = enr.results

                # 必须先检查 res 是否为 DataFrame，因为没有结果时它可能是 list
                if isinstance(res, pd.DataFrame) and not res.empty:
                    res['Source'] = db_name
                    sig_res = res[res['Adjusted P-value'] < Config.enrich_padj_threshold].copy()
                    if not sig_res.empty:
                        current_cluster_dfs.append(sig_res)

            if current_cluster_dfs:
                # 合并当前聚类结果
                merged_df = pd.concat(current_cluster_dfs)
                merged_df = merged_df.sort_values('Adjusted P-value')

                # ========== 提取核心富集信息 ==========
                top_pathways_core = []
                for _, row in merged_df.iterrows():
                    core_info = {
                        'term': str(row['Term']),
                        'source': row['Source'],
                        'adj_pvalue': float(row['Adjusted P-value']),
                        'genes': row['Genes'].split(';') if isinstance(row['Genes'], str) else row['Genes'],
                        'overlap_count': int(row['Overlap'].split('/')[0])  # 只保留重叠基因数
                    }
                    top_pathways_core.append(core_info)

                cluster['top_enriched_pathways'] = top_pathways_core
                # ========== 结束 ==========

                # 生成摘要字符串
                top_terms_list = []
                priority_order = ['Hallmark', 'KEGG', 'GO_BP']

                for src in priority_order:
                    if src in valid_gmts:
                        src_top = merged_df[merged_df['Source'] == src].head(1)
                        for _, row in src_top.iterrows():
                            # 安全处理 Term 字符串
                            term = str(row['Term']).split(' (GO:')[0]
                            top_terms_list.append(f"[{src}] {term}")

                # 如果优先列表里没找到，就用总排名的第一名
                if not top_terms_list and not merged_df.empty:
                    row = merged_df.iloc[0]
                    term = str(row['Term']).split(' (GO:')[0]
                    top_terms_list.append(f"[{row['Source']}] {term}")

                cluster['top_pathways'] = "; ".join(top_terms_list)

                # 添加 Cluster ID 并收集
                merged_df.insert(0, 'Cluster_ID', cluster_id)
                all_clusters_results_list.append(merged_df)
            else:
                cluster['top_pathways'] = "无显著富集"
                cluster['top_enriched_pathways'] = []  # 添加空列表

        except Exception as e:
            # 打印更详细的错误但不中断程序
            print(f"    警告: 聚类 {cluster_id} 分析遇到问题 (已跳过): {e}")
            cluster['top_pathways'] = "分析无结果"
            cluster['top_enriched_pathways'] = []  # 添加空列表

    # 保存汇总大表
    if all_clusters_results_list:
        master_df = pd.concat(all_clusters_results_list)
        save_path = os.path.join(output_dir, "tables", "all_clusters_enrichment_details.csv")
        master_df.to_csv(save_path, index=False)
        print(f"富集分析完成，详细结果已保存至: {save_path}")
    else:
        print("未发现显著的富集结果，未生成文件。")

    return cluster_biology


# --------------------------
# 7. 聚类详细信息输出
# --------------------------
def output_cluster_details(cluster_results, domain_gene_sets, raw_data, output_dir):
    """
    输出每个聚类的详细信息
    """
    print("正在生成聚类详细信息...")

    # 读取聚类统计信息
    stats_path = os.path.join(output_dir, "tables", "community_detection_statistics.json")
    if not os.path.exists(stats_path):
        print("警告: 聚类统计文件不存在，跳过详细报告生成")
        return pd.DataFrame()

    with open(stats_path, "r") as f:
        stats = json.load(f)

    cluster_biology = stats.get("cluster_biology", [])

    if len(cluster_biology) == 0:
        print("没有找到任何聚类，跳过详细报告生成")
        return pd.DataFrame()

    cluster_biology = run_cluster_enrichment(cluster_biology, output_dir)

    # 创建详细的聚类报告
    detailed_report = []

    for cluster_info in cluster_biology:
        cluster_id = cluster_info['cluster_id']
        size = cluster_info['size']
        gene_similarity = cluster_info['gene_similarity']
        core_gene_count = cluster_info['core_gene_count']
        total_gene_count = cluster_info['total_gene_count']
        spot_count = cluster_info['spot_count']
        spatial_compactness = cluster_info['spatial_compactness']
        core_genes = cluster_info['core_genes']

        # ========== 处理富集信息 ==========
        top_enriched_pathways = cluster_info.get('top_enriched_pathways', [])
        top_pathways_str = cluster_info.get('top_pathways', '')  # 原有的字符串摘要

        if top_enriched_pathways:
            # 生成人类可读的富集描述
            top_terms = []
            for pathway in top_enriched_pathways:
                term_clean = pathway['term'].split(' (GO:')[0]  # 清理GO术语
                top_terms.append(f"{term_clean} ({pathway['source']})")
            enrichment_summary = "; ".join(top_terms)

            # 提取核心富集信息用于JSON
            core_enrichment = [{
                'term': pathway['term'],
                'source': pathway['source'],
                'adj_pvalue': pathway['adj_pvalue'],
                'gene_count': pathway['overlap_count']
            } for pathway in top_enriched_pathways]
        else:
            enrichment_summary = top_pathways_str if top_pathways_str else "无显著富集"
            core_enrichment = []
        # ========== 结束 ==========

        # 获取该聚类的所有微域
        cluster_domains = cluster_info['domains']

        # 计算每个微域的特征基因数量
        domain_gene_counts = []
        for domain in cluster_domains:
            if domain in domain_gene_sets:
                gene_count = len(domain_gene_sets[domain])
                domain_gene_counts.append(gene_count)

        avg_genes_per_domain = np.mean(domain_gene_counts) if domain_gene_counts else 0

        # 判断聚类类型
        if gene_similarity > 0.3 and spatial_compactness > 0.7:
            cluster_type = "High Quality Module"
        elif gene_similarity > 0.2:
            cluster_type = "Gene-Consistent Module"
        elif spatial_compactness > 0.7:
            cluster_type = "Spatially Compact Module"
        else:
            cluster_type = "General Module"

        # 生成聚类描述
        if core_gene_count >= 10:
            gene_description = f"富含{core_gene_count}个核心基因"
        elif core_gene_count >= 5:
            gene_description = f"具有{core_gene_count}个核心基因"
        else:
            gene_description = f"仅有{core_gene_count}个核心基因"

        if size > 100:
            size_description = "大型聚类"
        elif size > 30:
            size_description = "中型聚类"
        else:
            size_description = "小型聚类"

        detailed_report.append({
            'cluster_id': cluster_id,
            'size': size,
            'gene_similarity': gene_similarity,
            'core_gene_count': core_gene_count,
            'total_gene_count': total_gene_count,
            'spot_count': spot_count,
            'avg_genes_per_domain': avg_genes_per_domain,
            'spatial_compactness': spatial_compactness,
            'spatial_range_x': cluster_info['spatial_range_x'],
            'spatial_range_y': cluster_info['spatial_range_y'],
            'spatial_center_x': cluster_info['spatial_center_x'],
            'spatial_center_y': cluster_info['spatial_center_y'],
            'cluster_type': cluster_type,
            'core_genes': core_genes,
            'enrichment_summary': enrichment_summary,  # 富集摘要
            'description': f"{size_description}，{gene_description}，{cluster_type}",
            'domains_sample': cluster_domains,
            'core_enrichment': core_enrichment  # 核心富集信息
        })

    # 保存详细报告
    detailed_df = pd.DataFrame(detailed_report)
    detailed_df.to_csv(
        os.path.join(output_dir, "tables", "cluster_detailed_report.csv"),
        index=False
    )

    # ========== 更新cluster_biology.csv ==========
    cluster_biology_for_csv = []
    for cluster in detailed_report:
        cluster_biology_for_csv.append({
            'cluster_id': cluster['cluster_id'],
            'size': cluster['size'],
            'gene_similarity': cluster['gene_similarity'],
            'core_gene_count': cluster['core_gene_count'],
            'total_gene_count': cluster['total_gene_count'],
            'spot_count': cluster['spot_count'],
            'spatial_compactness': cluster['spatial_compactness'],
            'spatial_center_x': cluster['spatial_center_x'],
            'spatial_center_y': cluster['spatial_center_y'],
            'spatial_range_x': cluster['spatial_range_x'],
            'spatial_range_y': cluster['spatial_range_y'],
            'core_genes': ', '.join(cluster['core_genes']) if cluster['core_genes'] else '',  # 将列表转为字符串
            'enrichment_summary': cluster['enrichment_summary'],  # 富集摘要
            'cluster_type': cluster['cluster_type']  # 聚类类型
        })

    cluster_biology_df = pd.DataFrame(cluster_biology_for_csv)
    cluster_biology_df.to_csv(
        os.path.join(output_dir, "tables", "cluster_biology.csv"),
        index=False
    )
    # ========== 结束 ==========

    # ========== 更新统计JSON文件 ==========
    updated_cluster_biology = []
    for cluster in detailed_report:
        updated_cluster_biology.append({
            'cluster_id': cluster['cluster_id'],
            'size': cluster['size'],
            'gene_similarity': cluster['gene_similarity'],
            'core_gene_count': cluster['core_gene_count'],
            'total_gene_count': cluster['total_gene_count'],
            'spot_count': cluster['spot_count'],
            'spatial_compactness': cluster['spatial_compactness'],
            'spatial_range_x': cluster['spatial_range_x'],
            'spatial_range_y': cluster['spatial_range_y'],
            'spatial_center_x': cluster['spatial_center_x'],
            'spatial_center_y': cluster['spatial_center_y'],
            'core_genes': cluster['core_genes'],
            'core_enrichment': cluster['core_enrichment'],  # 新增：核心富集信息
            'domains': cluster_info['domains']  # 从原始cluster_info获取
        })

    # 更新统计信息
    stats['cluster_biology'] = updated_cluster_biology
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    # ========== 结束 ==========

    # 在控制台输出聚类信息
    print("\n" + "=" * 80)
    print("聚类详细信息")
    print("=" * 80)

    for cluster in detailed_report:
        print(f"\n聚类 {cluster['cluster_id']}:")
        print(f"  - 大小: {cluster['size']} 个微域")
        print(f"  - 类型: {cluster['cluster_type']}")
        print(f"  - 基因相似度: {cluster['gene_similarity']:.3f}")
        print(f"  - 核心基因: {cluster['core_gene_count']} 个 (总基因: {cluster['total_gene_count']} 个)")
        print(f"  - 平均每个微域基因数: {cluster['avg_genes_per_domain']:.1f}")
        print(f"  - 包含spots: {cluster['spot_count']} 个")
        print(f"  - 空间紧凑性: {cluster['spatial_compactness']:.3f}")
        print(f"  - 空间范围: X={cluster['spatial_range_x']:.1f}, Y={cluster['spatial_range_y']:.1f}")
        print(f"  - 空间中心: ({cluster['spatial_center_x']:.1f}, {cluster['spatial_center_y']:.1f})")
        print(f"  - 富集功能: {cluster['enrichment_summary']}")
        print(f"  - 描述: {cluster['description']}")

    # 输出总体统计
    print("\n" + "=" * 80)
    print("聚类总体统计")
    print("=" * 80)

    total_clusters = len(detailed_report)
    if total_clusters == 0:
        print("没有有效的聚类")
        return detailed_df

    total_domains = sum(cluster['size'] for cluster in detailed_report)
    avg_gene_similarity = np.mean([cluster['gene_similarity'] for cluster in detailed_report])
    avg_core_genes = np.mean([cluster['core_gene_count'] for cluster in detailed_report])
    avg_compactness = np.mean([cluster['spatial_compactness'] for cluster in detailed_report])

    # 按类型统计
    type_counts = {}
    for cluster in detailed_report:
        cluster_type = cluster['cluster_type']
        type_counts[cluster_type] = type_counts.get(cluster_type, 0) + 1

    print(f"总聚类数: {total_clusters}")
    print(f"总微域数: {total_domains}")
    print(f"平均基因相似度: {avg_gene_similarity:.3f}")
    print(f"平均核心基因数: {avg_core_genes:.1f}")
    print(f"平均空间紧凑性: {avg_compactness:.3f}")
    print(f"聚类类型分布:")
    for cluster_type, count in type_counts.items():
        percentage = count / total_clusters * 100
        print(f"  - {cluster_type}: {count} 个 ({percentage:.1f}%)")

    # 按大小统计
    small_clusters = len([c for c in detailed_report if c['size'] < 30])
    medium_clusters = len([c for c in detailed_report if 30 <= c['size'] <= 100])
    large_clusters = len([c for c in detailed_report if c['size'] > 100])

    print(f"聚类大小分布:")
    print(f"  - 小型聚类 (<30微域): {small_clusters} 个 ({small_clusters / total_clusters * 100:.1f}%)")
    print(f"  - 中型聚类 (30-100微域): {medium_clusters} 个 ({medium_clusters / total_clusters * 100:.1f}%)")
    print(f"  - 大型聚类 (>100微域): {large_clusters} 个 ({large_clusters / total_clusters * 100:.1f}%)")

    return detailed_df


# --------------------------
# 8. 主程序
# --------------------------
def main():
    print("===== 肿瘤微域特征基因分析流程开始（社区检测聚类版） =====")

    # 1. 数据读取与预处理
    all_expr, spot_coords, raw_data = load_and_preprocess_data(Config.input_json_path)

    # 2. 特征基因筛选
    feature_genes, feature_genes_stats = filter_feature_genes(all_expr, raw_data=raw_data)
    if len(feature_genes) == 0:
        print("未筛选到符合条件的特征基因，分析终止")
        return

    # 3. 社区检测聚类分析
    cluster_results, num_clusters, unclustered_domains, domain_gene_sets = community_detection_clustering(
        raw_data, feature_genes, spot_coords
    )

    # 4. 可视化聚类结果
    if num_clusters == 0:
        print("警告: 没有生成任何聚类，将只输出未聚类的结果")
        # 创建一个只包含未聚类微域的结果
        cluster_results = []
        for domain_id in raw_data.keys():
            cluster_results.append({
                "domain_id": domain_id,
                "cluster": -1,
                "cluster_type": "Unclustered"
            })
        cluster_df = pd.DataFrame(cluster_results).merge(spot_coords, on="domain_id", how="left")
        cluster_df.to_csv(
            os.path.join(Config.output_dir, "tables", "community_detection_clusters.csv"),
            index=False
        )
    else:
        visualize_community_detection(cluster_results, Config.output_dir)

    # 5. 输出聚类详细信息
    detailed_report = output_cluster_details(cluster_results, domain_gene_sets, raw_data, Config.output_dir)

    print("===== 肿瘤微域特征基因分析流程完成（社区检测聚类版） =====")
    print(f"聚类结果已保存至: {os.path.abspath(Config.output_dir)}")

    # 读取并显示聚类统计
    stats_path = os.path.join(Config.output_dir, "tables", "community_detection_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        print(f"\n聚类统计:")
        print(f"  总微域数量: {stats['total_domains']}")
        print(f"  已聚类微域: {stats['clustered_domains']}")
        print(f"  未聚类微域: {stats['unclustered_domains']}")
        print(f"  聚类覆盖率: {stats['coverage_rate']:.1f}%")
        print(f"  最终聚类数量: {stats['final_clusters_count']}")
        print(f"  平均聚类大小: {stats['average_cluster_size']:.1f}")


if __name__ == "__main__":
    main()