
import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import distance_matrix
from scipy.stats import percentileofscore



class GssGeneCalculator:

    def __init__(self,
                 adata: sc.AnnData,
                 gene_counts: dict[str, int],
                 spatial_top_pct: int = 5,
                 spatial_threshold: float = 0.1,
                 cluster_threshold: float = 0.8
                 ):
        self.adata = adata
        self.gene_counts = gene_counts
        self.spatial_top_pct = spatial_top_pct
        self.spatial_threshold = spatial_threshold
        self.cluster_threshold = cluster_threshold

    def _get_spatial_expression_range(self, adata, gene_name, top_pct=5):
        """
        计算单个基因的高表达细胞占比
        """
        if gene_name not in adata.var_names:
            raise ValueError(f"基因 {gene_name} 不在数据中")

        # 提取基因表达值（处理稀疏矩阵）
        gene_data = adata[:, gene_name].X
        # 将稀疏矩阵转换为稠密数组并展平
        if hasattr(gene_data, 'toarray'):  # 检查是否为稀疏矩阵
            expr_values = gene_data.toarray().flatten()
        else:
            expr_values = gene_data.flatten()

        n_total = len(expr_values)  # 总细胞数

        # 确定高表达阈值
        top_threshold = np.percentile(expr_values, 100 - top_pct)

        # 筛选高表达细胞
        high_expr_mask = expr_values >= top_threshold
        n_high = np.sum(high_expr_mask)

        # 计算高表达占比
        ratio = n_high / n_total

        return ratio

    def _calculate_spatial_clustering(self, adata, gene_name, top_pct=5, n_permutations=100):
        """
        计算基因高表达细胞的空间聚集性（基于最邻近距离的聚集指数）

        参数:
            adata: AnnData对象，包含空间转录组数据
            gene_name: 目标基因名称
            top_pct: 定义高表达的百分比（前top_pct%）
            n_permutations: 随机置换次数，用于计算期望最邻近距离
        返回:
            clustering_index: 聚集指数（实际平均距离/期望平均距离，<1表示聚集）
            p_value: 聚集性显著性P值
        """
        # 1. 提取基因表达值和空间坐标
        if gene_name not in adata.var_names:
            raise ValueError(f"基因 {gene_name} 不在数据中")

        # 表达值处理（支持稀疏矩阵）
        expr_values = adata[:, gene_name].X
        if hasattr(expr_values, 'toarray'):
            expr_values = expr_values.toarray().flatten()
        else:
            expr_values = expr_values.flatten()

        # 空间坐标
        spatial_coords = adata.obsm['spatial']
        n_total = len(expr_values)

        # 2. 筛选高表达细胞
        top_threshold = np.percentile(expr_values, 100 - top_pct)
        high_expr_mask = expr_values >= top_threshold
        high_expr_coords = spatial_coords[high_expr_mask]
        n_high = len(high_expr_coords)

        if n_high < 10:  # 细胞数量太少无法计算聚集性
            return 1.0, 1.0  # 默认返回非聚集

        # 3. 计算实际最邻近距离（每个高表达细胞到最近高表达细胞的平均距离）
        dist_matrix = distance_matrix(high_expr_coords, high_expr_coords)
        np.fill_diagonal(dist_matrix, np.inf)  # 排除自身距离
        nearest_distances = np.min(dist_matrix, axis=1)
        actual_mean = np.mean(nearest_distances)

        # 4. 通过随机置换计算期望最邻近距离（零假设：随机分布）
        permuted_means = []
        for _ in range(n_permutations):
            # 随机从所有细胞中抽取相同数量的细胞
            random_indices = np.random.choice(n_total, size=n_high, replace=False)
            random_coords = spatial_coords[random_indices]

            # 计算随机细胞的最邻近距离
            perm_dist_matrix = distance_matrix(random_coords, random_coords)
            np.fill_diagonal(perm_dist_matrix, np.inf)
            perm_nearest = np.min(perm_dist_matrix, axis=1)
            permuted_means.append(np.mean(perm_nearest))

        # 5. 计算聚集指数和P值
        expected_mean = np.mean(permuted_means)
        clustering_index = actual_mean / expected_mean  # <1表示聚集

        # P值：随机置换中均值 <= 实际均值的比例（聚集性越显著，P值越小）
        p_value = percentileofscore(permuted_means, actual_mean) / 100.0

        return clustering_index, p_value

    def filter_restricted_and_clustered_genes(self, adata, candidate_genes, top_pct=5,
                                              spatial_threshold=0.1, cluster_threshold=0.8):
        """
        筛选同时满足：
        1. 空间范围受限（高表达占比 < spatial_threshold）
        2. 高表达细胞呈聚集分布（聚集指数 < cluster_threshold）的基因
        """
        results = []
        for gene in candidate_genes:
            try:
                # 计算高表达占比
                ratio = self._get_spatial_expression_range(adata, gene, top_pct)
                # 计算聚集指数
                cluster_idx, p_val = self._calculate_spatial_clustering(adata, gene, top_pct)

                # 判断是否符合条件
                is_restricted = ratio < spatial_threshold
                is_clustered = cluster_idx < cluster_threshold and p_val < 0.05  # 聚集性显著
                is_cross = is_restricted and is_clustered

                results.append({
                    "gene": gene,
                    "high_expr_ratio": ratio,
                    "clustering_index": cluster_idx,
                    "p_value": p_val,
                    "is_restricted": is_restricted,
                    "is_clustered": is_clustered,
                    "is_cross": is_cross
                })
                # print(f"基因 {gene}: 高表达占比 {ratio:.2%}, 聚集指数 {cluster_idx:.3f}, P值 {p_val:.3f}, {'合格' if is_cross else '不合格'}")
            except ValueError as e:
                print(f"跳过基因 {gene}: {str(e)}")

        result_df = pd.DataFrame(results)
        qualified_genes = result_df[result_df["is_cross"]]["gene"].tolist()
        return qualified_genes, result_df

    def run_pipeline(self):
        print("=== 第一步：基于频率的硬阈值筛选 ===")
        cell_count = self.adata.n_obs
        init_threshold = cell_count * 0.01
        freq_filtered_genes = [key for key, value in self.gene_counts.items() if value >= init_threshold]

        print(f"\n=== 第二步：空间范围+聚集性筛选 ===")
        final_genes, result_df = self.filter_restricted_and_clustered_genes(
            self.adata,
            freq_filtered_genes,
            top_pct=self.spatial_top_pct,
            spatial_threshold=self.spatial_threshold,
            cluster_threshold=self.cluster_threshold
        )

        # 合并结果
        count_df = pd.DataFrame(list(self.gene_counts.items()), columns=['gene', 'frequency'])
        results = pd.merge(count_df, result_df, on='gene', how='left')

        print(f"\n=== 筛选结果总结 ===")
        print(f"初始基因数量: {len(self.gene_counts)}")
        print(f"频率筛选后保留: {len(freq_filtered_genes)}")
        print(f"空间范围+聚集性筛选后最终保留: {len(final_genes)}")

        return final_genes, results