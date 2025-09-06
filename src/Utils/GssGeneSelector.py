import pandas as pd
import numpy as np
import scanpy as sc
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score
from skimage import measure
from typing import List, Tuple, Optional
from scipy.spatial import distance
import scipy
import os
import SpatialDE as sd
import matplotlib.pyplot as plt
import seaborn as sns


class GssGeneSelector:
    """基于GSS分数的基因选择与验证工具"""

    def __init__(self,
                 adata: sc.AnnData,
                 gss_df: pd.DataFrame,
                 output_dir: str,
                 min_expr_threshold: float = 0.1,
                 min_gss_threshold: float = 0.5,
                 concentration_threshold = 90,
                 corr_threshold = 0.4,
                 entropy_threshold: float = 0.2,
                 morans_i_threshold: float = 0.3,
                 icc_threshold: float = 0.7,
                 ):
        """
        初始化基因选择器

        参数:
            adata: AnnData对象，包含基因表达矩阵和空间坐标
            gss_df: GSS分数DataFrame，行名为基因名，列名为样本名
            output_dir: 输出文件地址
            min_expr_threshold: 最小表达量阈值
            min_gss_threshold: GSS分数阈值
            concentration_threshold: 表达离散度和集中性阈值
            corr_threshold: 相关系数阈值
            entropy_threshold：GSS信息熵阈值
            morans_i_threshold: Moran's I指数阈值
            icc_threshold: 组内相关系数阈值
        """
        self.adata = adata
        self.gss_df = gss_df
        self.output_dir = output_dir
        self.min_expr_threshold = min_expr_threshold
        self.min_gss_threshold = min_gss_threshold
        self.concentration_threshold = concentration_threshold
        self.corr_threshold = corr_threshold
        self.entropy_threshold = entropy_threshold
        self.morans_i_threshold = morans_i_threshold
        self.icc_threshold = icc_threshold

        # 确保基因名称匹配
        common_genes = np.intersect1d(adata.var_names, gss_df.index)
        self.adata = adata[:, common_genes].copy()
        self.gss_df = gss_df.loc[common_genes]

        # 从 adata 中读取空间坐标
        if 'spatial' in adata.obsm:
            spatial_coords = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names, columns=['x', 'y'])
        else:
            raise ValueError("AnnData对象中未找到空间坐标信息，请确保 'spatial' 在 adata.obsm 中")

        # 确保空间坐标匹配
        common_samples = np.intersect1d(adata.obs_names, spatial_coords.index)
        self.adata = self.adata[common_samples].copy()
        self.spatial_coords = spatial_coords.loc[common_samples]

    def select_genes_by_expression(self) -> pd.Series:
        """基于表达量筛选基因"""
        # 计算每个基因的平均表达量
        if scipy.sparse.issparse(self.adata.X):
            mean_expr = np.array(self.adata.X.mean(axis=0)).flatten()
        else:
            mean_expr = np.mean(self.adata.X, axis=0)

        # 归一化处理
        mean_expr = (mean_expr - np.min(mean_expr)) / (np.max(mean_expr) - np.min(mean_expr))
        gene_expr = pd.Series(mean_expr, index=self.adata.var_names)

        # 筛选高于表达阈值的基因
        high_expr_genes = gene_expr[gene_expr > self.min_expr_threshold]
        return high_expr_genes

    def select_genes_by_gss(self) -> pd.Series:
        """基于GSS分数筛选基因"""
        # 计算每个基因非零值的平均GSS分数
        mean_gss = (self.gss_df.replace(0, pd.NA).mean(axis=1, skipna=True))

        # 归一化处理
        mean_gss = (mean_gss - np.min(mean_gss)) / (np.max(mean_gss) - np.min(mean_gss))

        # 筛选高于GSS阈值的基因
        high_gss_genes = mean_gss[mean_gss > self.min_gss_threshold]
        return high_gss_genes

    def calculate_spatial_reproducibility(self, genes: List[str]) -> pd.DataFrame:
        """
        计算基因空间表达的重复性（跨样本一致性）

        参数:
            genes: 待计算的基因列表

        返回:
            重复性结果DataFrame
        """
        if len(self.adata.obs['sample'].unique()) < 2:
            print("警告：数据中只有一个样本，无法计算空间重复性")
            return pd.DataFrame(columns=['gene', 'icc', 'spatial_purity'])

        results = []

        for gene in genes:
            # 获取基因在各样本中的表达矩阵
            expr_by_sample = {}
            for sample in self.adata.obs['sample'].unique():
                sample_cells = self.adata.obs_names[self.adata.obs['sample'] == sample]
                expr = self.adata[sample_cells, gene].X
                if scipy.sparse.issparse(expr):
                    expr = np.array(expr).flatten()
                expr_by_sample[sample] = expr

            # 计算ICC（简化版，实际应使用专门的ICC计算函数）
            # 这里用样本间表达相关性的平均值近似
            corr_matrix = np.zeros((len(expr_by_sample), len(expr_by_sample)))
            samples = list(expr_by_sample.keys())

            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    corr, _ = stats.pearsonr(expr_by_sample[samples[i]], expr_by_sample[samples[j]])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

            # 计算平均相关性作为ICC近似
            icc = np.mean(corr_matrix[corr_matrix > 0])

            # 计算空间纯度（如果有空间聚类标签）
            if 'spatial_cluster' in self.adata.obs.columns:
                cluster_labels = self.adata.obs['spatial_cluster']
                expr = self.adata[:, gene].X
                if scipy.sparse.issparse(expr):
                    expr = expr.toarray().flatten()

                # 计算每个聚类内的平均表达
                cluster_expr = {}
                for cluster in cluster_labels.unique():
                    cluster_cells = cluster_labels == cluster
                    cluster_expr[cluster] = np.mean(expr[cluster_cells])

                # 找到表达最高的聚类
                max_cluster = max(cluster_expr, key=cluster_expr.get)
                max_expr = cluster_expr[max_cluster]

                # 计算纯度（最高聚类表达占总表达的比例）
                total_expr = np.sum(expr)
                purity = max_expr / total_expr if total_expr > 0 else 0
            else:
                purity = np.nan

            results.append({
                'gene': gene,
                'icc': icc,
                'spatial_purity': purity
            })

        return pd.DataFrame(results)

    def calculate_spatial_gene_qval(self, genes: List[str]) -> pd.DataFrame:
        expr = self.adata[:, genes].to_df()
        gss = self.gss_df.loc[genes].T
        coords = self.spatial_coords.values
        # 运行空间差异表达分析
        results = sd.run(coords, expr)

        return results

    def calculate_expression_concentration(self, genes: list) -> pd.DataFrame:
        """
        计算基因的表达集中性综合评分（融合离散指数和表达占比集中度）
        参数：
            genes: 待分析的基因列表
        返回：包含原始指标和综合评分的DataFrame
        """
        top_k = 0.05  # 前5%细胞的表达占比
        results = []

        for gene in genes:
            # 1. 提取基因表达量
            if scipy.sparse.issparse(self.adata.X):
                expr = self.adata[:, gene].X.A.flatten()
            else:
                expr = self.adata[:, gene].X.flatten()

            # 2. 计算表达离散指数（VMR）
            mean_expr = np.mean(expr)
            var_expr = np.var(expr)
            if mean_expr < 1e-10:
                disp_index = np.inf
            else:
                disp_index = var_expr / mean_expr

            # 3. 计算表达占比集中度
            sorted_expr = np.sort(expr)[::-1]
            top_n = max(1, int(len(expr) * top_k))
            top_sum = np.sum(sorted_expr[:top_n])
            total_sum = np.sum(sorted_expr)
            concentration_ratio = top_sum / total_sum if total_sum >= 1e-10 else 0.0

            results.append({
                'gene': gene,
                'dispersion_index': disp_index,
                'concentration_ratio': concentration_ratio
            })

        # 转换为DataFrame并处理极端值
        df = pd.DataFrame(results)

        # 4. 指标归一化（映射到0-1范围）
        # 离散指数：值越大越好（取倒数后归一化，避免inf影响）
        df['dispersion_norm'] = 1 / (1 + df['dispersion_index'])  # 转换为0-1（值越大越集中）
        # 集中率：值越大越好（直接归一化）
        cr_max, cr_min = df['concentration_ratio'].max(), df['concentration_ratio'].min()
        df['concentration_norm'] = (df['concentration_ratio'] - cr_min) / (cr_max - cr_min + 1e-10)  # 0-1归一化

        # 5. 融合为综合评分（采用排名平均，避免主观权重）
        # 对两个归一化指标分别排名（升序=1,2,3...）
        df['disp_rank'] = df['dispersion_norm'].rank(ascending=False)  # 高离散归一值排名靠前
        df['cr_rank'] = df['concentration_norm'].rank(ascending=False)  # 高集中率排名靠前
        # 综合评分为排名的平均值（值越小表示综合表现越好）
        df['combined_score'] = (df['disp_rank'] + df['cr_rank']) / 2

        # 可选：将综合评分转换为0-100的分数（越高越好）
        max_score = df['combined_score'].max()
        df['concentration_score'] = 100 * (1 - df['combined_score'] / max_score)

        return df[['gene', 'concentration_score']]

    def calculate_gss_expression_correlation(self, genes: List[str]) -> pd.DataFrame:
        """
        计算GSS分数与表达量的相关性

        参数:
            genes: 待计算的基因列表

        返回:
            相关性结果DataFrame
        """
        results = []

        for gene in genes:
            # 获取基因表达量
            expr = self.adata[:, gene].X
            if scipy.sparse.issparse(expr):
                # 使用 toarray() 方法将稀疏矩阵转换为密集矩阵，然后展平
                expr = expr.toarray().flatten()

            # 获取GSS分数
            gss = self.gss_df.loc[gene].values

            # 计算皮尔逊相关系数和p值
            corr, p_value = stats.spearmanr(expr, gss)

            results.append({
                'gene': gene,
                'gss_expr_corr': corr,
                'gss_expr_p_value': p_value
            })

        return pd.DataFrame(results)

    def calculate_genes_entropy(self, genes: List[str]) -> pd.DataFrame:
        """
        计算考虑空间距离的加权信息熵（熵值越低，表达越集中且空间聚集性越高）
        权重基于六边形网格邻居关系，距离越近的细胞对熵值计算贡献越大

        参数:
            genes: 待计算的基因列表

        返回:
            DataFrame包含基因名和信息熵值，仅保留熵值低于阈值的基因
        """
        # 预计算空间权重矩阵（基于六边形网格结构）
        # 权重矩阵维度: [n_cells, n_cells]，值越大表示空间距离越近
        coords = self.spatial_coords.values
        spatial_weights = self._calculate_hexagonal_weights(coords=coords)

        results = []
        for gene in genes:
            # 获取基因表达量（处理稀疏矩阵）
            expr = self.adata[:, gene].X
            if scipy.sparse.issparse(expr):
                expr = expr.toarray().flatten()
            else:
                expr = expr.flatten()

            # 1. 先保留真实表达值，真实零表达直接设为 0（区分生物学零和技术噪声）
            expr = np.where(expr > 0, expr, 0)

            # 2. 计算加权表达量（融入邻居信息）
            weighted_expr = spatial_weights @ expr
            expr_threshold = np.max(weighted_expr)/50
            weighted_expr[weighted_expr < expr_threshold] = 0

            # 可视化
            #------------------------
            # # 计算基因的加权表达量分布
            # n, bins, patches = plt.hist(weighted_expr, bins=50)
            # plt.title(f"{gene}的加权表达量分布")
            # plt.show()
            #
            # # 随机选一个高表达细胞（加权值>0.5），检查其邻居的原始表达
            # high_idx = np.argmax(weighted_expr > 0.5)
            # neighbor_indices = np.nonzero(spatial_weights[high_idx])[0]  # 获取该细胞的邻居
            #
            # print("高表达细胞的原始表达:", expr[high_idx])
            # print("邻居的原始表达均值:", expr[neighbor_indices].mean())
            # print("邻居的加权表达贡献:", spatial_weights[high_idx, neighbor_indices] @ expr[neighbor_indices])
            #
            # # 绘制基因表达的空间分布（原始表达 vs 加权表达）
            # plt.figure(figsize=(12, 5))
            # plt.subplot(1, 2, 1)
            # sns.scatterplot(x=self.spatial_coords['x'], y=self.spatial_coords['y'], hue=expr, palette='viridis')
            # plt.title("原始表达的空间分布")
            #
            # plt.subplot(1, 2, 2)
            # sns.scatterplot(x=self.spatial_coords['x'], y=self.spatial_coords['y'], hue=weighted_expr,
            #                 palette='viridis')
            # plt.title("加权表达的空间分布")
            # plt.show()
            #
            # # 随机选一个细胞，可视化其权重分布
            # cell_idx = np.random.choice(len(self.spatial_coords))
            # cell_weight = spatial_weights[cell_idx]
            #
            # # 绘制该细胞的邻居权重空间分布
            # neighbor_coords = self.spatial_coords.iloc[np.nonzero(cell_weight)[0]]
            # neighbor_weights = cell_weight[np.nonzero(cell_weight)]
            #
            # plt.figure(figsize=(8, 6))
            # sns.scatterplot(x=self.spatial_coords['x'], y=self.spatial_coords['y'], color='gray', alpha=0.2)
            # sns.scatterplot(x=neighbor_coords['x'], y=neighbor_coords['y'], hue=neighbor_weights, palette='viridis')
            # plt.title(f"细胞 {cell_idx} 的空间权重分布")
            # plt.legend(title='权重值')
            # plt.show()
            #
            # # 计算每个细胞的概率占比，并排序
            # prob = weighted_expr / weighted_expr.sum()
            # sorted_prob = np.sort(prob)[::-1]  # 降序排列
            #
            # # 绘制概率分布的“长尾图”
            # plt.plot(sorted_prob, color='blue')
            # plt.yscale('log')  # 对数轴更易观察长尾
            # plt.title("加权概率分布的长尾")
            # plt.show()
            # ------------------------

            # 3. 归一化时处理零值，避免除零错误
            total_weighted = np.sum(weighted_expr)
            if total_weighted < 1e-10:  # 所有表达都是 0（极端情况）
                weighted_prob = np.zeros_like(weighted_expr)
            else:
                weighted_prob = weighted_expr / total_weighted
                # 强制极低值为 0，还原真实零表达的影响
                weighted_prob[weighted_prob < 1e-10] = 0

            # 4. 计算加权信息熵H = -Σp*log2(p)时，给 log 加保护（避免零值的 log 错误）
            weighted_entropy = -np.sum(
                weighted_prob * np.log2(weighted_prob + 1e-10)
            )

            results.append({
                'gene': gene,
                'entropy': weighted_entropy
            })

        return pd.DataFrame(results)

    def calculate_morans_i(self, genes: List[str]) -> pd.DataFrame:
        """
        计算基因表达的Moran's I指数（空间自相关性）

        参数:
            genes: 待计算的基因列表

        返回:
            Moran's I结果DataFrame
        """
        coords = self.spatial_coords.values
        n = len(coords)

        # # 1. 高效计算距离矩阵
        # dist_matrix = distance.squareform(distance.pdist(coords))
        # np.fill_diagonal(dist_matrix, np.inf)
        #
        # # 2. 构建权重矩阵 (距离倒数)
        # w = 1 / np.maximum(dist_matrix, 1e-10)  # 避免除0
        # np.fill_diagonal(w, 0)  # 对角线置0

        #---------------

        # # 1. 高效计算距离矩阵
        # dist_matrix = distance.squareform(distance.pdist(coords))
        # np.fill_diagonal(dist_matrix, np.inf)
        #
        # # 2. 构建权重矩阵
        # bandwidth = np.median(dist_matrix[dist_matrix < np.inf])
        # w = np.exp(-dist_matrix ** 2 / (2 * bandwidth ** 2))
        # np.fill_diagonal(w, 0)
        #
        # # 3. 行标准化 (处理孤立点)
        # w_rowsum = w.sum(axis=1)
        # isolated = w_rowsum == 0
        # if np.any(isolated):
        #     print(f"警告: {isolated.sum()}个孤立点存在")
        #     w_rowsum[isolated] = 1  # 避免除0
        # w /= w_rowsum[:, None]

        w = self._calculate_hexagonal_weights(coords=coords)

        # 4. 预计算全局常量
        results = []
        total_weight = w.sum()

        for gene in genes:
            expr = self.adata[:, gene].X
            expr = np.arcsinh(expr)  # 或 np.log1p(expr)

            if scipy.sparse.issparse(expr):
                expr = expr.toarray().flatten()

            # 5. 计算Moran's I
            mean_expr = np.mean(expr)
            deviations = expr - mean_expr

            numerator = np.sum(w * np.outer(deviations, deviations))
            denominator = np.sum(deviations ** 2)

            if denominator < 1e-10:  # 处理零方差
                morans_i = np.nan
            else:
                morans_i = (n / total_weight) * (numerator / denominator)

            results.append({
                'gene': gene,
                'morans_i': morans_i
            })

        return pd.DataFrame(results)

    def _calculate_hexagonal_weights(self, coords, max_rings=2, decay_rate=0.1, tolerance=0.05):
        """
        为六边形网格结构计算空间权重，支持多环邻居

        参数:
            coords: 细胞坐标数组 (n_samples, 2)
            max_rings: 要考虑的最大环数（默认为2，即第一环和第二环）
            decay_rate: 每环权重的衰减率（0-1之间）
            tolerance: 距离容差系数，用于识别邻居

        返回:
            w: 空间权重矩阵 (n_samples, n_samples)
        """
        n = len(coords)
        w = np.zeros((n, n))

        # 计算距离矩阵
        dist_matrix = distance.squareform(distance.pdist(coords))
        np.fill_diagonal(dist_matrix, np.inf)  # 避免自环

        # 确定第一近邻距离（六边形边长）
        first_neighbor_dist = np.min(dist_matrix)

        # 预计算各环的理论距离阈值（第n环距离≈n×边长）
        ring_thresholds = [(i + 1) * first_neighbor_dist for i in range(max_rings)]

        # 为每个点分配权重
        for i in range(n):
            # 初始化已分配的邻居集合
            assigned_neighbors = set()

            # 按环依次处理
            for ring in range(max_rings):
                # 确定当前环的距离范围
                lower_bound = ring_thresholds[ring - 1] * (1 + tolerance) if ring > 0 else 0
                upper_bound = ring_thresholds[ring] * (1 + tolerance)

                # 找出当前环的邻居
                current_ring_neighbors = np.where(
                    (dist_matrix[i] > lower_bound) &
                    (dist_matrix[i] <= upper_bound)
                )[0]

                # 排除已分配给前面环的邻居
                current_ring_neighbors = [j for j in current_ring_neighbors if j not in assigned_neighbors]

                # 为当前环邻居分配权重（按环数衰减）
                if current_ring_neighbors:
                    weight = decay_rate ** ring  # 权重衰减公式：decay_rate的环数次幂
                    w[i, current_ring_neighbors] = weight / len(current_ring_neighbors)

                    # 将这些邻居添加到已分配集合
                    assigned_neighbors.update(current_ring_neighbors)

        # 行标准化（确保每行权重和为1）
        row_sums = w.sum(axis=1)
        row_sums[row_sums == 0] = 1.0  # 防止除零
        w /= row_sums[:, np.newaxis]

        return w

    def run_pipeline(self) -> Tuple[List[str], pd.DataFrame]:
        """
        运行完整的基因选择与验证流程

        返回:
            最终选择的基因列表和验证结果DataFrame
        """
        # 1. 基于表达量和GSS分数筛选基因
        high_expr_genes = self.select_genes_by_expression()
        high_gss_genes = self.select_genes_by_gss()

        # 2. 整合两种指标选择基因
        initial_genes = list(
            set(high_expr_genes.index) &
            set(high_gss_genes.index)
        )

        print(f"初步筛选出 {len(initial_genes)} 个基因")

        # ---------
        # 运行空间差异表达分析
        # spatial_results = self.calculate_spatial_gene_qval(initial_genes)
        # # 筛选显著的空间特异性基因（q值<0.05）
        # high_spatialde_genes = spatial_results[spatial_results['qval'] < 0.05].index.tolist()
        # ---------

        # 3.0 计算表达离散度和集中性指标
        concentration_results = self.calculate_expression_concentration(initial_genes)
        high_concentration_genes = concentration_results[concentration_results['concentration_score'] > self.concentration_threshold]['gene'].tolist()
        print(f"基于表达离散度和集中性，保留 {len(high_concentration_genes)} 个基因")

        # 3.1 计算GSS与表达量的相关性
        corr_results = self.calculate_gss_expression_correlation(initial_genes)
        high_corr_genes = corr_results[corr_results['gss_expr_corr'] > self.corr_threshold]['gene'].tolist()
        print(f"基于GSS-表达量相关性，保留 {len(high_corr_genes)} 个基因")

        # 3.2 计算GSS的信息熵
        entropy_results = self.calculate_genes_entropy(initial_genes)
        high_entropy_genes = entropy_results[entropy_results['entropy'] < self.entropy_threshold]['gene'].tolist()
        print(f"基于GSS的信息熵，保留 {len(high_entropy_genes)} 个基因")

        # 3.3 计算空间自相关性
        morans_i_results = self.calculate_morans_i(initial_genes)
        high_spatial_genes = morans_i_results[morans_i_results['morans_i'] > self.morans_i_threshold]['gene'].tolist()
        print(f"基于空间自相关性，保留 {len(high_spatial_genes)} 个基因")

        # 3.4 计算空间重复性（如果有多个样本）
        if 'sample' in self.adata.obs.columns and len(self.adata.obs['sample'].unique()) >= 2:
            reproducibility_results = self.calculate_spatial_reproducibility(initial_genes)
            reliable_genes = reproducibility_results[reproducibility_results['icc'] > self.icc_threshold][
                'gene'].tolist()
            print(f"基于空间重复性，保留 {len(reliable_genes)} 个基因")
        else:
            print("警告：数据中缺少样本批次信息，跳过空间重复性验证")
            reliable_genes = initial_genes

        # 4. 整合所有验证结果
        # 取所有通过至少一项验证的基因
        all_validated_genes = list(
            set(high_concentration_genes) |
            set(high_corr_genes) |
            set(high_entropy_genes) |
            set(high_spatial_genes)
        )

        # 构建完整的验证结果DataFrame
        all_results = pd.DataFrame({
            'gene': all_validated_genes,
            'pass_concentration': [g in high_concentration_genes for g in all_validated_genes],
            'pass_gss_expr_corr': [g in high_corr_genes for g in all_validated_genes],
            'pass_entropy': [g in high_entropy_genes for g in all_validated_genes],
            'pass_spatial_auto_corr': [g in high_spatial_genes for g in all_validated_genes],
        })

        # 计算每个基因通过的验证数量
        all_results['count'] = all_results.iloc[:, 1:].sum(axis=1)

        # 提取每个基因的表达离散度和集中性归一化分数
        concentration_dict = {row['gene']: round(row['concentration_score'], 2) for _, row in concentration_results.iterrows()}
        all_results['concentration_score'] = all_results['gene'].map(concentration_dict)

        # 提取每个基因的GSS-表达量相关性
        corr_dict = {row['gene']: round(row['gss_expr_corr'], 2) for _, row in corr_results.iterrows()}
        all_results['gss_expr_corr'] = all_results['gene'].map(corr_dict)

        # 提取每个基因的信息熵分数
        entropy_dict = {row['gene']: round(row['entropy'], 2) for _, row in entropy_results.iterrows()}
        all_results['entropy'] = all_results['gene'].map(entropy_dict)

        # 提取每个基因的空间自相关系数
        morans_i_dict = {row['gene']: round(row['morans_i'], 2) for _, row in morans_i_results.iterrows()}
        all_results['morans_i'] = all_results['gene'].map(morans_i_dict)

        # 按验证数量和相关性排序
        all_results = all_results.sort_values(
            ['count', 'concentration_score', 'gss_expr_corr', 'entropy', 'morans_i'],
            ascending=[False, False, False, False, False]
        )

        # 通过所有验证的基因
        cross_genes = all_results[
            (all_results['pass_concentration']) &
            (all_results['pass_gss_expr_corr']) &
            (all_results['pass_entropy']) &
            (all_results['pass_spatial_auto_corr'])
            ]['gene'].tolist()
        print(f"一共有{len(cross_genes)}个基因通过了全部的筛选流程：{cross_genes}")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        all_results.to_csv(self.output_dir+"_selected_genes.csv", index=False, sep='\t')
        print(f"验证结果已保存至 {self.output_dir}")

        return all_validated_genes, all_results