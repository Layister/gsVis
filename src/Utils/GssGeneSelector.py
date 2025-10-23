import pandas as pd
import numpy as np
import scanpy as sc
from scipy import stats
from typing import List, Tuple, Optional
from scipy.spatial import distance
import scipy
import os



class GssGeneSelector:
    """基于GSS分数的基因选择与验证工具"""

    def __init__(self,
                 adata: sc.AnnData,
                 gss_df: pd.DataFrame,
                 output_dir: str,
                 concentration_threshold=80,
                 corr_threshold=0.6,
                 morans_i_threshold: float = 0.6,
                 icc_threshold: float = 0.6,
                 ):
        """
        初始化基因选择器

        参数:
            adata: AnnData对象，包含基因表达矩阵和空间坐标
            gss_df: GSS分数DataFrame，行名为基因名，列名为样本名
            output_dir: 输出文件地址
            concentration_threshold: 表达离散度和集中性阈值
            corr_threshold: 相关系数阈值
            morans_i_threshold: Moran's I指数阈值
            icc_threshold: 组内相关系数阈值
        """
        self.adata = adata
        self.gss_df = gss_df
        self.output_dir = output_dir
        self.concentration_threshold = concentration_threshold
        self.corr_threshold = corr_threshold
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
        self.cells = self.adata.X.shape[0]

        # 预处理表达矩阵为密集矩阵（如果是稀疏矩阵）
        if scipy.sparse.issparse(self.adata.X):
            self._expr_matrix = self.adata.X.toarray()
        else:
            self._expr_matrix = self.adata.X.copy()
        self._gene_names = self.adata.var_names.tolist()
        self._gene_index = {gene: i for i, gene in enumerate(self._gene_names)}

    def select_genes_by_expression(self) -> pd.Series:
        """
        基于表达量筛选基因：同时过滤低表达（覆盖率）和低变异（变异系数）的基因
        """
        expr_df = self._expr_matrix.T  # 转置为 (gene×spot) 以提高处理效率
        n_genes, n_spots = expr_df.shape

        # 定义极低表达阈值（自身表达的10%分位数）、覆盖率阈值、变异系数阈值
        low_expr_percentile = 10  # 用于计算自身极低表达阈值的分位数
        expr_threshold = 0.1  # 最小覆盖率阈值
        cv_threshold = 0.5  # 变异系数阈值

        # 1. 计算每个基因的low_expr_percentile分位数（非零值）
        non_zero_mask = expr_df > 0
        q_n = np.zeros(n_genes)
        for i in range(n_genes):
            non_zero_vals = expr_df[i, non_zero_mask[i]]
            q_n[i] = np.percentile(non_zero_vals, low_expr_percentile) if non_zero_vals.size > 0 else 0

        # 2. 计算表达覆盖率：表达量高于自身极低阈值的spot比例
        coverage = (expr_df > q_n[:, np.newaxis]).mean(axis=1)

        # 3. 计算变异系数（CV=标准差/均值）：衡量表达变异度
        mean_expr = expr_df.mean(axis=1)
        std_expr = expr_df.std(axis=1)
        cv = np.divide(std_expr, mean_expr, out=np.zeros_like(std_expr), where=mean_expr > 1e-10)

        # 4. 筛选基因
        high_quality_mask = (coverage > expr_threshold) & (cv > cv_threshold)
        high_quality_genes = np.array(self._gene_names)[high_quality_mask]

        high_quality_genes = pd.Series(high_quality_genes)
        print(f"表达筛选后保留基因数：{len(high_quality_genes)}")
        return high_quality_genes

    def select_genes_by_gss(self) -> pd.Series:
        """
        基于GSS分数筛选基因：同时过滤 低均值 和 高零值占比 的基因
        """
        # 1. 计算每个基因非零值的平均GSS分数（含NaN，对应全零基因），并过滤有效（非NaN）的mean_gss
        mean_gss = self.gss_df.replace(0, pd.NA).mean(axis=1, skipna=True)
        valid_mean_gss = mean_gss.dropna()

        # 2. 仅对有效基因计算排名，再映射回原索引（NaN基因保持NaN）
        normalized_valid = pd.Series(stats.rankdata(valid_mean_gss) / len(mean_gss), index=valid_mean_gss.index)
        normalized_gss = normalized_valid.reindex(mean_gss.index)

        # 3. 筛选条件
        zero_cutoff = 0.95 # 允许零值比例小于 zero_cutoff
        zero_ratio = (self.gss_df == 0).mean(axis=1)
        low_gss_percentile = 10 # 要求均值大于 low_gss_percentile%
        mean_cutoff = np.percentile(normalized_gss.dropna(), low_gss_percentile)

        # 4. 同时满足两个条件
        mask = (normalized_gss > mean_cutoff) & (zero_ratio <= zero_cutoff)
        high_gss_genes = normalized_gss[mask]

        print(f"GSS筛选后保留基因数：{len(high_gss_genes)}")
        return high_gss_genes

    def calculate_spatial_reproducibility(self, genes: List[str]) -> pd.DataFrame:
        """
        计算基因空间表达的重复性（跨样本一致性）
        """
        if len(self.adata.obs['sample'].unique()) < 2:
            print("警告：数据中只有一个样本，无法计算空间重复性")
            return pd.DataFrame(columns=['gene', 'icc', 'spatial_purity'])

        results = []
        samples = self.adata.obs['sample'].unique()
        sample_masks = {s: self.adata.obs['sample'] == s for s in samples}
        n_samples = len(samples)

        # 批量获取基因索引
        gene_indices = [self._gene_index[gene] for gene in genes]
        expr_matrix = self._expr_matrix[:, gene_indices].T  # (基因×样本)

        for i, gene in enumerate(genes):
            # 获取基因在各样本中的表达
            expr_by_sample = []
            for s in samples:
                expr = expr_matrix[i, sample_masks[s]]
                expr_by_sample.append(expr)

            # 优化相关性计算
            expr_array = np.array(expr_by_sample)
            corr_matrix = np.corrcoef(expr_array)
            icc = np.mean(corr_matrix[np.triu_indices(n_samples, k=1)])

            # 计算空间纯度
            if 'spatial_cluster' in self.adata.obs.columns:
                cluster_labels = self.adata.obs['spatial_cluster'].values
                expr = self._expr_matrix[:, self._gene_index[gene]]

                # 向量化计算聚类平均表达
                unique_clusters = np.unique(cluster_labels)
                cluster_expr = np.array([
                    np.mean(expr[cluster_labels == c])
                    for c in unique_clusters
                ])

                max_expr = cluster_expr.max()
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

    def calculate_expression_concentration(self, genes: List[str], spatial_coords) -> pd.DataFrame:
        """
        计算基因的表达集中性综合评分
        """
        spatial_threshold = 0.15  # 空间分散度阈值（高表达样本分散在15%的空间范围内）
        cum_expr_ratio = 0.50  # 累积表达量占比阈值（找贡献50%表达量的最小样本数）
        results = []

        # 归一化空间坐标（向量化）
        spatial_coords = spatial_coords.copy()
        for dim in range(2):
            dim_vals = spatial_coords[:, dim]
            dim_min, dim_max = dim_vals.min(), dim_vals.max()
            if dim_max - dim_min < 1e-10:
                spatial_coords[:, dim] = 0.0
            else:
                spatial_coords[:, dim] = (dim_vals - dim_min) / (dim_max - dim_min)

        # 批量获取基因表达
        gene_indices = [self._gene_index[gene] for gene in genes]
        expr_matrix = self._expr_matrix[:, gene_indices].T  # (gene×spot)
        expr_matrix = np.log1p(expr_matrix)  # np.log1p(expr_matrix) 或 np.arcsinh(expr_matrix)

        for i, gene in enumerate(genes):
            expr = expr_matrix[i]

            # 计算表达离散指数（VMR）
            mean_expr = np.mean(expr)
            var_expr = np.var(expr)
            if mean_expr < 1e-10:
                disp_index = 0
            else:
                disp_index = var_expr / mean_expr

            # 计算表达占比集中度（自适应样本比例）
            sorted_expr = np.sort(expr)[::-1]
            total_sum = np.sum(sorted_expr)

            if total_sum < 1e-10:
                concentration_ratio = 0.0
                top_n = 0
            else:
                cumulative_sum = np.cumsum(sorted_expr) # 计算累积表达量
                target = total_sum * cum_expr_ratio # 找到贡献cum_expr_ratio总表达量的最小样本数
                top_n = np.argmax(cumulative_sum >= target) + 1 if cumulative_sum[-1] >= target else len(sorted_expr) # 找到第一个累积表达量≥目标值的索引（+1转为样本数）

                # 设置top_n上下限约束（避免极端值）
                cell_count = len(expr)
                top_n = max(20, min(top_n, int(cell_count * 0.2), 500))
                # 计算前top_n样本的表达占比（占比越高，集中性越强）
                top_sum = np.sum(sorted_expr[:top_n])
                concentration_ratio = top_sum / total_sum

            # 空间聚集性约束（计算高表达样本的空间分散度）
            spatial_pass = True
            if top_n >= 10: # 至少10个样本才计算空间分布（避免极端值）
                top_indices = np.argsort(expr)[-top_n:] # 获取高表达样本的索引（前top_n个）
                top_coords = spatial_coords[top_indices] # 提取这些样本的空间坐标
                spatial_std = np.mean(np.std(top_coords, axis=0))# 计算x和y坐标的标准差（综合反映空间分散度）
                if spatial_std > spatial_threshold: # 若空间分散度超过阈值，则去掉
                    spatial_pass = False

            results.append({
                'gene': gene,
                'dispersion_index': disp_index,
                'concentration_ratio': concentration_ratio,
                'spatial_pass': spatial_pass
            })

        # 指标归一化（映射到0-1范围）
        df = pd.DataFrame(results)
        df = df[df['spatial_pass']]

        if not df.empty:
            # 归一化指标
            disp_max, disp_min = df['dispersion_index'].max(), df['dispersion_index'].min()
            df['dispersion_norm'] = (df['dispersion_index'] - disp_min) / (disp_max - disp_min + 1e-10)

            cr_max, cr_min = df['concentration_ratio'].max(), df['concentration_ratio'].min()
            df['concentration_norm'] = (df['concentration_ratio'] - cr_min) / (cr_max - cr_min + 1e-10)

            # 计算综合评分（采用排名平均，避免主观权重）
            df['disp_rank'] = df['dispersion_norm'].rank(ascending=False)
            df['cr_rank'] = df['concentration_norm'].rank(ascending=False)
            df['combined_score'] = 0.7 * df['disp_rank'] + 0.3 * df['cr_rank']

            max_score = df['combined_score'].max() # 转换为0-100分（越高越好）
            df['concentration_score'] = 100 * (1 - df['combined_score'] / max_score)
        else:
            df['concentration_score'] = 0

        return df[['gene', 'concentration_score']]

    def calculate_gss_expression_correlation(self, genes: List[str]) -> pd.DataFrame:
        """
        计算GSS分数与表达量的相关性
        """
        # 批量获取基因表达和GSS分数
        gene_indices = [self._gene_index[gene] for gene in genes]
        expr_matrix = self._expr_matrix[:, gene_indices].T  # (gene×spot)
        expr_matrix = np.log1p(expr_matrix)
        gss_matrix = self.gss_df.loc[genes].values  # (gene×spot)

        # 向量化计算Spearman相关性
        corr_values = []
        p_values = []
        for i in range(len(genes)):
            corr, p = stats.spearmanr(expr_matrix[i], gss_matrix[i])
            corr_values.append(corr)
            p_values.append(p)

        return pd.DataFrame({
            'gene': genes,
            'gss_expr_corr': corr_values,
            'gss_expr_p_value': p_values
        })

    def calculate_morans_i(self, genes: List[str], spatial_weights) -> pd.DataFrame:
        """
        计算基因表达的Moran's I指数（空间自相关性）
        """
        coords = self.spatial_coords.values
        n = len(coords)
        total_weight = spatial_weights.sum()

        # 批量获取基因表达
        gene_indices = [self._gene_index[gene] for gene in genes]
        expr_matrix = self._expr_matrix[:, gene_indices].T  # (gene×spot)
        expr_matrix = np.log1p(expr_matrix)

        # 向量化计算Moran's I
        mean_exprs = np.mean(expr_matrix, axis=1)
        deviations = expr_matrix - mean_exprs[:, np.newaxis]
        denominators = np.sum(deviations ** 2, axis=1)

        # 处理零方差
        valid_mask = denominators > 1e-10
        morans_i = np.full(len(genes), np.nan)

        # 对有效基因计算Moran's I
        for i in np.where(valid_mask)[0]:
            numerator = np.sum(spatial_weights * np.outer(deviations[i], deviations[i]))
            morans_i[i] = (n / total_weight) * (numerator / denominators[i])

        return pd.DataFrame({
            'gene': genes,
            'morans_i': morans_i
        })

    def _calculate_hexagonal_weights(self, spatial_coords, k=8):
        """
        使用K近邻计算动态空间权重w
        """
        n = len(spatial_coords)
        # 使用scipy的高效距离计算
        dist_matrix = distance.cdist(spatial_coords, spatial_coords, metric='euclidean')

        # 排除自身并获取K近邻（使用argpartition优化排序）
        neighbors = np.zeros((n, k), dtype=int)
        for i in range(n):
            dists = dist_matrix[i]
            dists[i] = np.inf  # 排除自身
            # 使用argpartition获取前k个最小值的索引
            neighbors[i] = np.argpartition(dists, k)[:k]

        # 构建权重矩阵
        w = np.zeros((n, n))
        for i in range(n):
            w[i, neighbors[i]] = 1 / k

        return w

    def run_pipeline(self) -> Tuple[List[str], pd.DataFrame]:
        """
        运行完整的基因选择与验证流程
        """
        # 如果细胞数量太少，就直接跳过
        if self.cells <= 700:
            if self.output_dir:
                skip_reason = f"细胞数量为{self.cells},不足700!"
                skip_info = pd.DataFrame({
                    'timestamp': [pd.Timestamp.now()],
                    'cells': [self.cells],
                    'reason': [skip_reason],
                    'status': ['skipped']
                })
                os.makedirs(self.output_dir, exist_ok=True)
                skip_info.to_csv(self.output_dir + "_processing_log.csv")
                print(f"验证结果已保存至 {self.output_dir}")
            return [], pd.DataFrame()

        # 1. 基于表达量和GSS分数筛选基因
        high_expr_genes = self.select_genes_by_expression()
        high_gss_genes = self.select_genes_by_gss()

        # 2. 整合两种指标选择基因
        initial_genes = list(
            set(high_expr_genes) &
            set(high_gss_genes.index)
        )
        print(f"初步筛选出 {len(initial_genes)} 个基因")

        # 定义每个步骤的最大基因数量
        max_genes = min(100, int(0.1 * len(initial_genes))) if initial_genes else 0

        # 预计算空间权重矩阵（基于K近邻）
        spatial_coords = self.spatial_coords.values
        spatial_weights = self._calculate_hexagonal_weights(spatial_coords=spatial_coords)

        # 3.1 计算表达离散度和集中性指标（越高越好）
        concentration_results = self.calculate_expression_concentration(initial_genes, spatial_coords)
        filtered_concentration = concentration_results[
            concentration_results['concentration_score'] > self.concentration_threshold
            ].sort_values('concentration_score', ascending=False)
        high_concentration_genes = filtered_concentration.head(max_genes)['gene'].tolist()
        print(f"基于表达离散度和集中性，保留 {len(high_concentration_genes)} 个基因（最多{max_genes}个）")

        # 3.2 计算GSS与表达量的相关性（越高越好）
        corr_results = self.calculate_gss_expression_correlation(initial_genes)
        filtered_corr = corr_results[
            abs(corr_results['gss_expr_corr']) > self.corr_threshold
            ].sort_values('gss_expr_corr', ascending=False)
        high_corr_genes = filtered_corr.head(max_genes)['gene'].tolist()
        print(f"基于GSS-表达量相关性，保留 {len(high_corr_genes)} 个基因（最多{max_genes}个）")

        # 3.3 计算空间自相关性（越高越好）
        morans_i_results = self.calculate_morans_i(initial_genes, spatial_weights)
        filtered_morans = morans_i_results[
            morans_i_results['morans_i'] > self.morans_i_threshold
            ].sort_values('morans_i', ascending=False)
        high_spatial_genes = filtered_morans.head(max_genes)['gene'].tolist()
        print(f"基于空间自相关性，保留 {len(high_spatial_genes)} 个基因（最多{max_genes}个）")

        # 3.4 计算空间重复性（若有多个样本）
        if 'sample' in self.adata.obs.columns and len(self.adata.obs['sample'].unique()) >= 2:
            reproducibility_results = self.calculate_spatial_reproducibility(initial_genes)
            reliable_genes = reproducibility_results[reproducibility_results['icc'] > self.icc_threshold][
                'gene'].tolist()
            print(f"基于空间重复性，保留 {len(reliable_genes)} 个基因")
        else:
            print("警告：数据中缺少样本批次信息，跳过空间重复性验证")
            reliable_genes = initial_genes

        # 4. 整合所有验证结果
        all_validated_genes = list(
            set(high_concentration_genes) |
            set(high_corr_genes) |
            set(high_spatial_genes)
        )

        # 构建完整的验证结果DataFrame
        all_results = pd.DataFrame({
            'gene': all_validated_genes,
            'pass_concentration': [g in high_concentration_genes for g in all_validated_genes],
            'pass_gss_expr_corr': [g in high_corr_genes for g in all_validated_genes],
            'pass_spatial_auto_corr': [g in high_spatial_genes for g in all_validated_genes],
        })

        # 计算每个基因通过的验证数量
        all_results['count'] = all_results.iloc[:, 1:].sum(axis=1)

        # 提取各项指标
        concentration_dict = {row['gene']: round(row['concentration_score'], 2) for _, row in
                              concentration_results.iterrows()}
        all_results['concentration_score'] = all_results['gene'].map(concentration_dict)

        corr_dict = {row['gene']: round(row['gss_expr_corr'], 2) for _, row in corr_results.iterrows()}
        all_results['gss_expr_corr'] = all_results['gene'].map(corr_dict)

        morans_i_dict = {row['gene']: round(row['morans_i'], 2) for _, row in morans_i_results.iterrows()}
        all_results['morans_i'] = all_results['gene'].map(morans_i_dict)

        # 排序
        all_results = all_results.sort_values(
            ['count', 'concentration_score', 'gss_expr_corr', 'morans_i'],
            ascending=[False, False, False, False]
        )

        # 通过所有验证的基因
        cross_genes = all_results[
            (all_results['pass_concentration']) &
            (all_results['pass_gss_expr_corr']) &
            (all_results['pass_spatial_auto_corr'])
            ]['gene'].tolist()
        print(f"一共有{len(cross_genes)}个基因通过了全部的筛选流程：{cross_genes}")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            all_results.to_csv(self.output_dir+ "_selected_genes.csv", index=False, sep='\t')
            print(f"验证结果已保存至 {self.output_dir}")

        return all_validated_genes, all_results