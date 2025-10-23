import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, KDTree
from multiprocessing import Pool
from functools import partial
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class GssGeneCalculator:
    """
    大规模数据特异性基因分析器
    """

    def __init__(self,
                 adata: sc.AnnData,
                 gene_counts: Dict[str, int],
                 score_threshold: float = 0.5,
                 ):

        print("-----------------------------------")
        self.adata = adata
        self.gene_counts = gene_counts
        self.score_threshold = score_threshold  # 最终的评价阈值
        self.min_cells_ratio = 0.01  # 最少表达spot比例
        self.max_cells_ratio = 0.8 # 最多表达spot比例
        self.n_jobs = 40  # 限制最大进程数

        # 基础数据
        self.coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None
        self.n_cells = adata.n_obs
        self.max_cells = int(self.n_cells * self.max_cells_ratio)
        self.min_cells = max(50, int(self.n_cells * self.min_cells_ratio))


    def prefilter_genes(self) -> List[str]:
        """快速频率筛选"""
        filtered_genes = []
        for gene, count in self.gene_counts.items():
            if gene not in self.adata.var_names:
                continue
            if self.min_cells <= count <= self.max_cells:
                filtered_genes.append(gene)

        print(f"频率筛选: {len(self.gene_counts)} -> {len(filtered_genes)}")
        return filtered_genes

    def _get_expression_vector(self, gene: str) -> np.ndarray:
        """获取表达向量"""
        expr = self.adata[:, gene].X

        if sp.issparse(expr):
            return expr.toarray().flatten()
        return expr.flatten()

    def _calculate_efficient_rarity_metrics(self, expr: np.ndarray) -> Dict[str, float]:
        """
        高效稀有性指标计算
        针对大规模数据优化
        """
        metrics = {}

        # 1. 零表达比例 - 最稳定的指标
        zero_ratio = np.mean(expr == 0)
        metrics['zero_ratio'] = zero_ratio

        # 2. 表达集中度 (基尼系数)
        non_zero = expr[expr > 0]
        if len(non_zero) > 1:
            # 对大规模数据使用近似计算
            if len(non_zero) > 1000:
                # 采样计算基尼系数
                sample_size = min(1000, len(non_zero))
                sampled = np.random.choice(non_zero, sample_size, replace=False)
                sorted_expr = np.sort(sampled)
            else:
                sorted_expr = np.sort(non_zero)

            n = len(sorted_expr)
            index = np.arange(1, n + 1)
            gini = np.sum((2 * index - n - 1) * sorted_expr) / (n * np.sum(sorted_expr))
            metrics['gini'] = gini
        else:
            metrics['gini'] = 0.0

        # 3. 高效离群度检测
        metrics['outlier_score'] = self._calculate_efficient_outlier_score(expr)

        # 4. 表达强度对比度
        metrics['intensity_contrast'] = self._calculate_intensity_contrast(expr)

        return metrics

    def _calculate_efficient_outlier_score(self, expr: np.ndarray, sample_size: int = 1000) -> float:
        """高效的离群度计算"""
        non_zero = expr[expr > 0]
        if len(non_zero) < 10:
            return 0.0

        # 对大规模数据采样
        if len(non_zero) > sample_size:
            non_zero = np.random.choice(non_zero, sample_size, replace=False)

        contamination = min(0.2, max(0.05, 10 / len(non_zero)))

        try:
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                max_samples=min(256, len(non_zero)),
                n_estimators=50  # 减少树的数量
            )
            labels = iso_forest.fit_predict(non_zero.reshape(-1, 1))
            outlier_ratio = np.mean(labels == -1)
            return outlier_ratio
        except:
            return 0.0

    def _calculate_intensity_contrast(self, expr: np.ndarray) -> float:
        """表达强度对比度计算"""
        non_zero = expr[expr > 0]
        if len(non_zero) < 2:
            return 0.0

        # 对大规模数据使用分位数
        if len(non_zero) > 1000:
            q75, q25 = np.percentile(non_zero, [75, 25])
            high_expr = non_zero[non_zero > q75]
            low_expr = non_zero[non_zero <= q25]
        else:
            median_expr = np.median(non_zero)
            high_expr = non_zero[non_zero > median_expr]
            low_expr = non_zero[non_zero <= median_expr]

        if len(high_expr) == 0 or len(low_expr) == 0:
            return 0.0

        mean_high = np.mean(high_expr)
        mean_low = np.mean(low_expr)

        contrast = (mean_high - mean_low) / (mean_high + mean_low + 1e-8)
        return max(0, contrast)

    def _calculate_efficient_spatial_metrics(self, expr: np.ndarray, gene: str) -> Dict[str, float]:
        """高效空间指标计算"""
        metrics = {}

        if self.coords is None:
            return metrics

        try:
            coords_to_use = self.coords
            expr_to_use = expr

            # 快速空间自相关
            metrics['spatial_autocorrelation'] = self._calculate_fast_moran(expr_to_use, coords_to_use)

            # 快速空间聚焦度
            metrics['spatial_focus'] = self._calculate_fast_spatial_focus(expr_to_use, coords_to_use)

        except Exception as e:
            print(f"基因 {gene} 空间指标计算跳过: {e}")

        return metrics

    def _calculate_fast_moran(self, expr: np.ndarray, coords: np.ndarray) -> float:
        """快速Moran's I计算"""
        if np.sum(expr) == 0 or len(expr) < 10:
            return 0.0

        # 使用k-d树进行快速最近邻搜索
        k = min(6, len(coords) - 1)
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=k + 1)

        expr_mean = np.mean(expr)
        expr_centered = expr - expr_mean

        numerator = 0
        denominator = np.sum(expr_centered ** 2)

        if denominator == 0:
            return 0.0

        for i in range(len(expr)):
            neighbors = indices[i, 1:]  # 排除自身
            numerator += expr_centered[i] * np.sum(expr_centered[neighbors])

        W = k * len(expr)
        moran_i = (len(expr) / W) * (numerator / denominator)

        return max(0, moran_i)

    def _calculate_fast_spatial_focus(self, expr: np.ndarray, coords: np.ndarray) -> float:
        """快速空间聚焦度计算"""
        if np.sum(expr) == 0:
            return 0.0

        expr_weights = expr / np.sum(expr)
        center_of_mass = np.average(coords, weights=expr_weights, axis=0)

        distances = np.linalg.norm(coords - center_of_mass, axis=1)
        weighted_avg_distance = np.average(distances, weights=expr_weights)

        # 计算参考距离
        overall_center = np.mean(coords, axis=0)
        overall_distances = np.linalg.norm(coords - overall_center, axis=1)
        reference_distance = np.mean(overall_distances)

        focus_score = 1 - (weighted_avg_distance / reference_distance)
        return max(0, focus_score)

    def _calculate_pattern_uniqueness_batch(self, genes: List[str]) -> Dict[str, float]:
        """
        批量计算模式独特性
        避免重复计算相关性矩阵
        """
        print("批量计算模式独特性...")

        # 构建表达矩阵
        expr_matrix = []
        valid_genes = []

        for gene in genes:
            expr = self._get_expression_vector(gene)
            if np.sum(expr) > 0:
                expr_matrix.append(expr)
                valid_genes.append(gene)

        if len(expr_matrix) < 2:
            return {gene: 0.0 for gene in genes}

        expr_matrix = np.array(expr_matrix)

        # 计算相关性矩阵
        correlation_matrix = np.corrcoef(expr_matrix)
        np.fill_diagonal(correlation_matrix, 0)  # 忽略自相关

        # 计算每个基因的独特性
        uniqueness_scores = {}
        for i, gene in enumerate(valid_genes):
            avg_correlation = np.mean(np.abs(correlation_matrix[i]))
            uniqueness_scores[gene] = 1 - avg_correlation

        # 为所有基因填充默认值
        result = {gene: uniqueness_scores.get(gene, 0.0) for gene in genes}
        return result

    def _assess_gene_specificity_optimized(self, gene: str, uniqueness_scores: Dict[str, float]) -> Dict[str, float]:
        """
        优化的单基因特异性评估
        """
        try:
            if gene not in self.adata.var_names:
                return {}

            expr = self._get_expression_vector(gene)
            if np.sum(expr) == 0:
                return {}

            results = {'gene': gene, 'frequency': self.gene_counts[gene]}

            # 1. 高效稀有性指标
            rarity_metrics = self._calculate_efficient_rarity_metrics(expr)
            results.update(rarity_metrics)

            # 2. 高效空间指标 - 传递基因名
            spatial_metrics = self._calculate_efficient_spatial_metrics(expr, gene)  # 修复这里！
            results.update(spatial_metrics)

            # 3. 模式独特性 (从预计算结果获取)
            results['pattern_uniqueness'] = uniqueness_scores.get(gene, 0.0)

            # 4. 计算综合特异性得分
            specificity_score = self._compute_optimized_specificity(results)
            results['specificity_score'] = specificity_score

            return results

        except Exception as e:
            print(f"基因 {gene} 分析失败: {e}")
            return {}

    def _compute_optimized_specificity(self, metrics: Dict[str, float]) -> float:
        """优化的特异性得分计算"""
        # 根据数据量调整权重
        weights = {
            'rarity': 0.35,  # 表达稀有性
            'spatial': 0.45,  # 空间模式
            'uniqueness': 0.20,  # 模式独特性
        }

        # 稀有性维度
        rarity_indicators = ['zero_ratio', 'gini', 'outlier_score', 'intensity_contrast']
        rarity_score = np.mean([metrics.get(ind, 0) for ind in rarity_indicators])

        # 空间维度
        spatial_indicators = ['spatial_autocorrelation', 'spatial_focus']
        spatial_score = np.mean([metrics.get(ind, 0) for ind in spatial_indicators])

        # 独特性维度
        uniqueness_score = metrics.get('pattern_uniqueness', 0)

        # 加权综合得分
        comprehensive_score = (
                weights['rarity'] * rarity_score +
                weights['spatial'] * spatial_score +
                weights['uniqueness'] * uniqueness_score
        )

        return min(1.0, comprehensive_score)

    def parallel_gene_analysis(self, candidate_genes: List[str]) -> List[Dict]:
        """并行基因分析 - 大规模数据优化版"""
        print(f"开始并行分析 {len(candidate_genes)} 个基因")

        # 预计算模式独特性（批量）
        uniqueness_scores = self._calculate_pattern_uniqueness_batch(candidate_genes)

        # 使用优化的并行策略
        chunksize = max(1, len(candidate_genes) // (self.n_jobs * 2))

        with Pool(self.n_jobs) as pool:
            # 部分函数应用
            analyze_func = partial(
                self._assess_gene_specificity_optimized,
                uniqueness_scores=uniqueness_scores
            )

            # 使用imap提高内存效率
            results = []
            for i, result in enumerate(pool.imap(analyze_func, candidate_genes, chunksize=chunksize)):
                if result:
                    results.append(result)
                if (i + 1) % 50 == 0:
                    print(f"已处理 {i + 1}/{len(candidate_genes)} 个基因")

        print(f"并行分析完成: {len(results)} 个有效结果")
        return results

    def _classify_specificity_types(self, results: pd.DataFrame) -> pd.DataFrame:
        """分类特异性类型"""
        df = results.copy()

        # 定义类型条件
        conditions = [
            (df['spatial_autocorrelation'] > 0.3) & (df['spatial_focus'] > 0.3),
            (df['zero_ratio'] > 0.7) & (df['gini'] > 0.5),
            (df['pattern_uniqueness'] > 0.4),
            (df['outlier_score'] > 0.2) & (df['intensity_contrast'] > 0.3)
        ]

        choices = ['spatial', 'rare', 'unique', 'contrast']

        df['specificity_type'] = np.select(conditions, choices, default='mixed')

        return df

    def run_pipeline(self):
        """
        分析主函数
        """
        print("开始数据特异性分析")

        # 如果细胞数量太少，就直接跳过
        if self.n_cells <= 700:
            print(f"细胞数量为{self.n_cells},不足700!")
            return [], pd.DataFrame()

        # 第一步：快速频率筛选
        candidate_genes = self.prefilter_genes()

        if not candidate_genes:
            print("没有符合条件的候选基因")
            return pd.DataFrame(), pd.DataFrame()

        # 第二步：并行分析所有基因
        specificity_results = self.parallel_gene_analysis(candidate_genes)

        if not specificity_results:
            print("没有获得有效的分析结果")
            return pd.DataFrame(), pd.DataFrame()

        # 第三步：转换为DataFrame并筛选
        all_results = pd.DataFrame(specificity_results)
        all_results = all_results.sort_values('specificity_score', ascending=False)

        # 筛选高特异性基因
        high_specificity = all_results[all_results['specificity_score'] >= self.score_threshold].copy()

        # 识别特异性类型
        high_specificity = self._classify_specificity_types(high_specificity)

        #限制每类特异性基因的数量（最多前max_per_type个）
        max_per_type = 60  # 每类最大保留数量
        # 按类型分组，每组按specificity_score降序，取前max_per_type个
        high_specificity_limited = high_specificity.groupby('specificity_type', group_keys=False).apply(
            lambda x: x.sort_values('specificity_score', ascending=False).head(max_per_type)
        )

        print(f"分析完成: 限制后共{len(high_specificity_limited)}个高特异性基因（每类最多{max_per_type}个）")
        if len(high_specificity_limited) > 0:
            type_counts = high_specificity_limited['specificity_type'].value_counts()
            print(f"特异性类型分布: {type_counts.to_dict()}")

        high_specificity_genes = high_specificity_limited['gene'].tolist()

        all_results = all_results.copy()
        # 添加筛选状态列
        all_results['selected'] = all_results['gene'].isin(high_specificity_genes)
        # 添加基因类型列
        gene_type_mapping = high_specificity_limited.set_index('gene')['specificity_type'].to_dict()
        all_results['gene_type'] = all_results['gene'].map(gene_type_mapping).fillna('non-selected')

        return high_specificity_genes, all_results


