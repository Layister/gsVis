import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Callable
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import warnings

warnings.filterwarnings('ignore')


class GssGeneCalculator:
    """
    基因选择器

    参数:
        adata: AnnData对象，包含基因表达矩阵和空间坐标
        gss_df: GSS分数DataFrame，行名为基因名，列名为样本名
    """

    def __init__(self,
                 adata: sc.AnnData,
                 gene_counts: Dict[str, int],
                 ):

        print("-----------------------------------")
        self.adata = adata
        self.gene_counts = gene_counts
        self.min_cells_ratio = 0.01  # 最少表达spot比例
        self.max_cells_ratio = 0.8  # 最多表达spot比例
        self.max_genes_per_type = 60 # 每个类别保留的最多基因
        self.n_jobs = 40  # 限制最大进程数

        # 基础数据
        self.coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None
        self.n_cells = adata.n_obs
        self.max_cells = int(self.n_cells * self.max_cells_ratio)
        self.min_cells = max(50, int(self.n_cells * self.min_cells_ratio))

        # 预计算空间结构
        self.k = min(6, len(self.coords) - 1)
        tree = KDTree(self.coords)
        _, self.spatial_indices = tree.query(self.coords, k=self.k + 1)

        # 定义指标计算函数和基因类型判定规则
        self.metric_functions = self._define_base_metrics()
        self.gene_type_rules = self._define_gene_types()

    def _define_base_metrics(self) -> Dict[str, Callable]:
        """定义基础指标计算函数"""
        return {
            # 表达分布指标
            'expression_sparsity': self._calculate_expression_sparsity,
            'expression_concentration': self._calculate_expression_concentration,
            'expression_variability': self._calculate_expression_variability,

            # 空间模式指标
            'spatial_autocorrelation': self._calculate_enhanced_moran,
            'spatial_focus': self._calculate_spatial_focus,
            'spatial_clustering': self._calculate_spatial_clustering,
            'spatial_directionality': self._calculate_spatial_directionality,

            # 统计特征指标
            'intensity_contrast': self._calculate_intensity_contrast,
        }

    def _define_gene_types(self) -> Dict[str, Dict]:
        """基于多维度证据的基因类型定义"""
        return {
            'structural_marker': {
                'description': '组织结构标志物 - 标记组织区域和边界',
                'conditions': {
                    'expression_concentration': ('>', 0.5),  # 表达集中
                    'spatial_autocorrelation': ('>', 0.3),  # 空间自相关
                    'expression_variability': ('>', 0.2),  # 表达变异
                },
                'weight': {
                    'expression_concentration': 0.3,
                    'spatial_autocorrelation': 0.5,
                    'expression_variability': 0.2
                }
            },

            'rare_cell_marker': {
                'description': '稀有细胞标志物 - 标记稀有细胞亚群',
                'conditions': {
                    'expression_sparsity': ('>', 0.8),  # 高度稀有
                    'expression_concentration': ('>', 0.6),  # 表达集中
                    'spatial_clustering': ('>', 0.3),  # 空间聚集
                },
                'weight': {
                    'expression_sparsity': 0.4,
                    'expression_concentration': 0.3,
                    'spatial_clustering': 0.3
                }
            },

            'functional_hotspot': {
                'description': '功能热点基因 - 标记功能活跃区域',
                'conditions': {
                    'spatial_focus': ('>', 0.3),  # 空间聚焦
                    'intensity_contrast': ('>', 0.4),  # 强度对比
                    'expression_variability': ('>', 0.4),  # 表达变异
                },
                'weight': {
                    'spatial_focus': 0.4,
                    'intensity_contrast': 0.4,
                    'expression_variability': 0.2,
                }
            },

            'spatial_trend': {
                'description': '空间趋势基因 - 系统性空间变化模式',
                'conditions': {
                    'spatial_directionality': ('>', 0.4),  # 空间方向性
                    'expression_variability': ('>', 0.5),  # 表达变异
                    'expression_sparsity': ('<', 0.9), # 不可过于稀疏
                },
                'weight': {
                    'spatial_directionality': 0.5,
                    'expression_variability': 0.4,
                    'expression_sparsity': 0.1,
                }
            }
        }

    # ==================== 指标计算函数 ====================

    def _calculate_expression_sparsity(self, expr: np.ndarray) -> float:
        """表达稀疏性 - 零比例"""
        return np.mean(expr == 0)

    def _calculate_expression_concentration(self, expr: np.ndarray) -> float:
        """表达集中性评分"""
        non_zero = expr[expr > 0]
        if len(non_zero) == 0:
            return 0.0

        sorted_expr = np.sort(non_zero)[::-1]
        cumulative_sum = np.cumsum(sorted_expr)
        total_sum = cumulative_sum[-1]

        # 找到贡献50%表达量的最少细胞数
        target = total_sum * 0.5
        top_n = np.argmax(cumulative_sum >= target) + 1

        # 计算这些细胞的表达占比
        top_ratio = top_n / len(non_zero)
        concentration = 1 - top_ratio  # 越少细胞贡献越多表达，集中性越高

        return concentration

    def _calculate_expression_variability(self, expr: np.ndarray) -> float:
        """变异系数"""
        non_zero_count = np.sum(expr > 0)
        total_count = len(expr)
        non_zero_ratio = non_zero_count / total_count

        if non_zero_ratio < 0.05: # 非零表达值过少了，填充零值到全部spot的5%
            num_to_include = int(0.05 * total_count)
            sorted_indices = np.argsort(-expr)
            selected_expr = expr[sorted_indices[:num_to_include]]
        else: # 非零表达值没有过少，正常计算非零表达值的变异系数
            selected_expr = expr[expr > 0]

        if len(selected_expr) < 10:
            return 0.0

        mean = np.mean(selected_expr)
        std = np.std(selected_expr)

        if mean < 1e-8:
            return 0.0

        cv = std / mean
        max_cv = 2.0 # 标准化：超过max_cv的视为1，0-max_cv之间线性缩放到0-1
        return min(1.0, max(0.0, cv / max_cv))

    def _calculate_enhanced_moran(self, expr: np.ndarray) -> float:
        """增强版Moran's I空间自相关"""
        if np.sum(expr) == 0 or len(expr) < 10:
            return 0.0

        expr_mean = np.mean(expr)
        expr_centered = expr - expr_mean
        denominator = np.sum(expr_centered ** 2)

        if denominator == 0:
            return 0.0

        # 向量化计算
        neighbors = self.spatial_indices[:, 1:]
        neighbor_expr = expr_centered[neighbors]

        numerator = np.sum(expr_centered[:, np.newaxis] * neighbor_expr)
        W = self.k * len(expr)

        moran_i = max(0, (len(expr) / W) * (numerator / denominator))
        return moran_i

    def _calculate_spatial_focus(self, expr: np.ndarray) -> float:
        """空间聚焦度"""
        if np.sum(expr) == 0:
            return 0.0

        # 过滤低表达细胞
        filtered_expr = expr.copy()
        filtered_expr[filtered_expr <= 1e-8] = 0  # 低表达细胞权重归零

        # 若过滤后无有效表达，返回0
        filtered_total = np.sum(filtered_expr)
        if filtered_total == 0:
            return 0.0

        # 用过滤后的表达值计算权重
        expr_weights = filtered_expr / filtered_total
        center_of_mass = np.average(self.coords, weights=expr_weights, axis=0)

        # 距离计算
        distances = np.linalg.norm(self.coords - center_of_mass, axis=1)
        weighted_avg_distance = np.average(distances, weights=expr_weights)

        overall_center = np.mean(self.coords, axis=0)
        overall_distances = np.linalg.norm(self.coords - overall_center, axis=1)
        reference_distance = np.mean(overall_distances)

        if reference_distance < 1e-8:
            return 0.0

        focus_score = 1 - (weighted_avg_distance / reference_distance)
        return max(0, focus_score)

    def _calculate_spatial_clustering(self, expr: np.ndarray) -> float:
        """空间聚类 - 高表达细胞的空间聚集程度"""
        if np.sum(expr) == 0:
            return 0.0

        # 识别高表达细胞（前20%）
        non_zero = expr[expr > 0]
        if len(non_zero) == 0:
            return 0.0

        high_expr_threshold = np.percentile(non_zero, 80)
        high_expr_mask = expr > high_expr_threshold

        if np.sum(high_expr_mask) < 2:
            return 0.0

        # 计算高表达细胞间的平均距离
        high_expr_coords = self.coords[high_expr_mask]
        high_distances = pdist(high_expr_coords)

        # 计算随机期望距离
        n_high = len(high_expr_coords)
        random_indices = np.random.choice(len(self.coords), n_high, replace=False)
        random_coords = self.coords[random_indices]
        random_distances = pdist(random_coords)

        if len(random_distances) == 0 or np.mean(random_distances) < 1e-8:
            return 0.0

        clustering_score = 1 - (np.mean(high_distances) / np.mean(random_distances))
        return max(0, clustering_score)

    def _calculate_intensity_contrast(self, expr: np.ndarray) -> float:
        """表达强度对比度"""
        non_zero = expr[expr > 0]
        if len(non_zero) < 2:
            return 0.0

        # 使用百分位数分割高低表达区域
        q3 = np.percentile(non_zero, 75)
        high_expr = non_zero[non_zero > q3]
        low_expr = non_zero[non_zero <= q3]
        if len(high_expr) == 0 or len(low_expr) == 0:
            return 0.0

        # 取高区域的高分位数，低区域的低分位数（增大差异）
        median_high = np.percentile(high_expr, 60) # 取高表达区域的60%分位数
        median_low = np.percentile(low_expr, 40) # 取低表达区域的40%分位数

        if median_low < 1e-8:
            return 1.0

        contrast = (median_high - median_low) / (median_high + median_low)

        return max(0, contrast)

    # def _calculate_spatial_directionality(self, expr: np.ndarray) -> float:
    #     """空间方向性 - 检测表达的系统性空间变化"""
    #     if np.sum(expr) == 0:
    #         return 0.0
    #
    #     try:
    #         # 方法1: 加权PCA - 用表达量加权空间坐标
    #         weights = np.log1p(expr) / (np.sum(np.log1p(expr)) + 1e-8)
    #         weighted_coords = self.coords * weights[:, np.newaxis]
    #
    #         # PCA分析
    #         pca = PCA(n_components=2)
    #         pca.fit(weighted_coords)
    #
    #         # 第一主成分的方差解释率
    #         pca_directionality = pca.explained_variance_ratio_[0]
    #
    #         # 方法2: 多方向线性趋势
    #         trends = []
    #         # 检查多个方向
    #         for angle in [0, 45, 90, 135]:  # 0°, 45°, 90°, 135°
    #             # 旋转坐标系
    #             rad = np.radians(angle)
    #             rotation_matrix = np.array([
    #                 [np.cos(rad), -np.sin(rad)],
    #                 [np.sin(rad), np.cos(rad)]
    #             ])
    #             rotated_coords = self.coords @ rotation_matrix
    #
    #             # 在新方向上检查趋势
    #             x_rotated = rotated_coords[:, 0]
    #             _, _, r_value, _, _ = stats.linregress(x_rotated, expr)
    #             trends.append(abs(r_value) if not np.isnan(r_value) else 0)
    #
    #         max_trend = max(trends) if trends else 0
    #
    #         # 综合两种方法
    #         combined_directionality = (pca_directionality + max_trend) / 2
    #         return combined_directionality
    #
    #     except Exception as e:
    #         print(f"空间方向性计算错误: {e}")
    #         return 0.0

    def _calculate_spatial_directionality(self, expr: np.ndarray) -> float:
        """优化综合评分：分层加权+冲突校验+趋势方向验证"""
        if np.sum(expr) == 0:
            return 0.0

        # 计算三个基础分数
        weights = self._simple_balanced_weights(expr)
        if np.sum(weights) < 1e-8:
            return 0.0

        # 1. PCA评分
        weighted_coords = self.coords * weights[:, np.newaxis]
        pca = PCA(n_components=2)
        pca.fit(weighted_coords)
        pca_score = pca.explained_variance_ratio_[0]
        global_direction = pca.components_[0]

        # 2. 全局一致性评分
        consistency_score = self._simple_global_consistency(expr, global_direction, weights)

        # 3. 鲁棒趋势评分
        segment_score = self._robust_monotonic_trend(expr)

        # 验证趋势方向与PCA方向的一致性
        trend_pca_alignment = self._validate_trend_pca_alignment(expr, global_direction)

        # 如果趋势方向与PCA方向不一致，适当降低趋势评分
        if trend_pca_alignment < 0.5:
            segment_score *= 0.7  # 不一致时惩罚趋势评分

        # 综合评分优化
        scores = [pca_score, consistency_score, segment_score]
        mean_scores = np.mean(scores)
        std_scores = np.std(scores)

        # 步骤1：用一致性评分作为守门人（低一致性直接压制得分）
        if consistency_score < 0.3:
            # 一致性低时，用一致性限制整体得分
            return min(consistency_score * mean_scores, consistency_score)  # 双重保险

        # 步骤2：计算指标一致性（变异系数），动态调整权重
        if mean_scores < 1e-8:  # 避免除零
            return 0.0
        cv = std_scores / mean_scores  # 变异系数（值越小，指标越一致）
        weight_factor = max(0.5, 1 - cv)  # 一致性越高，权重系数越接近1

        # 步骤3：核心指标赋予更高基础权重
        base_score = pca_score * 0.4 + consistency_score * 0.4 + segment_score * 0.2

        # 最终评分 = 基础得分 × 一致性权重
        final_score = base_score * weight_factor
        return min(1.0, max(0.0, final_score))  # 确保在[0,1]范围内

    def _simple_balanced_weights(self, expr: np.ndarray) -> np.ndarray:
        """简化的平衡权重 - 增强数值稳定性"""
        # 分位数截断
        expr_nonzero = expr[expr > 0]
        if len(expr_nonzero) == 0:
            return np.zeros_like(expr)

        # 使用更鲁棒的分位数计算
        if len(expr_nonzero) > 10:
            upper_quantile = np.percentile(expr_nonzero, 99)  # 截断最高1%
        else:
            upper_quantile = np.max(expr_nonzero)  # 数据少时不用截断

        truncated_expr = np.minimum(expr, upper_quantile)

        # 添加小的epsilon避免log(0)问题
        weights = np.log1p(truncated_expr + 1e-12)
        total = np.sum(weights)

        # 归一化前检查有效性
        if total < 1e-12 or np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return np.zeros_like(expr)

        return weights / total

    def _robust_monotonic_trend(self, expr: np.ndarray) -> float:
        """鲁棒单调趋势检测"""
        trends = []

        for dim in [0, 1]:
            sort_idx = np.argsort(self.coords[:, dim])
            sorted_expr = expr[sort_idx]
            n_points = len(sorted_expr)

            # 数据点太少时无法检测趋势
            if n_points < 10:
                trends.append(0.0)
                continue

            # 自适应参数设置
            # 最大距离：覆盖约10-15%的细胞范围
            max_distance = min(n_points, max(8, n_points // 7))

            # 采样数量：确保足够的统计功效
            if n_points < 1000:
                n_samples = min(1500, n_points * 3)
            else:
                n_samples = min(2000, n_points * 2)

            n_increasing = 0
            n_decreasing = 0
            total_valid_pairs = 0

            # 采样点对进行趋势判断
            for _ in range(n_samples):
                # 确保有意义的距离（避免相邻点）
                if n_points > 10:
                    i = np.random.randint(0, n_points - 5)
                    j = np.random.randint(i + 3, min(i + max_distance, n_points))
                else:
                    # 数据点很少时的备用策略
                    i, j = np.random.choice(n_points, 2, replace=False)
                    if i >= j:
                        i, j = j, i
                    if j - i < 2:  # 避免相邻点
                        continue

                # 比较表达量变化方向
                if sorted_expr[j] > sorted_expr[i] + 1e-6:
                    n_increasing += 1
                elif sorted_expr[j] < sorted_expr[i] - 1e-6:
                    n_decreasing += 1
                total_valid_pairs += 1

            # 计算趋势强度和一致性
            if total_valid_pairs == 0:
                trends.append(0.0)
                continue

            inc_ratio = n_increasing / total_valid_pairs
            dec_ratio = n_decreasing / total_valid_pairs
            trend_strength = max(inc_ratio, dec_ratio)

            # 一致性计算：差异越大，一致性越好
            if max(inc_ratio, dec_ratio) > 0:
                consistency = 1.0 - min(inc_ratio, dec_ratio) / max(inc_ratio, dec_ratio)
            else:
                consistency = 0.0

            # 最终得分 = 趋势强度 × 方向一致性
            final_score = trend_strength * consistency
            trends.append(min(1.0, final_score))  # 确保不超过1

        # 返回两个维度的平均趋势得分
        return np.mean(trends) if trends else 0.0

    def _validate_trend_pca_alignment(self, expr: np.ndarray, pca_direction: np.ndarray) -> float:
        """验证趋势与PCA方向一致性"""
        alignment_scores = []

        for dim in [0, 1]:
            sort_idx = np.argsort(self.coords[:, dim])
            sorted_expr = expr[sort_idx]

            if len(sorted_expr) < 15:
                continue

            # 使用3个等分段落进行验证
            n_segments = 3
            segment_size = len(sorted_expr) // n_segments

            segment_medians = []

            for seg in range(n_segments):
                start = seg * segment_size
                end = (seg + 1) * segment_size if seg < n_segments - 1 else len(sorted_expr)
                segment_data = sorted_expr[start:end]

                if len(segment_data) < 5:
                    continue

                # 使用分位数平均值，对异常值适度鲁棒
                q1 = np.percentile(segment_data, 30)  # 30%分位数
                q3 = np.percentile(segment_data, 70)  # 70%分位数
                seg_median = (q1 + q3) / 2  # 分位数平均值

                segment_medians.append(seg_median)

            if len(segment_medians) < 2:
                continue

            # 简单趋势方向检测（内联实现）
            if segment_medians[-1] > segment_medians[0] + 1e-6:
                overall_trend = 1  # 上升
            elif segment_medians[-1] < segment_medians[0] - 1e-6:
                overall_trend = -1  # 下降
            else:
                overall_trend = 0  # 无趋势

            if overall_trend == 0:
                continue

            # 验证与PCA方向的一致性
            if abs(pca_direction[dim]) > 0.1:
                expected_direction = 1 if pca_direction[dim] > 0 else -1
                alignment = 1.0 if overall_trend == expected_direction else 0.0

                # 添加趋势强度权重
                if max(segment_medians) > 0:
                    trend_strength = abs(segment_medians[-1] - segment_medians[0]) / max(segment_medians)
                else:
                    trend_strength = 0.0

                alignment *= min(1.0, trend_strength * 2)  # 适当缩放

                alignment_scores.append(alignment)

        return np.mean(alignment_scores) if alignment_scores else 0.5

    def _simple_global_consistency(self, expr: np.ndarray, global_direction: np.ndarray, weights: np.ndarray) -> float:
        """全局一致性验证：平衡高低权重区域，避免低估低表达区域"""
        n_samples = min(200, len(expr) // 5)
        if n_samples < 10:  # 样本量过小时返回0，保证统计意义
            return 0.0

        # 复用预计算的平衡权重，检查有效性
        if np.sum(weights) < 1e-8:
            return 0.0

        # 1. 混合采样概率：平衡高低权重区域
        # 权重概率：高权重点优先
        weight_probs = weights / np.sum(weights)
        # 均匀概率：低权重点保底
        uniform_probs = np.ones_like(weights) / len(weights)
        # 混合概率: 权重分布均匀时多用均匀采样
        mix_ratio = 0.6 if np.std(weights) > 0.1 else 0.4  # 权重分布均匀时多用均匀采样
        sample_probs = mix_ratio * weight_probs + (1 - mix_ratio) * uniform_probs
        sample_probs = sample_probs / np.sum(sample_probs)  # 重新归一化

        # 2. 初始采样
        sample_idx = np.random.choice(len(expr), n_samples, replace=False, p=sample_probs)

        # 3. 检查低权重样本占比，不足则补充（保底机制）
        # 定义低权重阈值（如低于权重均值的样本视为低权重）
        weight_mean = np.mean(weights)
        is_low_weight = weights[sample_idx] < weight_mean
        low_weight_ratio = np.mean(is_low_weight)

        # 若低权重样本占比<20%，补充5个低权重样本（不超过总采样量）
        if low_weight_ratio < 0.2 and n_samples > 10:
            # 筛选低权重样本索引
            low_weight_candidates = np.where(weights < weight_mean)[0]
            if len(low_weight_candidates) >= 5:
                # 从低权重样本中额外采样5个（不与已有重复）
                new_idx = np.random.choice(
                    [idx for idx in low_weight_candidates if idx not in sample_idx],
                    size=5,
                    replace=False
                )
                sample_idx = np.concatenate([sample_idx, new_idx])
                # 截断到最大采样量（避免过多）
                sample_idx = sample_idx[:min(220, n_samples + 5)]

        # 4. 计算局部梯度与全局方向的一致性
        alignment_scores = []
        sample_weights = []

        for idx in sample_idx:
            neighbors = self.spatial_indices[idx, 1:7]  # 6个最近邻居
            if len(neighbors) < 3:
                continue

            # 计算坐标差和表达差
            coord_diff = self.coords[neighbors] - self.coords[idx]
            expr_diff = expr[neighbors] - expr[idx]

            # 求解局部梯度
            try:
                gradient = np.linalg.lstsq(coord_diff, expr_diff, rcond=1e-6)[0]
                grad_norm = np.linalg.norm(gradient)
                if grad_norm < 1e-8:
                    continue  # 过滤无梯度区域

                # 计算方向一致性
                alignment = abs(np.dot(gradient / grad_norm, global_direction))
                alignment = min(1.0, alignment)  # 限制最大值

                # 软化权重：设置权重下限（最大权重的10%），避免低权重被过度压制
                max_weight = np.max(weights)
                softened_weight = max(weights[idx], 0.1 * max_weight)  # 最低为最大权重的10%

                alignment_scores.append(alignment)
                sample_weights.append(softened_weight)

            except np.linalg.LinAlgError:
                continue  # 忽略奇异矩阵

        # 5. 加权平均（平衡高低权重贡献）
        if not alignment_scores:
            return 0.0
        total_weight = np.sum(sample_weights)
        if total_weight < 1e-8:
            return 0.0

        weighted_avg = np.sum(np.array(alignment_scores) * np.array(sample_weights)) / total_weight
        return weighted_avg

    # ==================== 核心流程函数 ====================

    def prefilter_genes(self) -> List[str]:
        """快速频率筛选"""
        print("进行基因频率预筛选...")
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
        try:
            gene_idx = self.adata.var_names.get_loc(gene)
            expr = self.adata.X[:, gene_idx]

            if sp.issparse(expr):
                return expr.toarray().flatten()
            return expr.flatten()
        except:
            return np.zeros(self.n_cells)

    def _calculate_gene_metrics(self, gene: str) -> Dict[str, float]:
        """计算单个基因的所有指标"""
        try:
            expr = self._get_expression_vector(gene)
            if np.sum(expr) == 0:
                return {}

            metrics = {'gene': gene, 'frequency': self.gene_counts[gene]}

            # 计算所有指标
            for metric_name, metric_func in self.metric_functions.items():
                try:
                    value = metric_func(expr)
                    metrics[metric_name] = value
                except Exception as e:
                    metrics[metric_name] = 0.0

            return metrics

        except Exception as e:
            return {}

    def calculate_all_metrics(self, candidate_genes: List[str]) -> pd.DataFrame:
        """计算所有基因的指标"""
        print(f"开始计算 {len(candidate_genes)} 个基因的 {len(self.metric_functions)} 个指标...")

        def process_gene(gene):
            return self._calculate_gene_metrics(gene)

        # 使用joblib并行处理
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(process_gene)(gene) for gene in candidate_genes
        )

        # 过滤空结果
        results = [res for res in results if res]
        return pd.DataFrame(results)

    def assign_gene_types(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        分配基因类型：判断是否满足类型条件，并计算类型专属分数
        """
        typed_df = metrics_df.copy()

        for gene_type, rule in self.gene_type_rules.items():
            type_col = f'is_{gene_type}'  # 标记是否属于该类型
            score_col = f'score_{gene_type}'  # 该类型的分数

            # 初始化列
            typed_df[type_col] = False
            typed_df[score_col] = 0.0

            # 检查是否满足该类型的所有条件
            all_conditions = pd.Series(True, index=metrics_df.index)
            for metric, (op, threshold) in rule['conditions'].items():
                if metric not in metrics_df.columns:
                    continue
                # 按运算符判断条件
                if op == '>':
                    all_conditions &= (metrics_df[metric] > threshold)
                elif op == '<':
                    all_conditions &= (metrics_df[metric] < threshold)

            # 标记符合条件的基因
            typed_df[type_col] = all_conditions

            # 计算该类型的分数（使用weight中定义的指标）
            if all_conditions.any():
                # 使用weight的键作为评分指标
                scoring_metrics = rule['weight'].keys()
                valid_metrics = [m for m in scoring_metrics if m in metrics_df.columns]
                if valid_metrics:
                    # 计算加权分数
                    weights = [rule['weight'][m] for m in valid_metrics]
                    # 标准化权重
                    weights = np.array(weights) / np.sum(weights)
                    # 计算加权平均分数
                    type_scores = np.average(metrics_df[valid_metrics], weights=weights, axis=1)
                    typed_df.loc[all_conditions, score_col] = type_scores[all_conditions]

        return typed_df

    def select_genes_by_type(self, typed_df: pd.DataFrame) -> pd.DataFrame:
        """按类型筛选基因（直接基于is_xxx标记和类型分数）"""
        selected_dfs = []

        for gene_type in self.gene_type_rules.keys():
            type_col = f'is_{gene_type}'
            score_col = f'score_{gene_type}'

            # 筛选符合该类型条件的基因
            type_mask = typed_df[type_col]
            type_genes = typed_df[type_mask].copy()

            if len(type_genes) == 0:
                print(f"  {gene_type}: 没有符合条件的基因")
                continue

            # 按该类型的分数降序取前N个
            top_genes = type_genes.nlargest(self.max_genes_per_type, score_col)
            selected_dfs.append(top_genes)
            description = self.gene_type_rules[gene_type]['description']
            print(f"  {description}: 筛选出 {len(top_genes)} 个候选基因")

        if not selected_dfs:
            return pd.DataFrame()

        # 合并去重（保留第一个出现的记录）
        union_df = pd.concat(selected_dfs, ignore_index=True).drop_duplicates(subset='gene', keep='first')
        return union_df

    def run_pipeline(self) -> Tuple[List[str], pd.DataFrame]:
        """
        空间基因分类主流程
        """
        if self.n_cells <= 700:
            print(f"细胞数量为{self.n_cells},不足700!")
            return [], pd.DataFrame()

        # 基础频率筛选
        candidate_genes = self.prefilter_genes()
        print(f"基础筛选后候选基因: {len(candidate_genes)}个")

        if not candidate_genes:
            return [], pd.DataFrame()

        # 计算所有指标
        metrics_df = self.calculate_all_metrics(candidate_genes)

        if metrics_df.empty:
            print("没有获得有效的分析结果")
            return [], pd.DataFrame()

        # 类型分配
        typed_df = self.assign_gene_types(metrics_df)

        # 按类型筛选
        selected_df = self.select_genes_by_type(typed_df)
        if len(selected_df) == 0:
            print("没有选中任何基因")
            return [], typed_df

        # 结果处理
        selected_genes = selected_df['gene'].tolist()
        print(f"最终筛选出 {len(selected_genes)} 个特异性基因")

        typed_df['selected'] = typed_df['gene'].isin(selected_genes)
        return selected_genes, typed_df