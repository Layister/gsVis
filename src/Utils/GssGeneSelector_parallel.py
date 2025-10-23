import os
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import stats
from typing import List, Tuple, Dict, Callable
from sklearn.neighbors import KDTree
import warnings
import scipy
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')


class GssGeneSelector:
    def __init__(self,
                 adata: sc.AnnData,
                 gss_df: pd.DataFrame,
                 output_dir: str
                 ):
        """
        基因选择器

        参数:
            adata: AnnData对象，包含基因表达矩阵和空间坐标
            gss_df: GSS分数DataFrame，行名为基因名，列名为样本名
            output_dir: 输出文件地址
        """
        self.adata = adata
        self.gss_df = gss_df
        self.output_dir = output_dir
        self.n_jobs = 40

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

        # 预处理表达矩阵为密集矩阵（如果是稀疏矩阵），并转置
        if scipy.sparse.issparse(self.adata.X):
            self.expr_matrix = self.adata.X.toarray().T  # 基因×细胞
        else:
            self.expr_matrix = self.adata.X.T
        self.gene_names = self.adata.var_names.tolist()
        self.gene_to_index = {gene: i for i, gene in enumerate(self.gene_names)}

        # 预计算max_intensity
        self.max_intensity = self._compute_max_intensity()

        # 预计算空间结构
        self.k = min(6, len(self.spatial_coords) - 1)
        tree = KDTree(self.spatial_coords)
        _, self.indices = tree.query(self.spatial_coords, k=self.k + 1)  # 查询每个细胞的k+1个近邻（第0个是自身）

        # 定义指标计算函数和基因类型判定规则
        self.metric_functions = self._define_base_metrics()
        self.gene_type_rules = self._define_gene_types()

    def _define_base_metrics(self) -> Dict[str, Callable]:
        """定义基础指标计算函数"""
        return {
            # 空间模式指标
            'spatial_autocorrelation': self._calculate_morans_i,
            'pattern_robustness': self._calculate_pattern_robustness,

            # 表达分布指标
            'expression_sparsity': self._calculate_expression_sparsity,
            'expression_concentration': self._calculate_concentration_score,
            'expression_variability': self._calculate_expression_variability,
            'expression_intensity': self._calculate_expression_intensity,

            # 统计特征指标
            'expression_separation': self._calculate_expression_separation,
            'expression_bimodality': self._calculate_expression_bimodality,

            # GSS相关指标
            'gss_correlation': self._calculate_gss_expr_correlation,
        }

    def _define_gene_types(self) -> Dict[str, Dict]:
        """基于多维度证据的基因类型定义"""
        return {
            'regional_marker_gene': {
                'description': '区域标志基因',
                'conditions': {
                    'expression_concentration': ('>', 0.6),  # 表达集中
                    'spatial_autocorrelation': ('>', 0.4),  # 强空间自相关
                    'expression_variability': ('>', 0.2),  # 适度变异
                },
                'weight': {
                    'expression_concentration': 0.4,
                    'spatial_autocorrelation': 0.4,
                    'expression_variability': 0.2
                }
            },

            'focused_expression_gene': {
                'description': '聚焦表达基因',
                'conditions': {
                    'expression_concentration': ('>', 0.70),  # 表达集中性
                    'expression_sparsity': ('>', 0.6),  # 适度稀疏
                    'pattern_robustness': ('>', 0.6),  # 高稳健性
                },
                'weight': {
                    'expression_concentration': 0.4,
                    'expression_sparsity': 0.3,
                    'pattern_robustness': 0.3
                }
            },

            'bimodal_expression_gene': {
                'description': '双峰表达基因',
                'conditions': {
                    'expression_intensity': ('>', 0.2),  # 表达强度
                    'expression_separation': ('>', 0.4),  # 表达分离
                    'expression_bimodality': ('>', 0.6),  # 双峰性
                },
                'weight': {
                    'expression_intensity': 0.2,
                    'expression_separation': 0.4,
                    'expression_bimodality': 0.4
                }
            },

            'validated_spatial_gene': {
                'description': '验证空间基因',
                'conditions': {
                    'spatial_autocorrelation': ('>', 0.4),  # 空间自相关
                    'gss_correlation': ('>', 0.5),  # 与GSS适度相关
                    'pattern_robustness': ('>', 0.6),  # 高稳健性
                },
                'weight': {
                    'spatial_autocorrelation': 0.4,
                    'gss_correlation': 0.3,
                    'pattern_robustness': 0.3
                }
            }
        }

    # ==================== 指标计算函数 ====================

    def _calculate_expression_sparsity(self, expr: np.ndarray):
        """零表达比例"""
        return np.mean(expr == 0)

    def _calculate_expression_variability(self, expr: np.ndarray):
        """变异系数"""
        non_zero_count = np.sum(expr > 0)
        total_count = len(expr)
        non_zero_ratio = non_zero_count / total_count

        if non_zero_ratio < 0.05:  # 非零表达值过少了，填充零值到全部spot的5%
            num_to_include = int(0.05 * total_count)
            sorted_indices = np.argsort(-expr)
            selected_expr = expr[sorted_indices[:num_to_include]]
        else:  # 非零表达值没有过少，正常计算非零表达值的变异系数
            selected_expr = expr[expr > 0]

        if len(selected_expr) < 10:
            return 0.0

        mean = np.mean(selected_expr)
        std = np.std(selected_expr)

        if mean < 1e-8:
            return 0.0

        cv = std / mean
        max_cv = 2.0  # 标准化：超过max_cv的视为1，0-max_cv之间线性缩放到0-1
        return min(1.0, max(0.0, cv / max_cv))

    def _calculate_expression_intensity(self, expr: np.ndarray):
        """表达强度计算"""
        non_zero = expr[expr > 0]
        if len(non_zero) < 50:
            return 0.0

        raw_intensity = np.median(non_zero)
        # 使用预计算的max_intensity归一化
        normalized_intensity = raw_intensity / self.max_intensity
        return min(1.0, max(0.0, normalized_intensity))

    def _calculate_expression_separation(self, expr: np.ndarray):
        """表达分离度"""
        non_zero = expr[expr > 0]
        if len(non_zero) < 10:
            return 0.0

        try:
            # 使用K-means尝试将表达值分为两类
            X = non_zero.reshape(-1, 1)

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            sil_score = silhouette_score(X, labels)
            if sil_score < 0.3:  # 聚类效果差，视为非双峰
                return 0.0

            # 计算两类中心的分离程度
            centers = kmeans.cluster_centers_.flatten()
            cluster_0 = non_zero[labels == 0]
            cluster_1 = non_zero[labels == 1]

            if len(cluster_0) == 0 or len(cluster_1) == 0:
                return 0.0

            # 分离度 = 类间距离 / (类内标准差之和)
            between_dist = abs(centers[0] - centers[1])
            within_std = np.std(cluster_0) + np.std(cluster_1)

            if within_std < 1e-8:
                return 0.0

            separation = between_dist / within_std
            return min(1.0, separation / 3.0)  # 归一化，假设分离度>3为很强
        except:
            return 0.0

    def _calculate_expression_bimodality(self, expr: np.ndarray):
        """表达双峰性"""
        non_zero = expr[expr > 0]
        if len(non_zero) < 10:
            return 0.0

        # 使用Hartigan's dip test的p值作为双峰性指标
        try:
            from scipy.stats import dip
            dip_stat, _ = dip(non_zero)
            raw_bimodality = 1 - dip_stat  # 值越大表示双峰性越强

            # 使用幂函数增强区分度
            normalized_bimodality = raw_bimodality ** 0.7

            return min(1.0, max(0.0, normalized_bimodality))
        except:
            # 退化为峰度计算，并进行归一化
            raw_kurtosis = abs(stats.kurtosis(non_zero))

            # 使用对数缩放处理长尾分布
            if raw_kurtosis > 0:
                normalized_kurtosis = min(1.0, np.log1p(raw_kurtosis) / np.log1p(10))
            else:
                normalized_kurtosis = 0.0

            return normalized_kurtosis

    def _calculate_morans_i(self, expr: np.ndarray):
        """Moran's I 空间自相关"""
        if np.sum(expr) == 0 or len(expr) < 10:
            return 0.0

        # 预计算常用值
        expr_mean = np.mean(expr)
        expr_centered = expr - expr_mean
        denominator = np.sum(expr_centered ** 2)

        if denominator == 0:
            return 0.0

        # 提取邻居索引（排除自身，形状为[N, k]）
        neighbors = self.indices[:, 1:]  # 排除自身

        # 更高效的计算方式
        neighbor_expr = expr_centered[neighbors]  # 形状: (n_cells, k)

        # 使用广播和逐元素乘法，避免显式循环
        numerator = np.sum(expr_centered[:, np.newaxis] * neighbor_expr)

        W = self.k * len(expr)
        return max(0, (len(expr) / W) * (numerator / denominator))

    def _calculate_pattern_robustness(self, expr: np.ndarray):
        """模式稳健性 - 检测表达模式是否稳定"""
        if np.sum(expr) == 0:
            return 0.0

        # 方法1: 基于子采样的稳定性
        n_subsamples = min(5, len(expr) // 10)
        if n_subsamples < 2:
            return 0.5  # 中性得分

        stability_scores = []
        for _ in range(5):  # 多次子采样
            # 生成子采样索引
            subsample_indices = np.random.choice(len(expr), size=len(expr) // 2, replace=False)
            subsample_expr = expr[subsample_indices]

            # 获取子采样对应的空间坐标
            subsample_coords = self.spatial_coords.iloc[subsample_indices].values

            if np.sum(subsample_expr) > 0 and len(subsample_coords) > 0:
                # 为子采样数据创建新的KDTree
                subsample_tree = KDTree(subsample_coords)

                # 计算子样本的Moran's I
                k = min(6, len(subsample_coords) - 1)
                if k < 1:  # 确保有足够的邻居
                    continue

                _, indices = subsample_tree.query(subsample_coords, k=k + 1)

                expr_mean = np.mean(subsample_expr)
                expr_centered = subsample_expr - expr_mean
                denominator = np.sum(expr_centered ** 2)

                if denominator > 0:
                    # 向量化计算分子
                    neighbor_centered = expr_centered[indices[:, 1:]]  # 排除自身
                    neighbor_sum = neighbor_centered.sum(axis=1)
                    numerator = np.sum(expr_centered * neighbor_sum)

                    W = k * len(subsample_expr)
                    moran_i = max(0, (len(subsample_expr) / W) * (numerator / denominator))
                    stability_scores.append(moran_i)

        if stability_scores:
            mean_score = np.mean(stability_scores)

            if mean_score < 1e-8:
                return 0.0

            cv = np.std(stability_scores) / mean_score
            stability_factor = 1.0 / (1.0 + cv)

            # 使用加权平均
            weight_mean = 0.4  # 模式强度
            weight_stability = 0.6  # 稳定性

            robustness = (weight_mean * mean_score +
                          weight_stability * stability_factor)

            return max(0, min(1, robustness))
        else:
            return 0.0

    def _calculate_concentration_score(self, expr: np.ndarray):
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

    def _calculate_gss_expr_correlation(self, expr: np.ndarray, gene: str):
        """GSS与表达量相关性"""
        if gene not in self.gss_df.index:
            return 0.0

        gss_scores = self.gss_df.loc[gene].values
        valid_indices = ~np.isnan(gss_scores)

        if np.sum(valid_indices) < 10:
            return 0.0

        corr, _ = stats.spearmanr(expr[valid_indices], gss_scores[valid_indices])
        return corr if not np.isnan(corr) else 0.0

    # ==================== 核心流程函数 ====================

    def _compute_max_intensity(self) -> float:
        """计算max_intensity：所有基因非零表达中位数的95%分位数"""
        non_zero_medians = []
        # 遍历基因（仅处理common_genes中的基因）
        for gene in self.gene_names:
            idx = self.gene_to_index[gene]
            expr = self.expr_matrix[idx]  # 基因的表达向量（细胞维度）
            non_zero = expr[expr > 0]  # 提取非零表达值
            # 仅保留有足够非零值的基因（避免噪声）
            if len(non_zero) >= 10:
                non_zero_medians.append(np.median(non_zero))

        # 处理空数据或极端情况
        if not non_zero_medians:
            return 1.0
        # 取95%分位数作为max_intensity（天然过滤极端高值）
        return np.percentile(non_zero_medians, 95) or 1.0

    def select_minimal_initial_genes(self) -> List[str]:
        """初始筛选 - 只排除明显不合适的基因"""

        # 条件1: 基本表达活性（避免完全沉默的基因）
        coverage = (self.expr_matrix > 0).mean(axis=1)
        min_coverage = 0.05

        # 条件2: 基本的GSS活性
        gss_coverage = (self.gss_df > 0).mean(axis=1)
        min_gss_coverage = 0.05

        mask = (coverage > min_coverage) & (gss_coverage > min_gss_coverage)
        selected_genes = np.array(self.gene_names)[mask]

        return selected_genes.tolist()

    def select_high_quality_genes(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """基于多重质量标准筛选高质量基因"""
        # 基础质量过滤
        quality_mask = (
                (metrics_df['pattern_robustness'] > 0.3) &  # 基本模式稳健性
                (metrics_df['spatial_autocorrelation'] > 0.1) &  # 基本空间聚集性
                (metrics_df['expression_concentration'] > 0.3)  # 基本表达集中性
        )

        filtered_df = metrics_df[quality_mask].copy()
        print(f"基础质量过滤后保留基因: {len(filtered_df)}个")

        # 类型分配
        typed_df = self.assign_gene_types(filtered_df)

        return typed_df

    def calculate_all_metrics(self, genes: List[str]) -> pd.DataFrame:
        """计算所有基因的所有指标"""
        print(f"开始计算 {len(genes)} 个基因的 {len(self.metric_functions)} 个指标...")

        def process_gene(gene):
            if gene not in self.gene_to_index:
                return None

            idx = self.gene_to_index[gene]
            expr = self.expr_matrix[idx]

            gene_metrics = {'gene': gene}

            # 计算所有指标
            for metric_name, metric_func in self.metric_functions.items():
                try:
                    # 特殊处理需要基因名的函数
                    if metric_name in ['gss_correlation']:
                        value = metric_func(expr, gene)
                    else:
                        value = metric_func(expr)
                    gene_metrics[metric_name] = value
                except Exception as e:
                    gene_metrics[metric_name] = 0.0
                    print(f"基因 {gene} 指标 {metric_name} 计算失败: {e}")

            return gene_metrics

        # 并行计算
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(process_gene)(gene) for gene in genes
        )

        results = [r for r in results if r is not None]
        return pd.DataFrame(results)

    def assign_gene_types(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        分配基因类型 - 必须满足类型的所有条件，使用权重计算分数
        """
        typed_df = metrics_df.copy()

        for gene_type, rule in self.gene_type_rules.items():
            type_col = f'is_{gene_type}'
            score_col = f'score_{gene_type}'

            # 初始化
            typed_df[type_col] = False
            typed_df[score_col] = 0.0

            # 检查是否满足所有条件
            all_conditions_met = pd.Series(True, index=metrics_df.index)

            for metric, (op, threshold) in rule['conditions'].items():
                if metric not in metrics_df.columns:
                    continue

                metric_vals = metrics_df[metric]

                if op == '>':
                    condition_mask = metric_vals > threshold
                elif op == '<':
                    condition_mask = metric_vals < threshold
                elif op == '>=':
                    condition_mask = metric_vals >= threshold
                elif op == '<=':
                    condition_mask = metric_vals <= threshold
                else:
                    condition_mask = pd.Series(False, index=metrics_df.index)

                all_conditions_met &= condition_mask

            # 只有满足所有条件的基因才被标记为该类型
            typed_df[type_col] = all_conditions_met

            # 计算加权分数（仅对属于该类型的基因）
            if all_conditions_met.any():
                weighted_scores = pd.Series(0.0, index=metrics_df.index)

                for metric, weight in rule['weight'].items():
                    if metric in metrics_df.columns:
                        # 使用原始指标值乘以权重
                        weighted_scores += metrics_df[metric] * weight

                # 只给属于该类型的基因赋值分数
                typed_df.loc[all_conditions_met, score_col] = weighted_scores[all_conditions_met]

        return typed_df

    def run_pipeline(self) -> Tuple[List[str], pd.DataFrame]:
        """运行完整的基因选择流程"""
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
                skip_info.to_csv(os.path.join(self.output_dir, "processing_log.csv"))
                print(f"验证结果已保存至 {self.output_dir}")
            return [], pd.DataFrame()

        # 基础筛选基因
        initial_genes = self.select_minimal_initial_genes()
        print(f"基础筛选后候选基因: {len(initial_genes)}个")

        if not initial_genes:
            return [], pd.DataFrame()

        # 计算所有指标
        metrics_df = self.calculate_all_metrics(initial_genes)

        # 质量过滤和类型分配
        typed_df = self.select_high_quality_genes(metrics_df)

        # 限制每类特异性基因的数量（最多前max_per_type个）
        max_per_type = 60  # 每类最大保留数量

        all_results = []
        for gene_type in self.gene_type_rules.keys():
            type_col = f'is_{gene_type}'
            score_col = f'score_{gene_type}'
            if type_col not in typed_df.columns or score_col not in typed_df.columns:
                continue

            # 筛选该类型的基因，并按分数排序
            type_genes_df = typed_df[typed_df[type_col]][['gene', score_col]]
            if len(type_genes_df) == 0:
                print(f"{gene_type} 保留 0 个基因（最多{max_per_type}个）")
                continue

            # 截断到最大数量
            top_genes = type_genes_df.sort_values(score_col, ascending=False).head(max_per_type)
            top_genes['gene_type'] = gene_type  # 标记类型
            top_genes.rename(columns={score_col: 'combined_score'}, inplace=True)  # 统一列名
            all_results.append(top_genes)
            print(f"{gene_type} 保留 {len(top_genes)} 个基因（最多{max_per_type}个）")

        # 汇总去重（每个基因保留最高分数的类型）
        if not all_results:
            print("无基因通过任何类型筛选")
            return [], pd.DataFrame()

        result_df = pd.concat(all_results).sort_values('combined_score', ascending=False)
        result_df = result_df.drop_duplicates('gene').reset_index(drop=True)  # 去重

        # 合并所有指标信息
        all_results_df = pd.merge(
            result_df,  # 筛选后的结果（含gene、gene_type、combined_score）
            metrics_df,  # 所有指标数据
            on='gene',  # 按基因名合并
            how='inner'  # 只保留两边都存在的基因（即筛选后的基因）
        )

        final_genes = all_results_df['gene'].tolist()
        print(f"最终筛选出 {len(final_genes)} 个特异性基因")

        # 保存结果
        os.makedirs(self.output_dir, exist_ok=True)
        if len(final_genes) > 0:
            all_results_df.to_csv(self.output_dir + "_selected_genes.csv", index=False, sep='\t')
        else:
            log_df = pd.DataFrame({
                'timestamp': [pd.Timestamp.now()],
                'cells': [self.cells],
                'initial_genes': [len(initial_genes)],
                'final_genes': [len(final_genes)],
                'status': ['completed']
            })
            log_df.to_csv(self.output_dir + "_processing_log.csv", index=False)

        print(f"详细结果已保存至 {self.output_dir}")
        return final_genes, all_results_df