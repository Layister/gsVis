import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Callable
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
import warnings
from sklearn.metrics import silhouette_score
from scipy import stats

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

        # 预计算max_intensity
        self.max_intensity = self._compute_max_intensity()

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
            'expression_intensity':self._calculate_expression_intensity,

            # 空间模式指标
            'spatial_autocorrelation': self._calculate_enhanced_moran,
            'spatial_focus': self._calculate_spatial_focus,
            'spatial_clustering': self._calculate_spatial_clustering,

            # 统计特征指标
            'intensity_contrast': self._calculate_intensity_contrast,
            'expression_separation': self._calculate_expression_separation,
            'expression_bimodality': self._calculate_expression_bimodality,
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

            'bimodal_expression_gene': {
                'description': '双峰表达基因 - 双峰分布特征基因',
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
            }
        }

    # ==================== 指标计算函数 ====================

    def _calculate_expression_sparsity(self, expr: np.ndarray):
        """表达稀疏性 - 零比例"""
        return np.mean(expr == 0)

    def _calculate_expression_concentration(self, expr: np.ndarray):
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

    def _calculate_expression_variability(self, expr: np.ndarray):
        """变异系数"""
        non_zero = expr[expr > 0]
        if len(non_zero) < 10:
            return 0.0

        mean = np.mean(non_zero)
        std = np.std(non_zero)

        if mean < 1e-8:
            return 0.0

        cv = std / mean

        # CV归一化：使用饱和函数cv/(cv+1) 将[0,∞)映射到[0,1)
        normalized_cv = cv / (cv + 1)
        return min(0.99, normalized_cv)  # 限制上限避免完全饱和

    def _calculate_enhanced_moran(self, expr: np.ndarray):
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

    def _calculate_spatial_focus(self, expr: np.ndarray):
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

    def _calculate_spatial_clustering(self, expr: np.ndarray):
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

    def _calculate_intensity_contrast(self, expr: np.ndarray):
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

    # ==================== 核心流程函数 ====================

    def _compute_max_intensity(self) -> float:
        """计算max_intensity：所有基因非零中位数的95%分位数"""
        # 收集所有基因的非零表达中位数
        non_zero_medians = []
        # 遍历基因（仅处理gene_counts中的基因）
        for gene in self.gene_counts.keys():
            if gene not in self.adata.var_names:
                continue
            # 获取基因表达向量
            expr = self._get_expression_vector(gene)
            non_zero = expr[expr > 0]
            if len(non_zero) >= 10:  # 仅保留有足够表达的基因
                non_zero_medians.append(np.median(non_zero))

        # 处理空数据或极端情况
        if not non_zero_medians:
            return 1.0
        # 取95%分位数作为max_intensity（天然过滤极端高值）
        return np.percentile(non_zero_medians, 95) or 1.0

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