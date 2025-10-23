
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os
import json
import argparse
import scanpy as sc
import matplotlib.pyplot as plt
from src.Utils.DataProcess import get_self_data_dir, get_hest_data_dir, set_chinese_font, read_data
from src.Utils.GssGeneCalculator_parallel import GssGeneCalculator
import seaborn as sns

warnings.filterwarnings('ignore')


# def extract_distribution_features(gss_matrix):
#     """
#     向量化版本的特征提取
#     gss_matrix: (n_spots, n_genes) 的numpy数组
#     """
#     n_spots = gss_matrix.shape[0]
#
#     # 基础统计量 - 一次性计算所有spots
#     features = {'mean': np.mean(gss_matrix, axis=1), 'std': np.std(gss_matrix, axis=1),
#                 'median': np.median(gss_matrix, axis=1), 'range': np.ptp(gss_matrix, axis=1),
#                 'q10': np.percentile(gss_matrix, 10, axis=1), 'q90': np.percentile(gss_matrix, 90, axis=1),
#                 'q99': np.percentile(gss_matrix, 99, axis=1), 'zero_ratio': np.mean(gss_matrix == 0, axis=1),
#                 'skewness': np.array([stats.skew(row) for row in gss_matrix]),
#                 'kurtosis': np.array([stats.kurtosis(row) for row in gss_matrix]),
#                 'iqr': np.array([stats.iqr(row) for row in gss_matrix])}
#
#     # 需要循环但可以部分向量化的特征
#
#     # 变异系数
#     with np.errstate(divide='ignore', invalid='ignore'):
#         features['cv'] = np.where(features['mean'] != 0, features['std'] / features['mean'], 1.0)
#
#     # 批量处理KDE峰值检测
#     peak_counts = []
#     main_peak_positions = []
#     main_peak_heights = []
#     peak_distances = []
#
#     for i in range(n_spots):
#         try:
#             kde = stats.gaussian_kde(gss_matrix[i])
#             x_range = np.linspace(np.min(gss_matrix[i]), np.max(gss_matrix[i]), 100)  # 减少点数
#             kde_values = kde(x_range)
#             peaks, properties = find_peaks(kde_values, height=np.max(kde_values) * 0.3)
#
#             peak_counts.append(len(peaks))
#
#             if len(peaks) >= 1:
#                 main_peak_positions.append(x_range[peaks[0]])
#                 main_peak_heights.append(kde_values[peaks[0]])
#                 if len(peaks) >= 2:
#                     peak_distances.append(x_range[peaks[-1]] - x_range[peaks[0]])
#                 else:
#                     peak_distances.append(0)
#             else:
#                 main_peak_positions.append(np.median(gss_matrix[i]))
#                 main_peak_heights.append(1)
#                 peak_distances.append(0)
#
#         except:
#             peak_counts.append(1)
#             main_peak_positions.append(np.median(gss_matrix[i]))
#             main_peak_heights.append(1)
#             peak_distances.append(0)
#
#     features['peak_count'] = np.array(peak_counts)
#     features['main_peak_position'] = np.array(main_peak_positions)
#     features['main_peak_height'] = np.array(main_peak_heights)
#     features['peak_distance'] = np.array(peak_distances)
#
#     # 极端值占比
#     features['pct_above_mean2sd'] = np.mean(gss_matrix > (features['mean'][:, None] + 2 * features['std'][:, None]),
#                                             axis=1)
#     features['pct_above_median1.5iqr'] = np.mean(
#         gss_matrix > (features['median'][:, None] + 1.5 * features['iqr'][:, None]), axis=1)
#
#     # 尾比
#     with np.errstate(divide='ignore', invalid='ignore'):
#         tail_ratio = np.where(
#             (features['median'] - features['q10']) > 0,
#             (features['q99'] - features['median']) / (features['median'] - features['q10']),
#             1.0
#         )
#     features['tail_ratio'] = tail_ratio
#
#     # 熵（简化计算）
#     try:
#         hist_entropy = []
#         for i in range(n_spots):
#             hist, _ = np.histogram(gss_matrix[i], bins=20, density=True)
#             hist = hist[hist > 0]
#             hist_entropy.append(-np.sum(hist * np.log(hist)) if len(hist) > 0 else 0.0)
#         features['entropy'] = np.array(hist_entropy)
#     except:
#         features['entropy'] = np.zeros(n_spots)
#
#     # KS检验（抽样计算以节省时间）
#     ks_pvalues = []
#     sample_size = min(1000, gss_matrix.shape[1])
#     for i in range(n_spots):
#         try:
#             if gss_matrix.shape[1] > sample_size:
#                 sample_data = np.random.choice(gss_matrix[i], sample_size, replace=False)
#             else:
#                 sample_data = gss_matrix[i]
#             _, pvalue = stats.kstest(sample_data, 'norm')
#             ks_pvalues.append(pvalue)
#         except:
#             ks_pvalues.append(0.0)
#     features['ks_norm_pvalue'] = np.array(ks_pvalues)
#
#     return pd.DataFrame(features)
#
#
# def distribution_labeling(cluster_features):
#     """优化版的分布类型标注"""
#     distribution_types = {}
#
#     for cluster_id in cluster_features.index:
#         cf = cluster_features.loc[cluster_id]
#
#         # 安全获取特征值
#         zero_ratio = cf.get('zero_ratio', 0)
#         peak_count = cf.get('peak_count', 1)
#         skewness = cf.get('skewness', 0)
#         kurtosis = cf.get('kurtosis', 0)
#         data_range = cf.get('range', 0)
#         cv_val = cf.get('cv', 1)
#         ks_pvalue = cf.get('ks_norm_pvalue', 0)
#
#         # 1. 零膨胀分布检测
#         if zero_ratio > 0.7:
#             distribution_types[cluster_id] = 'zero_inflated'
#
#         # 2. 多峰分布细化
#         elif peak_count >= 3:
#             distribution_types[cluster_id] = 'multimodal'
#         elif peak_count == 2:
#             if abs(skewness) < 0.3:
#                 distribution_types[cluster_id] = 'symmetric_bimodal'
#             else:
#                 distribution_types[cluster_id] = 'asymmetric_bimodal'
#
#         # 3. 偏态分布细化
#         elif skewness > 1.2:
#             distribution_types[cluster_id] = 'right_skewed'
#         elif skewness < -1.2:
#             distribution_types[cluster_id] = 'left_skewed'
#         elif abs(skewness) > 0.8:
#             distribution_types[cluster_id] = 'moderately_skewed'
#
#         # 4. 重尾分布检测
#         elif kurtosis > 2.5:
#             distribution_types[cluster_id] = 'heavy_tailed'
#         elif kurtosis < 0.5:
#             distribution_types[cluster_id] = 'light_tailed'
#
#         # 5. 均匀分布检测
#         elif (data_range > 0 and
#               cv_val < 0.3 and
#               abs(skewness) < 0.5):
#             distribution_types[cluster_id] = 'uniform'
#
#         # 6. 正态分布
#         elif (abs(skewness) < 0.3 and
#               abs(kurtosis) < 1 and
#               ks_pvalue > 0.1):
#             distribution_types[cluster_id] = 'normal'
#
#         # 7. 复杂分布
#         else:
#             distribution_types[cluster_id] = 'complex'
#
#     return distribution_types


def extract_distribution_features(gss_matrix):
    """
    特征提取 - 在保持功能的前提下最大化速度
    特征名称与原版保持一致，确保后续步骤兼容
    """
    n_spots, n_genes = gss_matrix.shape

    # 基础统计量
    features = {
        'mean': np.mean(gss_matrix, axis=1),
        'std': np.std(gss_matrix, axis=1),
        'median': np.median(gss_matrix, axis=1),
        'range': np.ptp(gss_matrix, axis=1),
        'zero_ratio': np.mean(gss_matrix == 0, axis=1),
        'q10': np.percentile(gss_matrix, 10, axis=1),
        'q90': np.percentile(gss_matrix, 90, axis=1),
        'q99': np.percentile(gss_matrix, 99, axis=1),
        'iqr': np.percentile(gss_matrix, 75, axis=1) - np.percentile(gss_matrix, 25, axis=1),
    }

    # 计算分位数用于近似特征
    q25 = np.percentile(gss_matrix, 25, axis=1)
    q75 = np.percentile(gss_matrix, 75, axis=1)

    # 1. 近似偏度
    mean = features['mean'][:, None]
    std = features['std'][:, None]
    normalized = (gss_matrix - mean) / (std + 1e-8)
    features['skewness'] = np.mean(normalized ** 3, axis=1)  # 仍叫skewness

    # 2. 近似峰度
    features['kurtosis'] = np.mean(normalized ** 4, axis=1) - 3  # 仍叫kurtosis

    # 3. 近似多峰检测（保持原名称但含义不同）
    # 基于分位数间距的比值来检测多峰
    q_diff_ratio = (features['q90'] - q75) / (q75 - q25 + 1e-8)
    # 将连续值转换为近似的峰值计数
    peak_count_approx = np.ones(n_spots)  # 默认单峰
    peak_count_approx[q_diff_ratio > 2.0] = 3  # 高比值认为是多峰
    peak_count_approx[(q_diff_ratio > 1.2) & (q_diff_ratio <= 2.0)] = 2  # 中等比值认为是双峰
    features['peak_count'] = peak_count_approx  # 仍叫peak_count

    # 4. 主峰位置近似（用中位数代替）
    features['main_peak_position'] = features['median']  # 简化

    # 5. 简化的KS检验p值（用偏度峰度组合判断正态性）
    normal_likelihood = np.exp(-0.5 * (features['skewness'] ** 2 + features['kurtosis'] ** 2))
    features['ks_norm_pvalue'] = normal_likelihood  # 仍叫ks_norm_pvalue但含义不同

    # 6. 其他必要特征
    features['pct_above_mean2sd'] = np.mean(gss_matrix > (features['mean'][:, None] + 2 * features['std'][:, None]),
                                            axis=1)
    features['pct_above_median1.5iqr'] = np.mean(
        gss_matrix > (features['median'][:, None] + 1.5 * features['iqr'][:, None]), axis=1)

    # 变异系数
    features['cv'] = np.where(features['mean'] != 0, features['std'] / features['mean'], 0)

    # 尾比
    features['tail_ratio'] = np.where(
        (features['median'] - features['q10']) > 0,
        (features['q99'] - features['median']) / (features['median'] - features['q10'] + 1e-8),
        1.0
    )

    return pd.DataFrame(features)


def distribution_labeling(cluster_features):
    """分布类型标注 - 调整阈值以适应近似特征"""
    distribution_types = {}

    for cluster_id in cluster_features.index:
        cf = cluster_features.loc[cluster_id]

        zero_ratio = cf.get('zero_ratio', 0)
        peak_count = cf.get('peak_count', 1)  # 近似值
        skewness = cf.get('skewness', 0)  # 近似值
        kurtosis = cf.get('kurtosis', 0)  # 近似值
        ks_pvalue = cf.get('ks_norm_pvalue', 0)  # 近似值

        # 调整阈值以适应近似特征
        if zero_ratio > 0.7:
            distribution_types[cluster_id] = 'zero_inflated'
        elif peak_count >= 2.5:  # 调整阈值，因为现在是连续值
            distribution_types[cluster_id] = 'multimodal'
        elif peak_count >= 1.5:
            distribution_types[cluster_id] = 'bimodal'
        elif skewness > 1.0:  # 调整阈值
            distribution_types[cluster_id] = 'right_skewed'
        elif skewness < -1.0:  # 调整阈值
            distribution_types[cluster_id] = 'left_skewed'
        elif kurtosis > 2.0:  # 调整阈值
            distribution_types[cluster_id] = 'heavy_tailed'
        elif ks_pvalue > 0.3:  # 调整阈值，因为现在是连续值
            distribution_types[cluster_id] = 'normal'
        else:
            distribution_types[cluster_id] = 'complex'

    return distribution_types


def precompute_thresholds_batch(gss_matrix, cluster_labels, distribution_types):
    """批量预计算所有spots的阈值"""
    n_spots = gss_matrix.shape[0]
    n_genes = gss_matrix.shape[1]
    thresholds = np.zeros(n_spots)

    if(n_genes < 1000):
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_data = gss_matrix[cluster_mask]
            dist_type = distribution_types.get(cluster_id, 'complex')

            if dist_type == 'zero_inflated':
                # 80%分位
                non_zero_arrays = [row[row > 0] for row in cluster_data]
                percentiles = [np.percentile(arr, 80) if len(arr) > 0 else 0 for arr in non_zero_arrays]
                thresholds[cluster_mask] = percentiles

            elif dist_type == 'normal':
                # 均值+1.3倍标准差
                means = np.mean(cluster_data, axis=1)
                stds = np.std(cluster_data, axis=1)
                thresholds[cluster_mask] = means + 1.3 * stds

            elif dist_type == 'right_skewed':
                # 75%分位+0.1*IQR
                q75 = np.percentile(cluster_data, 75, axis=1)
                q25 = np.percentile(cluster_data, 25, axis=1)
                iqr = q75 - q25  # 用75%分位计算IQR，进一步降低阈值
                thresholds[cluster_mask] = q75 + 0.1 * iqr

            elif dist_type == 'multimodal':
                # 85%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 85, axis=1)

            elif dist_type == 'symmetric_bimodal':
                # 75%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 75, axis=1)

            else:  # 复杂类型
                # 88%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 88, axis=1)
    else:
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_data = gss_matrix[cluster_mask]
            dist_type = distribution_types.get(cluster_id, 'complex')

            if dist_type == 'zero_inflated':
                # 原95%分位 -> 92%分位
                non_zero_arrays = [row[row > 0] for row in cluster_data]
                percentiles = [np.percentile(arr, 92) if len(arr) > 0 else 0 for arr in non_zero_arrays]
                thresholds[cluster_mask] = percentiles

            elif dist_type == 'normal':
                # 原均值+2倍标准差 -> 原均值+1.8倍标准差
                means = np.mean(cluster_data, axis=1)
                stds = np.std(cluster_data, axis=1)
                thresholds[cluster_mask] = means + 1.8 * stds

            elif dist_type == 'right_skewed':
                # 原75%分位+1.5*IQR -> 降低到75%分位+1.2*IQR
                q75 = np.percentile(cluster_data, 75, axis=1)
                q25 = np.percentile(cluster_data, 25, axis=1)
                iqr = q75 - q25  # 用75%分位计算IQR，进一步降低阈值
                thresholds[cluster_mask] = q75 + 1.2 * iqr

            elif dist_type == 'multimodal':
                # 原85%分位 -> 不变
                thresholds[cluster_mask] = np.percentile(cluster_data, 85, axis=1)

            elif dist_type == 'symmetric_bimodal':
                # 原75%分位 -> 不变
                thresholds[cluster_mask] = np.percentile(cluster_data, 75, axis=1)

            else:  # 复杂类型
                # 原90%分位 -> 增加到98%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 98, axis=1)

    return thresholds


def run_gss_analysis(mk_score_df, adata, output_dir, max_clusters=8):
    """针对大规模数据优化的GSS分析逻辑"""
    print("开始GSS分析...")

    # 1. 数据准备
    valid_spots = [spot for spot in adata.obs_names if spot in mk_score_df.columns]
    print(f"处理 {len(valid_spots)} 个有效spots, {mk_score_df.shape[0]} 个基因")

    # 直接使用numpy数组，避免多次复制
    gss_matrix = mk_score_df[valid_spots].values.T.astype(np.float32)  # (spots, genes)，使用float32节省内存

    # 2. 向量化特征提取
    print("步骤1: 向量化特征提取...")
    feature_matrix = extract_distribution_features(gss_matrix)

    # 3. 优化的聚类 - 使用MiniBatchKMeans处理大数据
    print("步骤2: 聚类分析...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)

    # 确定最优聚类数（简化版）
    n_samples = len(feature_matrix)
    max_k = min(max_clusters, n_samples // 10)  # 确保每个簇至少有10个样本

    best_k = 2
    best_silhouette = -1

    for k in range(2, max_k + 1):
        if n_samples < k * 5:  # 更严格的样本数检查
            continue

        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=3)
        labels = kmeans.fit_predict(features_scaled)

        if len(np.unique(labels)) > 1:
            try:
                # 使用子采样计算轮廓系数以加速
                if n_samples > 1000:
                    indices = np.random.choice(n_samples, min(1000, n_samples), replace=False)
                    sil_sample = silhouette_score(features_scaled[indices], labels[indices])
                else:
                    sil_sample = silhouette_score(features_scaled, labels)

                if sil_sample > best_silhouette:
                    best_silhouette = sil_sample
                    best_k = k
            except:
                continue

    print(f"选择最优聚类数: {best_k}, 轮廓系数: {best_silhouette:.3f}")

    # 最终聚类
    kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=1000, n_init=5)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # 4. 分布类型标注
    cluster_features = feature_matrix.groupby(cluster_labels).mean()
    distribution_types = distribution_labeling(cluster_features)

    # 5. 批量基因筛选
    print("步骤3: 批量筛选高GSS基因...")

    # 预计算所有spot的阈值
    thresholds = precompute_thresholds_batch(gss_matrix, cluster_labels, distribution_types)

    spot_genes_dict = {}
    spot_metadata = {}
    gene_names = mk_score_df.index.tolist()

    for i, spot_name in enumerate(valid_spots):
        cluster_id = cluster_labels[i]
        dist_type = distribution_types.get(cluster_id, 'complex')

        # 使用预计算的阈值
        threshold = thresholds[i]
        high_genes_indices = np.where(gss_matrix[i] > threshold)[0]
        high_genes = [gene_names[idx] for idx in high_genes_indices]

        spot_genes_dict[spot_name] = high_genes
        spot_metadata[spot_name] = {
            'cluster': cluster_id,
            'distribution_type': dist_type,
            'n_high_genes': len(high_genes)
        }

    # 统计各类分布的数量
    dist_type_counts = {}
    for metadata in spot_metadata.values():
        dist_type = metadata['distribution_type']
        dist_type_counts[dist_type] = dist_type_counts.get(dist_type, 0) + 1

    cluster_analysis = {
        'cluster_labels': cluster_labels,
        'distribution_types': distribution_types,
        'n_clusters': len(set(cluster_labels)),
        'distribution_type_counts': dist_type_counts,
        'scaler': scaler,
        'kmeans': kmeans
    }

    # 返回结果
    results = {
        'spot_genes_dict': spot_genes_dict,
        'spot_metadata': spot_metadata,
        'cluster_analysis': cluster_analysis,
        'feature_matrix': feature_matrix
    }

    # 可视化
    visualize_results(
        results=results,
        mk_score_df=mk_score_df,
        thresholds = thresholds,
        top_n_spots=6,  # 每种分布类型展示6个spot
        output_dir=output_dir
    )

    print("分析完成!")
    return results


def visualize_results(results, mk_score_df, thresholds, top_n_spots=6, output_dir=None):
    """
    展示results中的详细信息并可视化
    参数:
    - results: run_gss_analysis的输出结果字典
    - mk_score_df: 包含GSS值的DataFrame（行为基因，列为spot）
    - thresholds: 预计算好的各个spot阈值
    - top_n_spots: 每种分布类型展示的spot数量
    - output_dir: 图像和日志保存目录（None则不保存）
    """
    sns.set_style("whitegrid")

    # 确定日志文件路径
    log_path = output_dir + "_results_summary.txt"

    # 用于存储日志内容的列表
    log_content = []

    # --------------------------
    # 1. 收集results信息摘要
    # --------------------------
    log_content.append("=" * 50)
    log_content.append("results信息摘要")
    log_content.append("=" * 50)

    # 1.1 聚类和分布类型统计
    cluster_analysis = results['cluster_analysis']
    dist_type_counts = cluster_analysis['distribution_type_counts']
    n_clusters = cluster_analysis['n_clusters']
    total_spots = len(results['spot_metadata'])

    log_content.append(f"总spot数量: {total_spots}")
    log_content.append(f"聚类数量: {n_clusters}")
    log_content.append("\n分布类型统计:")

    dist_df = pd.DataFrame(list(dist_type_counts.items()), columns=['分布类型', 'spot数量'])
    dist_df['占比'] = dist_df['spot数量'] / total_spots * 100
    dist_df = dist_df.sort_values('spot数量', ascending=False)
    log_content.append(dist_df.to_string(index=False))  # 添加数据框内容

    # 1.2 高GSS基因数量统计
    spot_genes = results['spot_genes_dict']
    gene_counts = [len(genes) for genes in spot_genes.values()]
    log_content.append(f"\n单个spot的高GSS基因数量分布:")
    log_content.append(f"  均值: {np.mean(gene_counts):.1f}")
    log_content.append(f"  中位数: {np.median(gene_counts)}")
    log_content.append(f"  范围: [{np.min(gene_counts)}, {np.max(gene_counts)}]")

    # --------------------------
    # 2. 收集可视化过程信息
    # --------------------------
    log_content.append("\n" + "=" * 50)
    log_content.append("可视化GSS值分布")
    log_content.append("=" * 50)

    # 按分布类型分组spot
    dist_to_spots = {}
    for spot_name, meta in results['spot_metadata'].items():
        dist_type = meta['distribution_type']
        if dist_type not in dist_to_spots:
            dist_to_spots[dist_type] = []
        dist_to_spots[dist_type].append(spot_name)

    # 获取spot名称列表
    spots_names = list(results['spot_metadata'].keys())

    # 为每种分布类型绘制代表性spot的GSS分布
    for dist_type, spots in dist_to_spots.items():
        selected_spots = spots[:top_n_spots]
        if not selected_spots:
            continue

        log_content.append(f"\n分布类型: {dist_type}（展示{len(selected_spots)}/{len(spots)}个spot）")

        # 创建子图
        ncols = min(3, len(selected_spots))
        nrows = (len(selected_spots) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten() if nrows * ncols > 1 else [axes]

        for i, spot_name in enumerate(selected_spots):
            ax = axes[i]
            gss_values = mk_score_df[spot_name].values

            # 使用预计算筛选阈值
            spot_index = spots_names.index(spot_name)
            threshold = thresholds[spot_index]

            # 绘制直方图
            sns.histplot(gss_values, bins=30, kde=True, ax=ax, color='#66b3ff', edgecolor='black')
            if threshold is not None:
                ax.axvline(x=threshold, color='red', linestyle='--',
                           label=f'screening threshold: {threshold:.2f}')
            n_high_genes = len(results['spot_genes_dict'][spot_name])
            ax.set_title(f"spot: {spot_name}\nnumber of high GSS genes: {n_high_genes}", fontsize=10)
            ax.set_xlabel("Gss")
            ax.set_ylabel("Genes")
            ax.legend(fontsize=8)

        plt.tight_layout()

        # 保存图像
        if output_dir:
            save_path = f"{output_dir}_gss_distribution_{dist_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log_content.append(f"已保存图像: {save_path}")  # 将保存路径写入日志
        plt.close()

    # --------------------------
    # 3. 保存日志到文件
    # --------------------------
    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))  # 用换行符连接所有日志内容
        print(f"结果信息已保存至: {log_path}")  # 仅提示日志保存路径到控制台


def calculate_adaptive_threshold(gene_counts, visualize=True, output_dir=None):
    """
    基于累积分布手动识别拐点计算阈值，同时可视化基因出现次数的直方图分布和累积分布
    """
    counts = np.array(list(gene_counts.values()))
    sorted_counts = np.sort(counts)[::-1]
    cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts)
    x = np.arange(len(sorted_counts))

    target_ratio = 0.5
    knee_idx = np.argmax(cumulative >= target_ratio)

    if knee_idx < len(sorted_counts):
        threshold = sorted_counts[knee_idx]
    else:
        threshold = np.median(counts)

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.hist(counts, bins=30, edgecolor='black', color='#66b3ff')
        ax1.axvline(x=threshold, color='red', linestyle='--',
                    label=f'screening threshold: {threshold:.1f}')
        ax1.set_title('occurrence number of genes histogram')
        ax1.set_xlabel('occurrence number of genes')
        ax1.set_ylabel('number of genes')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(x, cumulative, 'b-', linewidth=2)
        ax2.axvline(x=knee_idx, color='red', linestyle='--',
                    label=f'inflection point: {knee_idx}')
        ax2.axhline(y=cumulative[knee_idx], color='green', linestyle='--')
        ax2.set_title(f'cumulative distribution(goal proportion {target_ratio * 100:.0f}%)')
        ax2.set_xlabel('number of genes(descending)')
        ax2.set_ylabel('cumulative proportion')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}_gene_counts_distribution.png', dpi=300, bbox_inches='tight')
            print(f"分布图已保存至: {output_dir}_gene_counts_distribution.png")
        plt.close()

    return threshold


def calculate_gene_func(adata, gene_counts,
                        output_dir=None,
                        score_threshold=0.5):
    """基因功能计算"""
    adaptive_threshold = calculate_adaptive_threshold(gene_counts=gene_counts,
                                                      visualize=True,
                                                      output_dir=output_dir)

    filtered_genes_len = len([key for key, value in gene_counts.items() if value >= adaptive_threshold])
    print(f"占据50%的度的频率阈值是{adaptive_threshold},有{filtered_genes_len}个基因通过筛选（仅供展示）！！！")

    selector = GssGeneCalculator(
        adata=adata,
        gene_counts=gene_counts,
        #score_threshold=score_threshold
    )

    selected_genes, results = selector.run_pipeline()
    if(len(selected_genes) > 0):
        results.to_csv(output_dir + "_selected_genes.csv", index=False, sep='\t')
    else:
        results.to_csv(output_dir + "_processing_log.csv")
    print(f"验证结果已保存至 {output_dir}")

    return selected_genes, results


def plot_gene_spatial(mk_score_df, adata, gene_name, high_specificity_genes_per_cell,
                      output_dir=None, visual_indicators=None,
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

    new_columns = {}
    if visual_indicators == "GSS" and gene_name in mk_score_df.index:
        print(f"基因 '{gene_name}' 使用GSS分数数据")
        obs_column_name = f"Marker_score_{gene_name}"
        new_columns[obs_column_name] = mk_score_df.loc[gene_name, adata.obs_names].values
        color = obs_column_name
    else:
        print(f"基因 '{gene_name}' 使用原始基因表达值")
        color = gene_name  # 直接使用 adata.var_names 中的基因

    # 一次性合并新列
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=adata.obs.index)
        adata.obs = pd.concat([adata.obs, new_df], axis=1)

    # 获取所有细胞的空间坐标
    spatial_coords = adata.obsm['spatial']

    # 找到四个角落的细胞
    corner_cells = []
    # 左上: x最小，y最小
    left_upper_idx = np.argmin(spatial_coords[:, 0] + spatial_coords[:, 1])
    corner_cells.append(adata.obs_names[left_upper_idx])
    # 右上: x最大，y最小
    right_upper_idx = np.argmax(spatial_coords[:, 0] - spatial_coords[:, 1])
    corner_cells.append(adata.obs_names[right_upper_idx])
    # 左下: x最小，y最大
    left_lower_idx = np.argmin(spatial_coords[:, 0] - spatial_coords[:, 1])
    corner_cells.append(adata.obs_names[left_lower_idx])
    # 右下: x最大，y最大
    right_lower_idx = np.argmax(spatial_coords[:, 0] + spatial_coords[:, 1])
    corner_cells.append(adata.obs_names[right_lower_idx])

    # 创建新的AnnData对象，包含高特异性细胞和四个角落的细胞
    cells_with_gene = [cell for cell, genes in high_specificity_genes_per_cell.items() if gene_name in genes]
    cells_to_include = list(set(cells_with_gene + corner_cells))
    adata_subset = adata[cells_to_include].copy()

    # 确保新数据集中有表达值
    if visual_indicators == "GSS" and obs_column_name not in adata_subset.obs.columns:
        adata_subset.obs[obs_column_name] = mk_score_df.loc[gene_name, adata_subset.obs_names].values

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 使用 Scanpy 的空间可视化函数
    try:
        fig = sc.pl.spatial(
            adata_subset,
            cmap=cmap,
            color=color,
            # size=size,
            # alpha=alpha,
            alpha_img=background_alpha,
            img_key='downscaled_fullres',  # 'hires'
            title=f'{gene_name}({visual_indicators})',
            return_fig=True,
            frameon=True,
            show=show,
        )

        fig.savefig(os.path.join(output_dir, f"{color}_calibrated.png"), dpi=400)
        plt.close(fig)

    except Exception as e:
        print(f"可视化失败: {e}")


def plot_multiple_genes(mk_score_df, adata, gene_names, high_specificity_genes_per_cell,
                        output_dir=None, visual_indicators=None,
                        cmap='viridis', size=0.8, alpha=0.8,
                        background_alpha=0.7, show=False):
    """
    可视化多个基因在空间上的表达分布

    参数:
    - mk_score_df: 标记分数DataFrame
    - adata: AnnData对象
    - gene_names: 要可视化的基因名称列表
    - high_specificity_genes_per_cell: 每个细胞对应的高特异性基因列表的字典
    - output_dir: 图像保存目录
    - cmap: 颜色映射
    - size: 点大小
    - alpha: 透明度
    """
    for gene_name in gene_names:
        plot_gene_spatial(
            mk_score_df, adata, gene_name, high_specificity_genes_per_cell,
            output_dir=output_dir,
            visual_indicators=visual_indicators,
            cmap=cmap,
            size=size,
            alpha=alpha,
            background_alpha=background_alpha,
            show=show  # 批量绘制时不显示，提高效率
        )

    print(f"已完成 {len(gene_names)} 个基因的可视化")


def analyze_single_sample(output_dir, feather_path, h5ad_path):
    """处理单个样本：计算Top GSS基因并保存结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取GSS数据
    mk_score_df, adata = read_data(feather_path, h5ad_path)

    # 2. 计算样本各个spot的高GSS基因
    results = run_gss_analysis(mk_score_df, adata, output_dir)

    # 3. 获取高特异性基因并统计频率
    high_specificity_genes_per_cell = results['spot_genes_dict']  # spot -> 基因列表的字典
    gene_counts = {}
    for genes in high_specificity_genes_per_cell.values():
        for gene in genes:
            # 若基因已在字典中，计数+1；否则初始化为1
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

    # 4. 基因筛选
    selected_genes, _ = calculate_gene_func(
        adata, gene_counts,
        output_dir=output_dir,
        score_threshold=0.40,  # 判断 “特异性” 的阈值
    )
    print(selected_genes)

    # 5. 可视化空间表达模式
    plot_multiple_genes(
        mk_score_df, adata, selected_genes,
        high_specificity_genes_per_cell,
        output_dir=output_dir,
        visual_indicators="Expr",  # ["GSS", "Expr"]
        cmap='viridis',
        size=1.0,
        alpha=0.6
    )


def batch_analysis(select_n, json_path, data_dir, output_root):
    """
    批量分析JSON中所有样本的Top GSS基因
    :param select_n: 至多分析的样本数量
    :param json_path: 样本映射JSON路径
    :param data_dir: 数据存放根目录
    :param output_root: 结果输出根目录
    """
    # 1. 读取JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2. 遍历物种
    n = 5  # 每种癌症选取的样本数
    method = 'calculateTopGSS'
    species = "Homo sapiens"
    selected_cancers = ['COAD', 'COADREAD', 'EPM', 'GBM', 'HCC', 'HGSOC', 'IDC',
                        'ILC', 'LUAD', 'PAAD', 'PRAD', 'READ', 'SCCRCC', 'SKCM']
    species_data = data.get(species, {})
    cancer_types = species_data.get("cancer_types", {})

    # 3. 遍历每个癌症类型
    i = 1
    for cancer_type, sample_ids in cancer_types.items():
        # 4. 遍历该癌症类型下的每个样本
        ids_to_query = sample_ids[:] # min(n, len(sample_ids))
        for id in ids_to_query:
            # 构建文件路径
            if i <= select_n and len(ids_to_query) > 1 and cancer_type in selected_cancers:
                output_dir = os.path.join(output_root, species, cancer_type, id, method)
                feather_path = os.path.join(data_dir, species, cancer_type, id,
                                            f"latent_to_gene/{id}_gene_marker_score.feather")
                h5ad_path = os.path.join(data_dir, species, cancer_type, id,
                                         f"find_latent_representations/{id}_add_latent.h5ad")

                # 5. 调用单样本分析函数
                print(f"------------开始分析：物种={species}, 癌症类型={cancer_type}, 样本={id}------------")
                analyze_single_sample(output_dir, feather_path, h5ad_path)
                print(f"完成分析：{output_dir}")
            else:
                if i > select_n:
                    break

            i += 1


if __name__ == "__main__":

    # 命令行参数
    parser = argparse.ArgumentParser(description="批量计算样本的Top GSS基因")
    parser.add_argument("--select-n", type=str,
                        default=9999,
                        help="至多分析样本")
    parser.add_argument("--json", type=str,
                        default="/home/wuyang/hest-data/cancer_samples.json",
                        help="样本映射JSON路径")
    parser.add_argument("--data-dir", type=str,
                        default="/home/wuyang/hest-data/process/",
                        help="数据根目录")
    parser.add_argument("--output-root", type=str,
                        default="../output/HEST",
                        help="结果输出根目录")

    # 解析命令行参数
    args = parser.parse_args()
    # 设置中文字体
    set_chinese_font()

    # 执行批量分析
    batch_analysis(
        select_n=args.select_n,
        json_path=args.json,
        data_dir=args.data_dir,
        output_root=args.output_root
    )