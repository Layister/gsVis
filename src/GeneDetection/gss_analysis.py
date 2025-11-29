import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')


def extract_distribution_features(gss_matrix):
    """特征提取"""
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

    # 3. 近似多峰检测（会损失一些精度）
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
    """分布类型标注"""
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
        elif peak_count >= 2.5:  # 调整阈值，因为是连续值
            distribution_types[cluster_id] = 'multimodal'
        elif peak_count >= 1.5:
            distribution_types[cluster_id] = 'bimodal'
        elif skewness > 1.0:  # 调整阈值
            distribution_types[cluster_id] = 'right_skewed'
        elif skewness < -1.0:  # 调整阈值
            distribution_types[cluster_id] = 'left_skewed'
        elif kurtosis > 2.0:  # 调整阈值
            distribution_types[cluster_id] = 'heavy_tailed'
        elif ks_pvalue > 0.3:  # 调整阈值，因为是连续值
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
                # 50%分位
                non_zero_arrays = [row[row > 0] for row in cluster_data]
                percentiles = [np.percentile(arr, 50) if len(arr) > 0 else 0 for arr in non_zero_arrays]
                thresholds[cluster_mask] = percentiles

            elif dist_type == 'normal':
                # 均值+1.2倍标准差
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
                # 80%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 80, axis=1)

            elif dist_type == 'symmetric_bimodal':
                # 75%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 75, axis=1)

            else:  # 复杂类型
                # 85%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 85, axis=1)
    else:
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_data = gss_matrix[cluster_mask]
            dist_type = distribution_types.get(cluster_id, 'complex')

            if dist_type == 'zero_inflated':
                # 原95%分位 -> 50%分位
                non_zero_arrays = [row[row > 0] for row in cluster_data]
                percentiles = [np.percentile(arr, 50) if len(arr) > 0 else 0 for arr in non_zero_arrays]
                thresholds[cluster_mask] = percentiles

            elif dist_type == 'normal':
                # 原均值+2倍标准差 -> 原均值+1.7倍标准差
                means = np.mean(cluster_data, axis=1)
                stds = np.std(cluster_data, axis=1)
                thresholds[cluster_mask] = means + 1.7 * stds

            elif dist_type == 'right_skewed':
                # 原75%分位+1.5*IQR -> 降低到75%分位+1.0*IQR
                q75 = np.percentile(cluster_data, 75, axis=1)
                q25 = np.percentile(cluster_data, 25, axis=1)
                iqr = q75 - q25  # 用75%分位计算IQR，进一步降低阈值
                thresholds[cluster_mask] = q75 + 1.0 * iqr

            elif dist_type == 'multimodal':
                # 原85%分位 -> 80%分位
                thresholds[cluster_mask] = np.percentile(cluster_data, 80, axis=1)

            elif dist_type == 'symmetric_bimodal':
                # 原75%分位 -> 不变
                thresholds[cluster_mask] = np.percentile(cluster_data, 75, axis=1)

            else:  # 复杂类型
                # 原90%分位 -> 不变
                thresholds[cluster_mask] = np.percentile(cluster_data, 90, axis=1)

    return thresholds


def run_gss_analysis(mk_score_df, adata, output_dir, max_clusters=8):
    """基于GSS的筛选基因逻辑"""

    # 1. 数据准备
    valid_spots = [spot for spot in adata.obs_names if spot in mk_score_df.columns]
    print(f"处理 {len(valid_spots)} 个有效spots, {mk_score_df.shape[0]} 个基因")

    # 直接使用numpy数组，避免多次复制
    gss_matrix = mk_score_df[valid_spots].values.T.astype(np.float32)  # (spots, genes)

    # 2. 向量化特征提取
    print("向量化特征提取...")
    feature_matrix = extract_distribution_features(gss_matrix)

    # 3. 聚类分析（使用MiniBatchKMeans处理大数据）
    print("聚类分析...")
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
    print("批量筛选高GSS基因...")

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
    os.makedirs(output_dir, exist_ok=True)
    log_path = output_dir + "/results_summary.txt"

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
            save_path = f"{output_dir}/gss_distribution_{dist_type}.png"
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
