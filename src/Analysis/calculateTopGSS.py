
import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from src.Utils.DataProcess import get_self_data_dir, get_hest_data_dir, set_chinese_font, read_data
from src.Utils.GssGeneCalculator import GssGeneCalculator


def get_high_specificity_genes_kde_single_cell(gss_values, min_peak_height=0.05):
    """
    单个细胞的KDE法筛选高特异性基因
    逻辑：通过核密度估计找到GSS值分布的峰值，取峰值右侧显著下降后的高值基因
    参数：
    - gss_values: 单个细胞的GSS值（pandas Series，索引为基因名）
    - min_peak_height: 检测峰值的最小高度阈值（归一化后）
    返回：
    - 筛选出的高特异性基因列表
    """
    # 提取非零GSS值（零值无意义）
    non_zero_gss = gss_values[gss_values > 0].sort_values(ascending=False)

    # 核密度估计（KDE）
    kde = gaussian_kde(non_zero_gss.values)
    x = np.linspace(non_zero_gss.min(), non_zero_gss.max(), 1000)  # GSS值范围
    y = kde(x)  # 密度值
    y_norm = y / y.max()  # 归一化到0-1，便于统一阈值

    # 检测峰值（高特异性基因可能形成独立峰值）
    peaks, peak_info = find_peaks(y_norm, height=min_peak_height)

    # 选择最高的峰值作为主峰值（最可能的高特异性基因分布中心）
    main_peak_idx = peaks[np.argmax(peak_info['peak_heights'])]
    main_peak_gss = x[main_peak_idx]

    # 找到峰值右侧密度下降到峰值2%的位置作为阈值（动态比例）
    right_of_peak = x[x >= main_peak_gss]  # 仅看峰值右侧
    right_density = y_norm[x >= main_peak_gss]
    threshold_idx = np.argmax(right_density <= 0.02 * peak_info['peak_heights'].max())

    if threshold_idx == 0:  # 未找到下降点时，用峰值作为阈值
        threshold = main_peak_gss
    else:
        threshold = right_of_peak[threshold_idx]

    # 返回阈值以上的基因
    return non_zero_gss[non_zero_gss >= threshold].index.tolist()


def get_high_specificity_genes_nd_single_cell(gss_values, multiplier=3, use_median=False):
    """
    单个细胞的均值+标准差（或中位数+四分位距）筛选法
    逻辑：取显著高于整体水平的基因（适应偏态分布）
    参数：
    - gss_values: 单个细胞的GSS值（pandas Series，索引为基因名）
    - multiplier: 倍数（标准差或四分位距的倍数）
    - use_median: 是否使用中位数+四分位距（适合偏态分布）
    返回：
    - 筛选出的高特异性基因列表
    """
    # 提取非零GSS值
    non_zero_gss = gss_values[gss_values > 0]

    # 计算阈值（根据分布类型选择方法）
    if use_median:
        # 中位数+四分位距（适合偏态分布，抗极端值）
        median = np.median(non_zero_gss.values)
        iqr = np.percentile(non_zero_gss.values, 75) - np.percentile(non_zero_gss.values, 25)
        threshold = median + multiplier * iqr
    else:
        # 均值+标准差（适合近似正态分布）
        mean = np.mean(non_zero_gss.values)
        std = np.std(non_zero_gss.values)
        threshold = mean + multiplier * std

    # 确保阈值不低于最大GSS值的10%（避免阈值过高导致无结果）
    threshold = max(threshold, non_zero_gss.max() * 0.1)
    return non_zero_gss[non_zero_gss >= threshold].index.tolist()


def get_high_specificity_genes_top_n_single_cell(gss_values, top_n=10, min_threshold=None):
    """
    单个细胞的Top-N筛选法
    逻辑：取GSS值最高的N个基因（结合绝对阈值避免低质量基因）
    参数：
    - gss_values: 单个细胞的GSS值（pandas Series，索引为基因名）
    - top_n: 最多返回的基因数量
    - min_threshold: 最低GSS值阈值（低于此值的基因即使排名靠前也排除）
    返回：
    - 筛选出的高特异性基因列表
    """
    # 降序排列所有基因（包括零值，但零值会被过滤）
    sorted_gss = gss_values.sort_values(ascending=False)
    # 过滤零值
    sorted_gss = sorted_gss[sorted_gss > 0]

    # 应用绝对阈值（如果设置）
    if min_threshold is not None:
        sorted_gss = sorted_gss[sorted_gss >= min_threshold]

    # 取前N个（如果不足N个则返回全部）
    n = min(top_n, len(sorted_gss))
    return sorted_gss.head(n).index.tolist()


def get_distribution_features(gss_values, min_peak_height=0.05, visualize=True):
    """
    分析GSS值的分布特征，可选可视化分布曲线及关键指标
    参数：
    - gss_values: 单个细胞的GSS值（pandas Series）
    - min_peak_height: 检测峰值的最小高度阈值（归一化后）
    - visualize: 是否生成可视化图
    返回：
    - features: 分布特征字典
    """
    non_zero_gss = gss_values[gss_values > 0].values.astype(np.float64)
    n_non_zero = len(non_zero_gss)

    # 1. 基础统计特征计算
    if n_non_zero == 0:
        features = {
            "n_non_zero": 0,
            "skewness": 0,  # 偏度（正值=右偏，负值=左偏，0=对称）
            "kurtosis": 0,  # 峰度（正值=尖峰，负值=平峰，0=正态）
            "n_peaks": 0,  # KDE检测到的峰值数量
            "has_high_tail": False,  # 是否有高值尾部（前0.5%高值/中位值 > 1.3）
            "mean": 0,
            "median": 0,
            "top5p": 0
        }
    else:
        mean_gss = np.mean(non_zero_gss)
        median_gss = np.median(non_zero_gss)
        top5p_gss = np.percentile(non_zero_gss, 99.5)  # 前0.5%高值阈值

        # 检查数据是否几乎恒定
        data_range = np.max(non_zero_gss) - np.min(non_zero_gss)
        if data_range < 1e-8:  # 使用范围而不是方差作为判断条件
            skew_val = 0.0
            kurt_val = 0.0
        else:
            # 使用更稳健的偏度和峰度计算方法
            try:
                # 使用基于分位数的偏度和峰度估计（Bowley偏度和Moors峰度）
                q75, q25 = np.percentile(non_zero_gss, [75, 25])
                q90, q10 = np.percentile(non_zero_gss, [90, 10])
                q87_5, q62_5, q37_5, q12_5 = np.percentile(non_zero_gss, [87.5, 62.5, 37.5, 12.5])

                # Bowley偏度
                if (q75 - q25) > 1e-10:
                    skew_val = (q75 + q25 - 2 * median_gss) / (q75 - q25)
                else:
                    skew_val = 0.0

                # Moors峰度
                if (q75 - q25) > 1e-10:
                    kurt_val = (q87_5 - q62_5 - q37_5 + q12_5) / (q75 - q25)
                else:
                    kurt_val = 0.0
            except:
                skew_val = 0.0
                kurt_val = 0.0

        features = {
            "n_non_zero": n_non_zero,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "mean": mean_gss,
            "median": median_gss,
            "top5p": top5p_gss,
            "has_high_tail": (top5p_gss / median_gss) > 1.3 if median_gss > 0 else False
        }

        # 2. KDE峰值检测（仅当非零值足够多时）
        if n_non_zero >= 100:  # 数据量太少时不检测峰值（避免噪声）
            kde = gaussian_kde(non_zero_gss)
            x = np.linspace(non_zero_gss.min(), non_zero_gss.max(), 1000, dtype=np.float64)
            y = kde(x)
            y_norm = y / y.max()  # 归一化到0-1（方便统一峰值高度阈值）
            peaks, peak_info = find_peaks(y_norm, height=min_peak_height)
            features["n_peaks"] = len(peaks)
            features["kde_x"] = x  # 保存KDE曲线x轴（用于可视化）
            features["kde_y"] = y  # 保存KDE曲线y轴
            features["peaks_x"] = x[peaks] if len(peaks) > 0 else []  # 峰值对应的GSS值
        else:
            features["n_peaks"] = 0
            features["kde_x"] = None
            features["kde_y"] = None
            features["peaks_x"] = []

    # 3. 可视化分布（若开启）
    if visualize:
        # 绘制子图：1行2列（直方图+KDE曲线 + 统计指标标注）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 子图1：直方图+KDE曲线+峰值标记
        if n_non_zero > 0:
            # 直方图（展示原始数据分布）
            ax1.hist(non_zero_gss, bins=min(20, n_non_zero // 2), edgecolor="black", alpha=0.6, label="GSS值直方图")
            # KDE曲线（平滑分布）
            if features["kde_x"] is not None:
                ax1.plot(features["kde_x"], features["kde_y"], color="red", linewidth=2, label="KDE分布曲线")
                # 标记峰值
                if len(features["peaks_x"]) > 0:
                    for peak_x in features["peaks_x"]:
                        ax1.axvline(x=peak_x, color="orange", linestyle="--", linewidth=1.5,
                                    label="峰值" if peak_x == features["peaks_x"][0] else "")
            # 标注均值和中位数
            ax1.axvline(x=features["mean"], color="blue", linestyle="-", linewidth=1.5,
                        label=f"均值: {features['mean']:.3f}")
            ax1.axvline(x=features["median"], color="green", linestyle="-", linewidth=1.5,
                        label=f"中位数: {features['median']:.3f}")

        ax1.set_xlabel("GSS值")
        ax1.set_ylabel("频数/密度")
        ax1.set_title(f"GSS值分布（非零值数量: {n_non_zero}）")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # 子图2：关键分布特征文本标注（方便快速判断）
        text_content = [
            f"非零GSS值数量: {features['n_non_zero']}",
            f"偏度 (Skewness): {features['skewness']:.3f}",
            f"峰度 (Kurtosis): {features['kurtosis']:.3f}",
            f"峰值数量: {features['n_peaks']}",
            f"有高值尾部: {'是' if features['has_high_tail'] else '否'}",
            f"前0.5%高值阈值: {features['top5p']:.3f}"
        ]
        # 文本分行显示
        ax2.text(0.1, 0.9, "\n".join(text_content), transform=ax2.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")  # 隐藏坐标轴
        ax2.set_title("分布特征指标")

        # 展示图像
        plt.tight_layout()
        plt.show()

    return features


def select_best_method(features):
    """
    根据分布特征选择最合适的筛选方法
    返回方法名称及对应的参数
    """
    # 规则1：非零值太少 → 直接用Top-N（最稳健）
    if features["n_non_zero"] < 100:
        return "topn", {"top_n": min(10, features["n_non_zero"])}  # 非零值少，只取前n

    # 规则2：有明显多峰（≥2个峰值）→ KDE法（适合区分多簇分布）
    elif features["n_peaks"] >= 2:
        return "kde", {}  # KDE法默认参数

    # 规则3：高度偏态（偏度>2）且有高值尾部 → Top-N法（适合长尾分布）
    elif features["skewness"] > 2 and features["has_high_tail"]:
        # 偏度越大，取的N越小（避免纳入太多低质量基因）
        topn = 20 if features["skewness"] < 4 else 10
        return "topn", {"top_n": topn}

    # 规则4：近似正态分布（偏度<1.0）→ 均值+标准差法（适合对称分布）
    elif abs(features["skewness"]) < 1.0:
        # 峰度越高（峰值越尖），倍数"multiplier"越低（避免漏筛）
        return "std", {"use_median":False}

    # 规则5：其他情况（如单峰、中等偏态）→ 中位数+四分位距法（抗极端值，适合中等偏态）
    else:
        return "iqr", {"use_median":True}


def dynamic_select_high_specific_genes(mk_score_df, adata):
    """
    动态为每个细胞选择最佳筛选方法，返回高特异性基因
    """
    high_specific_genes_per_cell = {}

    for cell in adata.obs_names:
        gss_values = mk_score_df[cell]
        # 1. 计算当前细胞GSS值的分布特征
        features = get_distribution_features(gss_values, visualize=False)

        # 2. 根据特征选择最佳方法
        method_name, params = select_best_method(features)
        # print(method_name)

        # 3. 调用对应方法筛选基因
        if method_name == "kde":
            genes = get_high_specificity_genes_kde_single_cell(gss_values)
        elif method_name == "iqr":
            genes = get_high_specificity_genes_nd_single_cell(gss_values, **params)
        elif method_name == "std":
            genes = get_high_specificity_genes_nd_single_cell(gss_values, **params)
        elif method_name == "topn":
            genes = get_high_specificity_genes_top_n_single_cell(gss_values, **params)
        else:
            genes = []  # 无匹配方法时返回空

        high_specific_genes_per_cell[cell] = genes
        # 可选：打印日志，查看每个细胞选择的方法
        # print(f"细胞 {cell} 选择方法: {method_name}, 特征: {features}")

    return high_specific_genes_per_cell


def calculate_gene_func(adata, gene_counts,
                        output_dir=None,
                        spatial_top_pct=5,
                        spatial_threshold=0.6,
                        cluster_threshold=0.7
                        ):
    # 获取筛选的阈值
    adaptive_threshold = calculate_adaptive_threshold(gene_counts=gene_counts,
                                                      visualize=True,
                                                      output_dir=output_dir)

    filtered_genes_len = len([key for key, value in gene_counts.items() if value >= adaptive_threshold])
    print(f"占据50%的度的频率阈值是{adaptive_threshold},有{filtered_genes_len}个基因通过筛选（仅供展示）！！！")

    # 初始化选择器
    selector = GssGeneCalculator(
        adata=adata,
        gene_counts=gene_counts,
        spatial_top_pct=spatial_top_pct,  # 定义 “高表达细胞” 的百分比阈值
        spatial_threshold=spatial_threshold,  # 判断 “空间范围受限” 的高表达占比阈值
        cluster_threshold=cluster_threshold  # 判断 “聚集分布” 的聚集指数阈值
    )

    # 筛选具有特定空间表达模式的基因
    selected_genes, results = selector.run_pipeline()

    # 保存筛选结果
    results.to_csv(output_dir + "_selected_genes.csv", index=False, sep='\t')
    print(f"验证结果已保存至 {output_dir}")

    return selected_genes, results


def calculate_adaptive_threshold(gene_counts, visualize=True, output_dir=None):
    """
    基于累积分布手动识别拐点计算阈值，同时可视化基因出现次数的直方图分布和累积分布
    参数:
    - gene_counts: 基因出现次数的字典 {gene: count}
    - visualize: 是否可视化分布（默认True）
    - output_dir: 图像保存目录
    返回:
    - threshold: 计算得到的筛选阈值
    """
    counts = np.array(list(gene_counts.values()))
    sorted_counts = np.sort(counts)[::-1]  # 对次数降序排列
    cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts)  # 计算累积占比
    x = np.arange(len(sorted_counts))  # 构建基因数量（降序）的横轴

    # 手动识别拐点：找累积占比达到 50% 的位置
    target_ratio = 0.5
    knee_idx = np.argmax(cumulative >= target_ratio)  # 找到首个满足累积占比条件的索引

    # 确定阈值
    if knee_idx < len(sorted_counts):
        threshold = sorted_counts[knee_idx]
    else:
        threshold = np.median(counts)  # 若未找到，用中位数兜底

    # 可视化验证（同时展示直方图和累积分布）
    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # 1行2列子图

        # 子图1：gene_counts 直方图分布
        ax1.hist(counts, bins=30, edgecolor='black', color='#66b3ff')
        ax1.axvline(x=threshold, color='red', linestyle='--',
                    label=f'筛选阈值: {threshold:.1f}')  # 标注阈值竖线
        ax1.set_title('基因出现次数直方图分布')
        ax1.set_xlabel('基因出现次数')
        ax1.set_ylabel('基因数量')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 子图2：累积分布曲线
        ax2.plot(x, cumulative, 'b-', linewidth=2)
        ax2.axvline(x=knee_idx, color='red', linestyle='--',
                    label=f'拐点位置: {knee_idx}')  # 标注拐点竖线
        ax2.axhline(y=cumulative[knee_idx], color='green', linestyle='--')
        ax2.set_title(f'基因出现次数累积分布（目标占比 {target_ratio*100:.0f}%）')
        ax2.set_xlabel('基因数量（按出现次数降序排列）')
        ax2.set_ylabel('累积占比')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()  # 调整子图间距

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}gene_counts_distribution.png', dpi=300, bbox_inches='tight')
        print(f"分布图已保存至: {output_dir}gene_counts_distribution.png")


    return threshold


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
            title=f'{gene_name}空间表达分布({visual_indicators})',
            return_fig=True,
            frameon=True,
            show=show,
        )

        fig.savefig(os.path.join(output_dir, f"_{color}_calibrated.png"), dpi=400)
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
    high_specificity_genes_per_cell = dynamic_select_high_specific_genes(mk_score_df, adata)
    gene_nums = {
        cell_type: len(genes)
        for cell_type, genes in high_specificity_genes_per_cell.items()
    }

    # 3. 获取高特异性基因
    all_high_specificity_genes = set()
    for genes in high_specificity_genes_per_cell.values():
        all_high_specificity_genes.update(genes)
    all_high_specificity_genes = list(all_high_specificity_genes)

    # 4. 统计各基因在列表中的频率
    gene_counts = {gene: 0 for gene in all_high_specificity_genes}
    for genes in high_specificity_genes_per_cell.values():
        for gene in genes:
            gene_counts[gene] += 1

    # 5. 基因筛选
    selected_genes, _ = calculate_gene_func(
        adata, gene_counts,
        output_dir=output_dir,
        spatial_top_pct=5,  # 定义 “高表达细胞” 的百分比阈值
        spatial_threshold=0.6,  # 判断 “离散度受限” 的离散阈值
        cluster_threshold=0.6  # 判断 “聚集分布” 的聚集指数阈值
    )
    print(selected_genes)

    # 6. 可视化空间表达模式
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
    species_data = data.get(species, {})
    cancer_types = species_data.get("cancer_types", {})

    # 3. 遍历每个癌症类型
    i = 1
    for cancer_type, sample_ids in cancer_types.items():
        # 4. 遍历该癌症类型下的每个样本
        ids_to_query = sample_ids[: min(n, len(sample_ids))]
        for id in ids_to_query:
            # 构建文件路径
            if i <= select_n:
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
                        default=29,
                        help="至多分析样本")
    parser.add_argument("--json", type=str,
                        default="/Users/wuyang/Documents/MyPaper/3/dataset/cancer_samples.json",
                        help="样本映射JSON路径")
    parser.add_argument("--data-dir", type=str,
                        default="/Users/wuyang/Documents/MyPaper/3/dataset/HEST-data/",
                        help="数据根目录")
    parser.add_argument("--output-root", type=str,
                        default="/Users/wuyang/Documents/MyPaper/3/gsVis/output/HEST",
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