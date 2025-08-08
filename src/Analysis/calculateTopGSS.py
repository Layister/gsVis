import os
import argparse
import pandas as pd
import scanpy as sc
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def set_chinese_font():
    """设置可用的中文字体"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 定义常用中文字体列表
    chinese_fonts = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei", "Arial Unicode MS"]

    # 查找第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            print(f"已设置中文字体: {font}")
            return

    print("警告: 未找到可用的中文字体，中文可能无法正确显示")
    print(f"可用字体示例: {available_fonts[:10]}")


def read_data(feather_path, h5ad_path):
    """读取latent_to_gene.py生成的两个文件"""
    # 读取标记分数文件
    print(f"读取标记分数文件: {feather_path}")
    mk_score_df = pd.read_feather(feather_path)

    # 检查数据结构
    if pd.api.types.is_numeric_dtype(mk_score_df.index):
        print("检测到DataFrame使用数字索引，尝试从第一列获取基因名...")

        # 检查第一列是否包含字符串类型的基因名
        first_column = mk_score_df.iloc[:, 0]
        if pd.api.types.is_string_dtype(first_column):
            # 将第一列设置为索引
            mk_score_df = mk_score_df.set_index(mk_score_df.columns[0])
            print(f"已将第一列 '{mk_score_df.index.name}' 设置为基因名索引")
        else:
            print("警告: 第一列不是字符串类型，无法作为基因名")
            print(f"第一列数据类型: {first_column.dtype}")
            print(f"第一列前5个值: {first_column.head().tolist()}")
            print("继续使用数字索引，但这可能导致后续处理错误")

    # 读取AnnData对象
    print(f"读取AnnData对象: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # 确保基因名称类型一致（转换为字符串）
    print("确保基因名称类型一致...")
    mk_score_df.index = mk_score_df.index.astype(str)
    adata.var_names = adata.var_names.astype(str)

    # 检查数据一致性
    print(f"标记分数DataFrame中的基因数量: {len(mk_score_df.index)}")
    print(f"AnnData对象中的基因数量: {len(adata.var_names)}")

    # 先比较长度
    if len(mk_score_df.index) != len(adata.var_names):
        print("警告: 标记分数DataFrame和AnnData的基因数量不匹配")

        # 计算共同基因
        common_genes = np.intersect1d(mk_score_df.index, adata.var_names)
        print(f"找到 {len(common_genes)} 个共同基因")

        # 计算各自独有的基因
        only_in_mk_score = np.setdiff1d(mk_score_df.index, adata.var_names)
        only_in_adata = np.setdiff1d(adata.var_names, mk_score_df.index)

        print(f"仅存在于标记分数文件中的基因数量: {len(only_in_mk_score)}")
        print(f"仅存在于AnnData文件中的基因数量: {len(only_in_adata)}")

        # 打印一些示例基因
        if len(only_in_mk_score) > 0:
            print(f"仅存在于标记分数文件中的基因示例: {', '.join(only_in_mk_score[:5])}")
        if len(only_in_adata) > 0:
            print(f"仅存在于AnnData文件中的基因示例: {', '.join(only_in_adata[:5])}")

        # 筛选共同基因
        print("筛选共同基因以继续分析...")
        mk_score_df = mk_score_df.loc[common_genes]
        adata = adata[:, common_genes].copy()
    else:
        # 长度相同但内容可能不同
        if not np.all(mk_score_df.index == adata.var_names):
            print("警告: 标记分数DataFrame和AnnData的基因索引顺序不完全匹配")
            # 找到不匹配的位置
            mismatches = np.where(mk_score_df.index != adata.var_names)[0]
            print(f"找到 {len(mismatches)} 个不匹配的基因位置")
            if len(mismatches) > 0:
                print(f"前5个不匹配的位置: {mismatches[:5]}")
                print(
                    f"对应的基因: {mk_score_df.index[mismatches[0]]} (标记分数) vs {adata.var_names[mismatches[0]]} (AnnData)")

            # 重新排序以匹配
            print("重新排序基因以匹配...")
            adata = adata[:, mk_score_df.index].copy()

    print(f"处理后的数据: 基因数量 = {len(mk_score_df.index)}, 细胞数量 = {adata.shape[0]}")
    return mk_score_df, adata


def get_high_specificity_genes_kde(mk_score_df, adata):
    high_specificity_genes_per_cell = {}
    for cell in adata.obs_names:
        gss_values = mk_score_df[cell]
        non_zero_gss = gss_values[gss_values > 0]
        if len(non_zero_gss) > 0:
            # 进行核密度估计
            kde = gaussian_kde(non_zero_gss)
            x = np.linspace(non_zero_gss.min(), non_zero_gss.max(), 1000)
            y = kde(x)
            # 找到峰值
            peaks, _ = find_peaks(y)
            if len(peaks) > 0:
                peak_values = x[peaks]
                # 选择最大的峰值
                main_peak = peak_values[np.argmax(y[peaks])]
                # 自动确定阈值，例如取峰值右侧下降到一定比例的值
                right_of_peak = x[x > main_peak]
                kde_right_of_peak = kde(right_of_peak)
                # 假设下降到峰值的20%作为阈值
                threshold_index = np.argmax(kde_right_of_peak < 0.01 * kde(main_peak))
                if threshold_index > 0:
                    threshold = right_of_peak[threshold_index]
                    top_genes = non_zero_gss[non_zero_gss > threshold].index.tolist()
                    high_specificity_genes_per_cell[cell] = top_genes
                else:
                    high_specificity_genes_per_cell[cell] = []
            else:
                high_specificity_genes_per_cell[cell] = []
        else:
            high_specificity_genes_per_cell[cell] = []
    return high_specificity_genes_per_cell


def get_high_specificity_genes_elbow(mk_score_df, adata):
    high_specificity_genes_per_cell = {}
    for cell in adata.obs_names:
        gss_values = mk_score_df[cell]
        non_zero_gss = gss_values[gss_values > 0].sort_values(ascending=False)
        if len(non_zero_gss) > 0:
            # 计算累积和
            cumulative_sum = np.cumsum(non_zero_gss)
            total_sum = cumulative_sum[-1]
            # 计算每个点的“曲率”
            curvature = []
            window_size = 3  # 调整窗口大小
            for i in range(window_size, len(non_zero_gss) - window_size):
                x0, y0 = i - window_size, cumulative_sum[i - window_size] / total_sum
                x1, y1 = i, cumulative_sum[i] / total_sum
                x2, y2 = i + window_size, cumulative_sum[i + window_size] / total_sum
                curvature.append(abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)))
            # 找到曲率最大的点
            if curvature:
                elbow_index = np.argmax(curvature) + window_size
                top_genes = non_zero_gss.head(elbow_index).index.tolist()
                high_specificity_genes_per_cell[cell] = top_genes
            else:
                high_specificity_genes_per_cell[cell] = []
        else:
            high_specificity_genes_per_cell[cell] = []
    return high_specificity_genes_per_cell


def get_high_specificity_genes_no_zero(mk_score_df, adata, multiplier=4):
    high_specificity_genes_per_cell = {}
    for cell in adata.obs_names:
        gss_values = mk_score_df[cell]
        non_zero_gss = gss_values[gss_values > 0]
        if len(non_zero_gss) > 0:
            mean_gss = non_zero_gss.mean()
            std_gss = non_zero_gss.std()
            threshold = mean_gss + multiplier * std_gss
            top_genes = gss_values[gss_values > threshold].index.tolist()
        else:
            top_genes = []
        high_specificity_genes_per_cell[cell] = top_genes
    return high_specificity_genes_per_cell


def get_high_specificity_genes_top_n(mk_score_df, adata, top_n=10):
    """
    根据GSS值为每个细胞找到高特异性基因列表

    参数:
    - mk_score_df: 标记分数DataFrame (GSS值)
    - adata: AnnData对象
    - top_n: 每个细胞选择的高特异性基因数量

    返回:
    - high_specificity_genes_per_cell: 每个细胞对应的高特异性基因列表的字典
    """
    high_specificity_genes_per_cell = {}

    # 遍历每个细胞
    for cell in adata.obs_names:
        # 获取该细胞对应的GSS值
        gss_values = mk_score_df[cell]

        # 根据GSS值排序，选择前top_n个基因
        top_genes = gss_values.sort_values(ascending=False).head(top_n).index.tolist()

        high_specificity_genes_per_cell[cell] = top_genes

    return high_specificity_genes_per_cell


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

        plt.show()

    return max(10, threshold)  # 设置最小阈值，避免筛选过少基因


# def plot_gene_spatial(mk_score_df, adata, gene_name, high_specificity_genes_per_cell,
#                       output_dir=None, visual_indicators=None,
#                       cmap='viridis', size=0.8, alpha=0.8,
#                       background_alpha=0.7, show=True):
#     """使用 Scanpy 内置方法可视化特定基因表达并叠加病理切片"""
#     # 基因存在性检查
#     if gene_name not in adata.var_names:
#         print(f"错误: 基因 '{gene_name}' 不在数据中")
#         return
#
#     # 检查空间坐标
#     if 'spatial' not in adata.obsm:
#         print("错误: 缺少空间坐标信息")
#         return
#
#     # 创建新的AnnData对象，只包含高特异性基因列表中包含该基因的细胞
#     cells_with_gene = [cell for cell, genes in high_specificity_genes_per_cell.items() if gene_name in genes]
#     adata_subset = adata[cells_with_gene].copy()
#
#     if len(adata_subset) == 0:
#         print(f"没有细胞的高特异性基因列表中包含基因 '{gene_name}'")
#         return
#
#     # 获取基因表达值
#     if visual_indicators == "GSS" and gene_name in mk_score_df.index:
#         print(f"基因 '{gene_name}' 使用GSS分数数据")
#         obs_column_name = f"Marker_score_{gene_name}"
#         adata_subset.obs[obs_column_name] = mk_score_df.loc[gene_name, cells_with_gene].values
#         color = obs_column_name
#     else:
#         print(f"基因 '{gene_name}' 使用原始基因表达值")
#         color = gene_name  # 直接使用 adata.var_names 中的基因
#
#     # 创建输出目录
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#
#     # 使用 Scanpy 的空间可视化函数
#     try:
#         # 设置 Scanpy 保存路径
#         if output_dir:
#             sc.settings.figdir = output_dir
#
#         sc.pl.spatial(
#             adata_subset,
#             color=color,
#             cmap=cmap,
#             size=size,
#             alpha=alpha,
#             img_key='hires',
#             alpha_img=background_alpha,
#             show=show,
#             save=f"{color}_calibrated.png",
#             frameon=False,
#             title=f'{gene_name}空间表达分布({visual_indicators})'
#         )
#
#     except Exception as e:
#         print(f"可视化失败: {e}")


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

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

    # 使用 Scanpy 的空间可视化函数
    try:
        # 设置 Scanpy 保存路径
        if output_dir:
            sc.settings.figdir = output_dir

        sc.pl.spatial(
            adata_subset,
            color=color,
            cmap=cmap,
            size=size,
            alpha=alpha,
            img_key='hires',
            alpha_img=background_alpha,
            show=show,
            save=f"{color}_calibrated.png",
            frameon=False,
            title=f'{gene_name}空间表达分布({visual_indicators})'
        )

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


def main():
    # 地址
    work_dir = "/Users/wuyang/Documents/MyPaper/3/gsVis/data"
    sample_id = "BRCA"
    sample_name = "Human_Breast_Cancer"

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='可视化基因在特定spot上的表达分布')
    parser.add_argument('--feather-path',
                        default=f'{work_dir}/{sample_id}/{sample_name}/latent_to_gene/{sample_name}_gene_marker_score.feather',
                        help='标记分数feather文件路径')
    parser.add_argument('--h5ad-path',
                        default=f'{work_dir}/{sample_id}/{sample_name}/find_latent_representations/{sample_name}_add_latent.h5ad',
                        help='AnnData h5ad文件路径')
    parser.add_argument('--top-n', type=int, default=10, help='每个细胞选择的高特异性基因数量')
    parser.add_argument('--output-dir',
                        default=f'/Users/wuyang/Documents/MyPaper/3/gsVis/output/{sample_id}/calculateTopGSS/',
                        help='图像保存目录')
    parser.add_argument('--cmap', default='viridis', help='颜色映射方案 (默认: viridis)')
    parser.add_argument('--size', type=float, default=1.0, help='点大小 (默认: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.6, help='透明度 (默认: 0.7)')
    parser.add_argument('--min-count', type=int, default=100, help='基因在高特异性基因列表中出现的最小次数')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置中文字体
    set_chinese_font()

    # 读取数据
    mk_score_df, adata = read_data(args.feather_path, args.h5ad_path)

    # 筛选取出各细胞GSS值的top n基因
    # high_specificity_genes_per_cell = get_high_specificity_genes_top_n(mk_score_df, adata, top_n=args.top_n)
    # 使用非零值的均值和标准差作为阈值
    high_specificity_genes_per_cell = get_high_specificity_genes_no_zero(mk_score_df, adata, multiplier=4)

    # 例如使用肘部法则方法
    # high_specificity_genes_per_cell = get_high_specificity_genes_elbow(mk_score_df, adata)
    # 例如使用核密度估计检测
    # high_specificity_genes_per_cell = get_high_specificity_genes_kde(mk_score_df, adata)


    # 获取所有高特异性基因
    all_high_specificity_genes = set()
    for genes in high_specificity_genes_per_cell.values():
        all_high_specificity_genes.update(genes)
    all_high_specificity_genes = list(all_high_specificity_genes)

    # 统计每个基因在高特异性基因列表中出现的次数
    gene_counts = {gene: 0 for gene in all_high_specificity_genes}
    for genes in high_specificity_genes_per_cell.values():
        for gene in genes:
            gene_counts[gene] += 1

    # 筛选出现次数大于等于阈值的基因
    min_count = calculate_adaptive_threshold(gene_counts=gene_counts, visualize=True, output_dir=args.output_dir)
    selected_genes = [gene for gene, count in gene_counts.items() if count >= min_count]

    # 可视化这些基因的空间表达模式
    plot_multiple_genes(
        mk_score_df, adata, selected_genes,
        high_specificity_genes_per_cell,
        output_dir=args.output_dir,
        visual_indicators="Expr", # ["GSS", "Expr"]
        cmap=args.cmap,
        size=args.size,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()