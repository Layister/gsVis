import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import argparse
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_gene_samples(gene_freq_path, target_gene):
    """从gene_frequency.csv加载目标基因出现的癌症类型和样本信息"""
    gene_freq_df = pd.read_csv(gene_freq_path, sep='\t')
    target_df = gene_freq_df[gene_freq_df['gene'] == target_gene].copy()

    if target_df.empty:
        raise ValueError(f"目标基因 {target_gene} 未在gene_frequency.csv中找到")

    samples_info = defaultdict(list)
    keys_str = target_df.iloc[0]['keys']
    key_entries = [k.strip() for k in keys_str.split(',')]

    for entry in key_entries:
        if '_' in entry:
            cancer_type, sample_id = entry.split('_', 1)
            samples_info[cancer_type].append(sample_id)
        else:
            print(f"警告: 无法解析条目格式 {entry}，跳过该样本")

    return samples_info


def load_spatial_data(sample_dir, cancer_type, sample_id):
    """加载单个样本的空间转录组数据（h5ad格式）"""
    # 尝试可能的文件路径格式
    path_formats = [
        os.path.join(sample_dir, cancer_type, f"{sample_id}_adata.h5ad"),
        os.path.join(sample_dir, f"{cancer_type}_{sample_id}.h5ad")
    ]

    for sample_path in path_formats:
        if os.path.exists(sample_path):
            return sc.read_h5ad(sample_path)

    raise FileNotFoundError(
        f"样本 {cancer_type}_{sample_id} 的数据文件不存在，尝试路径: {path_formats}")


def get_expression_info(adata, target_gene):
    """提取基因表达数据和坐标范围的通用函数"""
    if 'spatial' not in adata.obsm:
        return None, None, None, None, None

    coords = adata.obsm['spatial']
    expression = adata[:, target_gene].X

    # 处理不同类型的表达量数据
    if hasattr(expression, 'toarray'):
        expression = expression.toarray().flatten()
    else:
        expression = np.array(expression).flatten()

    # 计算表达区域坐标范围
    expr_indices = np.where(expression >= 0)[0]
    if len(expr_indices) == 0:
        return coords, expression, None, None, None, None

    expr_coords = coords[expr_indices]
    x_min, x_max = expr_coords[:, 0].min(), expr_coords[:, 0].max()
    y_min, y_max = expr_coords[:, 1].min(), expr_coords[:, 1].max()

    # 添加边距
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    return coords, expression, x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin


def plot_original_tissue(ax, adata, expr_x_min, expr_x_max, expr_y_min, expr_y_max):
    """绘制原始组织切片的通用函数"""
    if 'spatial' not in adata.uns:
        return False, 1.0

    spatial_data = adata.uns['spatial'].get('ST', {})
    scale_factor = 1.0
    img_keys = ['hires', 'lowres', 'downscaled_fullres', 'original']

    # 尝试不同的图像键
    for img_key in img_keys:
        if 'images' in spatial_data and img_key in spatial_data['images']:
            img = spatial_data['images'][img_key]
            scale_key = f'tissue_{img_key}_scalef'
            scale_factor = spatial_data['scalefactors'].get(scale_key, 1.0)

            ax.imshow(img)
            ax.set_title(f"tissue section ({img_key})")
            return True, scale_factor

    # 尝试其他图像格式
    for img_type in ['tissue_hires_image', 'tissue_lowres_image']:
        if img_type in spatial_data:
            img = spatial_data[img_type]
            scale_key = f'{img_type.split("_")[1]}_scalef'
            scale_factor = spatial_data['scalefactors'].get(scale_key, 1.0)

            ax.imshow(img)
            ax.set_title(f"组织切片 ({img_type.split('_')[1]})")
            return True, scale_factor

    return False, 1.0


def visualize_single_sample(target_gene, cancer_type, sample_id, adata, output_dir):
    """为单个样本生成可视化图片，包含切片和表达图"""
    # 创建输出目录
    sample_output_dir = os.path.join(output_dir, target_gene)
    os.makedirs(sample_output_dir, exist_ok=True)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 获取表达信息
    coords, expression, expr_x_min, expr_x_max, expr_y_min, expr_y_max = get_expression_info(adata, target_gene)

    # 绘制组织切片
    has_original, scale_factor = plot_original_tissue(ax1, adata, expr_x_min, expr_x_max, expr_y_min, expr_y_max)

    # 设置组织切片显示范围
    if all(v is not None for v in [expr_x_min, expr_x_max, expr_y_min, expr_y_max]):
        img_x_min, img_x_max = expr_x_min * scale_factor, expr_x_max * scale_factor
        img_y_min, img_y_max = expr_y_min * scale_factor, expr_y_max * scale_factor
        ax1.set_xlim(img_x_min, img_x_max)
        ax1.set_ylim(img_y_max, img_y_min)  # 注意y轴方向

    ax1.axis('off')

    # 绘制基因表达分布
    try:
        if coords is None:
            ax2.text(0.5, 0.5, "无空间坐标数据", ha='center', va='center', transform=ax2.transAxes)
        else:
            # 标准化表达量
            if expression.max() > expression.min():
                expression_norm = (expression - expression.min()) / (expression.max() - expression.min())
            else:
                expression_norm = np.zeros_like(expression)

            # 绘制散点图
            scaled_coords = coords * scale_factor
            scatter = ax2.scatter(
                scaled_coords[:, 0],
                -scaled_coords[:, 1],  # 上下翻转y坐标
                c=expression_norm,
                cmap='viridis',
                s=10,
                alpha=0.8
            )

            # 设置表达图显示范围
            if all(v is not None for v in [expr_x_min, expr_x_max, expr_y_min, expr_y_max]):
                ax2.set_xlim(expr_x_min * scale_factor, expr_x_max * scale_factor)
                ax2.set_ylim(-expr_y_max * scale_factor, -expr_y_min * scale_factor)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax2, orientation='vertical', shrink=0.8)
            cbar.set_label('Relative Expression')

        ax2.set_title(f"{target_gene} expression distribution")
        ax2.set_aspect('equal')
        ax2.axis('off')

    except Exception as e:
        ax2.text(0.5, 0.5, f"绘制表达图出错: {str(e)}", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f"{target_gene} 表达分布")
        ax2.axis('off')

    # 保存图像
    fig.suptitle(f"{cancer_type} - {sample_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(sample_output_dir, f"{cancer_type}_{sample_id}_{target_gene}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"样本 {cancer_type}_{sample_id} 可视化结果已保存至: {output_path}")
    return output_path


def visualize_gene_spatial_expression(target_gene, gene_freq_path, sample_dir, output_dir):
    """可视化目标基因在多个癌症样本中的空间表达分布"""
    print(f"加载基因 {target_gene} 的样本信息...")
    samples_info = load_gene_samples(gene_freq_path, target_gene)

    if not samples_info:
        print(f"未找到基因 {target_gene} 的任何样本信息")
        return

    print(f"开始绘制 {target_gene} 的空间表达图...")
    for cancer_type, sample_ids in samples_info.items():
        for sample_id in sample_ids:
            adata = load_spatial_data(sample_dir, cancer_type, sample_id)

            if target_gene not in adata.var_names:
                print(f"警告: 基因 {target_gene} 不在样本 {cancer_type}_{sample_id} 中，跳过该样本")
                continue

            visualize_single_sample(target_gene, cancer_type, sample_id, adata, output_dir)

    print(f"所有可用样本的可视化已完成，结果保存在: {os.path.join(output_dir, target_gene)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化特定基因在多个癌症样本中的空间表达分布')
    parser.add_argument('--gene-freq',
                        default='../output/HEST2/Homo sapiens/gene_frequency.csv',
                        help='gene_frequency.csv文件路径')
    parser.add_argument('--sample-dir',
                        default='/home/wuyang/hest-data/process/Homo sapiens',
                        help='存储样本h5ad文件的根目录')
    parser.add_argument('--output-dir', default='./Plot/gene_spatial_plots', help='输出可视化结果的目录')

    args = parser.parse_args()

    # 手动展示基因
    vis_genes = ['DPT']
    # 共现网络高频率基因
    #vis_genes = ['IGLC2', 'IGHG3', 'IGKC', 'IGHA1', 'HBA2', 'IGLC3', 'HBB', 'IGHG2', 'HBA1']
    # upset图高共有基因
    #vis_genes = ['KRT17', 'LRRC15', 'MFAP5', 'MS4A1', 'SFRP2']
    # 特定癌症独有基因
    #        癌症['COAD',    'COADREAD', 'EPM',    'HCC',    'IDC',   'ILC', 'PAAD', 'PRAD', 'READ', 'SCCRCC', 'SKCM']
    #vis_genes = ['IGKV1D-12', 'MCEMP1', 'CHI3L2', 'CYP2E1', 'KRT14', 'PIM1', 'GCG', 'TGM4', 'ITLN1', 'FABP7', 'MAL2']

    for gene in vis_genes:
        visualize_gene_spatial_expression(
            target_gene=gene,
            gene_freq_path=args.gene_freq,
            sample_dir=args.sample_dir,
            output_dir=args.output_dir
        )