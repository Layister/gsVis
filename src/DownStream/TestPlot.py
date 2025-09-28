import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import argparse

# 设置中文字体支持
plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


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
        return None, None, None, None, None, None

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
            return True, scale_factor

    # 尝试其他图像格式
    for img_type in ['tissue_hires_image', 'tissue_lowres_image']:
        if img_type in spatial_data:
            img = spatial_data[img_type]
            scale_key = f'{img_type.split("_")[1]}_scalef'
            scale_factor = spatial_data['scalefactors'].get(scale_key, 1.0)

            ax.imshow(img)
            return True, scale_factor

    return False, 1.0


def save_tissue_image(target_gene, cancer_type, sample_id, adata, output_dir):
    """保存组织切片图像，直接裁剪感兴趣区域"""
    # 创建输出目录
    tissue_dir = os.path.join(output_dir, target_gene, "tissue")
    os.makedirs(tissue_dir, exist_ok=True)

    # 获取表达信息
    coords, expression, expr_x_min, expr_x_max, expr_y_min, expr_y_max = get_expression_info(adata, target_gene)

    # 检查是否成功获取表达区域范围
    if any(v is None for v in [expr_x_min, expr_x_max, expr_y_min, expr_y_max]):
        print(f"警告: 无法确定基因 {target_gene} 的表达区域，使用全图")
        # 回退到设置显示范围的方法
        return save_tissue_image_fallback(target_gene, cancer_type, sample_id, adata, output_dir)

    # 尝试获取原始图像数据
    spatial_data = adata.uns['spatial']['ST']
    img_data = None
    scale_factor = 1.0

    # 尝试不同的图像键 - 使用更全面的方法
    img_keys_to_try = [
        ('images', 'hires'),
        ('images', 'lowres'),
        ('images', 'downscaled_fullres'),
        ('tissue_hires_image', None),
        ('tissue_lowres_image', None)
    ]

    for key1, key2 in img_keys_to_try:
        if key2:
            if key1 in spatial_data and key2 in spatial_data[key1]:
                img_data = spatial_data[key1][key2]
                scale_key = f'tissue_{key2}_scalef'
                if 'scalefactors' in spatial_data and scale_key in spatial_data['scalefactors']:
                    scale_factor = spatial_data['scalefactors'][scale_key]
                break
        else:
            if key1 in spatial_data:
                img_data = spatial_data[key1]
                # 尝试获取对应的缩放因子
                scale_key = f'tissue_{key1.split("_")[1]}_scalef'
                if 'scalefactors' in spatial_data and scale_key in spatial_data['scalefactors']:
                    scale_factor = spatial_data['scalefactors'][scale_key]
                break

    # 如果没有找到图像数据，回退到设置显示范围的方法
    if img_data is None:
        print("警告: 未找到组织图像数据，使用回退方法")
        return save_tissue_image_fallback(target_gene, cancer_type, sample_id, adata, output_dir)

    # 获取图像尺寸
    img_height, img_width = img_data.shape[:2]

    # 调试信息
    print(f"图像尺寸: {img_width}x{img_height}")
    print(f"缩放因子: {scale_factor}")
    print(f"表达区域空间坐标: x[{expr_x_min}, {expr_x_max}], y[{expr_y_min}, {expr_y_max}]")

    # 将空间坐标转换为图像像素坐标
    # 注意：空间坐标的原点通常在左下角，而图像坐标的原点在左上角
    # 所以需要翻转Y轴
    x_min_px = int(expr_x_min * scale_factor)
    x_max_px = int(expr_x_max * scale_factor)

    # Y坐标转换：空间坐标 -> 图像坐标
    # 空间坐标的y_max对应图像底部的y_min，空间坐标的y_min对应图像顶部的y_max
    # y_min_px = int((img_height / scale_factor - expr_y_max) * scale_factor)
    # y_max_px = int((img_height / scale_factor - expr_y_min) * scale_factor)

    y_min_px = int(expr_y_min * scale_factor)  # 原来的下边框
    y_max_px = int(expr_y_max * scale_factor)  # 原来的上边框


    # 确保裁剪区域在图像范围内
    x_min_px = max(0, x_min_px)
    x_max_px = min(img_width, x_max_px)
    y_min_px = max(0, y_min_px)
    y_max_px = min(img_height, y_max_px)

    # 检查裁剪区域是否有效
    if x_min_px >= x_max_px or y_min_px >= y_max_px:
        print(f"警告: 裁剪区域无效 ({x_min_px},{y_min_px})-({x_max_px},{y_max_px})，使用回退方法")
        return save_tissue_image_fallback(target_gene, cancer_type, sample_id, adata, output_dir)

    print(f"裁剪区域: x[{x_min_px}, {x_max_px}], y[{y_min_px}, {y_max_px}]")

    # 裁剪图像
    cropped_img = img_data[y_min_px:y_max_px, x_min_px:x_max_px]

    # 保存裁剪后的图像
    output_path = os.path.join(tissue_dir, f"{cancer_type}_{sample_id}_tissue.png")

    # 使用matplotlib保存图像
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cropped_img)
    ax.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 同时保存一份带有点位标记的图像用于验证
    validation_path = os.path.join(tissue_dir, f"{cancer_type}_{sample_id}_tissue_validation.png")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_data)

    # 在原图上标记裁剪区域
    rect = plt.Rectangle((x_min_px, y_min_px),
                         x_max_px - x_min_px,
                         y_max_px - y_min_px,
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # 标记表达点
    if coords is not None:
        # 转换所有坐标点
        spot_x = coords[:, 0] * scale_factor
        spot_y = coords[:, 1] * scale_factor
        ax.scatter(spot_x, spot_y, s=5, c='blue', alpha=0.5)

    plt.savefig(validation_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"验证图像已保存至: {validation_path}")

    return output_path


def save_tissue_image_fallback(target_gene, cancer_type, sample_id, adata, output_dir):
    """回退方法：使用设置显示范围的方式"""
    # 创建输出目录
    tissue_dir = os.path.join(output_dir, target_gene, "tissue")
    os.makedirs(tissue_dir, exist_ok=True)

    # 创建独立画布
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 获取表达信息
    _, _, expr_x_min, expr_x_max, expr_y_min, expr_y_max = get_expression_info(adata, target_gene)

    # 绘制组织切片
    has_original, scale_factor = plot_original_tissue(ax, adata, expr_x_min, expr_x_max, expr_y_min, expr_y_max)

    # 如果没有原始组织切片，显示空间坐标分布
    if not has_original:
        coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None
        if coords is not None:
            ax.scatter(coords[:, 0], coords[:, 1], s=1, c='gray')
        ax.set_aspect('equal')

    # 设置组织切片显示范围
    if all(v is not None for v in [expr_x_min, expr_x_max, expr_y_min, expr_y_max]):
        img_x_min, img_x_max = expr_x_min * scale_factor, expr_x_max * scale_factor
        img_y_min, img_y_max = expr_y_min * scale_factor, expr_y_max * scale_factor
        ax.set_xlim(img_x_min, img_x_max)
        ax.set_ylim(img_y_max, img_y_min)  # 注意y轴方向

    ax.axis('off')  # 关闭坐标轴

    # 保存图像，不添加任何标题
    output_path = os.path.join(tissue_dir, f"{cancer_type}_{sample_id}_tissue.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_path


def save_expression_image(target_gene, cancer_type, sample_id, adata, output_dir):
    """保存基因表达分布图（无标题）"""
    # 创建输出目录
    expr_dir = os.path.join(output_dir, target_gene, "expression")
    os.makedirs(expr_dir, exist_ok=True)

    # 创建独立画布，不使用子图
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)  # 直接创建单个轴，不是子图关系

    # 获取表达信息
    coords, expression, expr_x_min, expr_x_max, expr_y_min, expr_y_max = get_expression_info(adata, target_gene)

    try:
        if coords is None:
            ax.text(0.5, 0.5, "无空间坐标数据", ha='center', va='center', transform=ax.transAxes)
        else:
            # 获取缩放因子
            spatial_data = adata.uns['spatial'].get('ST', {})
            scale_factor = 1.0
            img_keys = ['hires', 'lowres', 'downscaled_fullres', 'original']

            for img_key in img_keys:
                if 'images' in spatial_data and img_key in spatial_data['images']:
                    scale_key = f'tissue_{img_key}_scalef'
                    scale_factor = spatial_data['scalefactors'].get(scale_key, 1.0)
                    break

            # 标准化表达量
            if expression.max() > expression.min():
                expression_norm = (expression - expression.min()) / (expression.max() - expression.min())
            else:
                expression_norm = np.zeros_like(expression)

            # 绘制散点图
            scaled_coords = coords * scale_factor
            scatter = ax.scatter(
                scaled_coords[:, 0],
                -scaled_coords[:, 1],  # 上下翻转y坐标
                c=expression_norm,
                cmap='viridis',
                s=10,
                alpha=0.8
            )

            # 设置表达图显示范围
            if all(v is not None for v in [expr_x_min, expr_x_max, expr_y_min, expr_y_max]):
                ax.set_xlim(expr_x_min * scale_factor, expr_x_max * scale_factor)
                ax.set_ylim(-expr_y_max * scale_factor, -expr_y_min * scale_factor)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
            cbar.set_label('相对表达量')

        ax.set_aspect('equal')
        ax.axis('off')  # 关闭坐标轴

    except Exception as e:
        ax.text(0.5, 0.5, f"绘制表达图出错: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')  # 关闭坐标轴

    # 保存图像，不添加任何标题
    output_path = os.path.join(expr_dir, f"{cancer_type}_{sample_id}_{target_gene}_expression.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭当前画布，确保与其他图完全独立

    return output_path


def visualize_specific_sample(target_gene, cancer_type, sample_id, sample_dir, output_dir):
    """可视化指定的样本和基因，生成完全独立的图像，无标题"""
    try:
        # 加载数据
        adata = load_spatial_data(sample_dir, cancer_type, sample_id)

        # 检查基因是否存在
        if target_gene not in adata.var_names:
            print(f"警告: 基因 {target_gene} 不在样本 {cancer_type}_{sample_id} 中")
            return None, None

        # 保存组织切片图（独立图像）
        tissue_path = save_tissue_image(target_gene, cancer_type, sample_id, adata, output_dir)
        # 保存基因表达分布图（独立图像）
        expr_path = save_expression_image(target_gene, cancer_type, sample_id, adata, output_dir)

        print(f"样本 {cancer_type}_{sample_id} 组织图已保存至: {tissue_path}")
        print(f"样本 {cancer_type}_{sample_id} 表达图已保存至: {expr_path}")
        return tissue_path, expr_path

    except Exception as e:
        print(f"处理样本 {cancer_type}_{sample_id} 时出错: {str(e)}")
        return None, None



def main():
    parser = argparse.ArgumentParser(description='可视化指定癌症样本中特定基因的空间表达分布')
    parser.add_argument('--sample-dir',
                        default='/Users/wuyang/Documents/MyPaper/3/dataset/HEST-data/Homo sapiens',
                        help='存储样本h5ad文件的根目录')
    parser.add_argument('--output-dir', default='./test',
                        help='输出可视化结果的目录')
    parser.add_argument('--gene', default='COL1A1', help='要可视化的基因名称')
    parser.add_argument('--cancer-type', default='COAD', help='癌症类型')
    parser.add_argument('--sample-id', default='TENX155', help='样本ID')

    args = parser.parse_args()

    # 可视化指定的样本和基因
    visualize_specific_sample(
        target_gene=args.gene,
        cancer_type=args.cancer_type,
        sample_id=args.sample_id,
        sample_dir=args.sample_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
