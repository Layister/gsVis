import os
import numpy as np
import pandas as pd
import cv2
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage import io, morphology, measure
from skimage.filters import gaussian
from skimage.draw import polygon
from scipy.spatial import KDTree
from scipy.ndimage import rotate
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import scanpy as sc

# 忽略绘图警告
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

# 设置中文字体支持
plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class SpatialColocalizationAnalyzer:
    """空间转录组基因表达与ROI共定位分析工具"""

    def __init__(self, output_dir="analysis_results"):
        """初始化分析器

        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, "gene_maps"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "manual_rois"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "colocalization"), exist_ok=True)

        # 存储分析数据
        self.adata = None
        self.coordinates = None
        self.expression_data = None
        self.histology_image = None
        self.roi_mask = None
        self.results = {}

    def load_h5ad_data(self, h5ad_path):
        """加载h5ad格式的空间转录组数据"""
        print(f"加载空间转录组数据: {h5ad_path}")
        self.adata = sc.read(h5ad_path)

        # 提取空间坐标
        self.coordinates = self.adata.obsm['spatial']
        print(f"提取空间坐标，共{len(self.coordinates)}个点")

        # 提取基因表达数据
        self.expression_data = self.adata.to_df()
        print(f"提取基因表达数据，共{self.expression_data.shape[1]}个基因")

        # 提取组织学图像和缩放因子和spot直径
        self.scale_factor = None
        self.spot_diameter = None  # 存储spot直径（像素）
        self.histology_image = None

        try:
            # 适配10x Genomics标准格式
            spatial_key = next(iter(self.adata.uns['spatial'].keys()))  # 通常是"ST"
            spatial_data = self.adata.uns['spatial'][spatial_key]

            # 提取缩放因子
            scalefactors = spatial_data['scalefactors']
            if 'tissue_downscaled_fullres_scalef' in scalefactors:
                self.scale_factor = scalefactors['tissue_downscaled_fullres_scalef']
                self.coord_type = "降采样全分辨率"
            elif 'tissue_hires_scalef' in scalefactors:
                self.scale_factor = scalefactors['tissue_hires_scalef']
                self.coord_type = '高分辨率'
            elif 'tissue_lowres_scalef' in scalefactors:
                self.scale_factor = scalefactors['tissue_lowres_scalef']
                self.coord_type = '低分辨率'

            # 提取spot直径
            if 'spot_diameter_fullres' in scalefactors:
                self.spot_diameter = scalefactors['spot_diameter_fullres']
                print(f"提取到spot直径: {self.spot_diameter} 像素（全分辨率）")

            # 提取组织学图像
            for img_key in spatial_data['images']:
                self.histology_image = spatial_data['images'][img_key]
                break

        except Exception as e:
            print(f"提取空间信息警告: {str(e)}")


        if self.histology_image is not None:
            print(f"提取组织学图像，尺寸: {self.histology_image.shape}")
            if len(self.histology_image.shape) == 2:
                self.histology_image = np.stack([self.histology_image] * 3, axis=-1)
        else:
            print("警告: 未找到组织学图像")

        return self

    def load_manual_roi(self, roi_path, black_line_threshold=50,
                        min_contour_area=50, min_area=50):
        """加载ROI图像并修复轮廓检测问题"""

        # 1. 读取图像并强制处理通道
        roi_image = io.imread(roi_path)
        print(f"原始ROI图像信息：形状{roi_image.shape}，数据类型{roi_image.dtype}")

        # 2. 转换为灰度图
        if len(roi_image.shape) == 3:
            if roi_image.shape[2] == 4:  # RGBA
                gray_image = cv2.cvtColor(roi_image, cv2.COLOR_RGBA2GRAY)
            elif roi_image.shape[2] == 3:  # RGB或BGR
                # 检测是否为BGR格式（OpenCV默认）
                if np.mean(roi_image[:, :, 0]) > np.mean(roi_image[:, :, 2]):
                    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"不支持的通道数: {roi_image.shape[2]}")
        else:  # 单通道
            gray_image = roi_image.copy()

        # 打印灰度值范围（关键！确定黑色线条的实际灰度值）
        print(f"灰度图像素值范围：最小值{gray_image.min()}, 最大值{gray_image.max()}, 平均值{gray_image.mean()}")
        # 保存灰度图
        gray_path = os.path.join(self.output_dir, "manual_rois", "gray_roi_image.png")
        io.imsave(gray_path, gray_image)
        print(f"灰度图已保存至: {gray_path}")

        # 3. 提取黑色线条（动态调整阈值逻辑）
        # 如果最小值接近0（存在纯黑），强制降低阈值
        if gray_image.min() < 5:
            adjusted_threshold = min(black_line_threshold, 20)  # 确保捕获纯黑
            print(f"检测到纯黑区域，自动调整阈值为{adjusted_threshold}")
        else:
            adjusted_threshold = black_line_threshold

        # 二值化：黑色线条（低灰度）设为1，其他为0
        black_lines = (gray_image <= adjusted_threshold).astype(np.uint8)
        # 保存二值化结果（必须看到白色线条）
        lines_path = os.path.join(self.output_dir, "manual_rois", "black_lines_debug.png")
        io.imsave(lines_path, black_lines * 255)  # 乘255转为可视的白色
        print(f"黑色线条二值图已保存至: {lines_path}")

        # 4. 强化线条（解决线条断裂/过细问题）
        if np.sum(black_lines) == 0:
            print("警告：未检测到任何黑色线条！可能阈值设置过高")
            # 尝试极端阈值（强制捕获可能的线条）
            black_lines = (gray_image <= 100).astype(np.uint8)
            print(f"尝试使用阈值100重新提取线条，像素数：{np.sum(black_lines)}")
        else:
            print(f"提取到黑色线条，像素数：{np.sum(black_lines)}")

        # 形态学处理：加粗线条+闭合间隙（关键修复）
        kernel = np.ones((5, 5), np.uint8)  # 更大的内核处理细线条
        black_lines = cv2.dilate(black_lines, kernel, iterations=1)  # 加粗线条
        black_lines = cv2.erode(black_lines, kernel, iterations=1)  # 保持形状同时去噪
        black_lines = cv2.morphologyEx(black_lines, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭合间隙

        # 5. 轮廓检测（增加容错逻辑）
        contours = measure.find_contours(black_lines, level=0.5)
        valid_contours = []
        for contour in contours:
            # 计算轮廓面积（兼容不同格式）
            try:
                area = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32))
            except:
                area = 0
            if area >= min_contour_area:
                valid_contours.append(contour)

        # 调试：输出轮廓统计
        print(f"原始轮廓数：{len(contours)}，有效轮廓数（面积达标）：{len(valid_contours)}")

        # 只保留面积最大的轮廓（关键修改）
        if len(valid_contours) > 1:
            print("检测到多个轮廓，只保留面积最大的轮廓")
            contour_areas = []
            for contour in valid_contours:
                area = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32))
                contour_areas.append(area)

            max_area_index = np.argmax(contour_areas)
            valid_contours = [valid_contours[max_area_index]]
            print(f"最大轮廓面积: {contour_areas[max_area_index]}")

        if len(valid_contours) == 0:
            # 尝试降低标准再检测一次
            print(f"尝试降低标准（min_contour_area={min_contour_area // 2}）重新检测...")
            for contour in contours:
                area = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32)) if len(contours) > 0 else 0
                if area >= (min_contour_area // 2):
                    valid_contours.append(contour)
            print(f"降低标准后有效轮廓数：{len(valid_contours)}")

        # 6. 填充轮廓（确保坐标有效）
        roi_mask = np.zeros_like(gray_image, dtype=np.uint8)
        if len(valid_contours) > 0:
            for i, contour in enumerate(valid_contours):
                # 转换坐标并裁剪
                contour = np.round(contour).astype(int)
                y_coords, x_coords = contour[:, 0], contour[:, 1]  # y=行，x=列
                y_coords = np.clip(y_coords, 0, gray_image.shape[0] - 1)
                x_coords = np.clip(x_coords, 0, gray_image.shape[1] - 1)

                # 填充
                rows, cols = polygon(y_coords, x_coords)
                roi_mask[rows, cols] = 1
                print(f"填充轮廓{i + 1}：{len(rows)}个像素")
        else:
            print("错误：未检测到任何有效轮廓，无法生成ROI掩码")

        # 7. 后处理与保存
        if min_area > 0 and np.sum(roi_mask) > 0:
            roi_mask = morphology.remove_small_objects(roi_mask.astype(bool), min_size=min_area).astype(np.uint8)

        # 强制保存掩码（即使全黑也保存以便调试）
        mask_path = os.path.join(self.output_dir, "manual_rois", "preprocessed_roi_mask.png")
        io.imsave(mask_path, roi_mask * 255)
        print(f"ROI掩码已保存至: {mask_path}，有效像素数：{np.sum(roi_mask)}")

        self.roi_mask = roi_mask
        return self

    def visualize_manual_roi(self, output_name="roi_overlay.png"):
        """将ROI叠加到组织学图像上可视化

        Args:
            output_name: 输出图像文件名
        """
        if self.histology_image is None:
            print("警告: 未加载组织学图像，无法可视化ROI叠加效果")
            return

        if self.roi_mask is None:
            print("警告: 未加载ROI掩码，请先调用load_manual_roi")
            return

        # 调整ROI掩码尺寸以匹配组织学图像
        roi_resized = cv2.resize(
            self.roi_mask,
            (self.histology_image.shape[1], self.histology_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # 创建叠加图像
        overlay = self.histology_image.copy()
        overlay[roi_resized == 1] = [255, 0, 0]  # 红色标记ROI区域

        # 绘制ROI轮廓
        contours = measure.find_contours(roi_resized, level=0.5)
        plt.figure(figsize=(10, 10))
        plt.imshow(self.histology_image)
        plt.imshow(overlay, alpha=0.3)  # 半透明叠加

        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)

        plt.axis('off')

        output_path = os.path.join(self.output_dir, "manual_rois", output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROI叠加图像已保存至: {output_path}")
        return self

    def create_gene_expression_maps(self, genes, image_size=None,
                                    interpolation_method='gaussian', sigma=2):
        """创建基因表达热图（使用scale_factor）

        Args:
            genes: 基因列表
            image_size: 热图尺寸，默认为组织学图像尺寸
            interpolation_method: 插值方法，'gaussian'或'idw'
            sigma: 高斯平滑的sigma参数
        """
        if self.coordinates is None or self.expression_data is None:
            raise ValueError("请先加载空间转录组数据（调用load_h5ad_data）")

        # 设置图像尺寸
        if image_size is None and self.histology_image is not None:
            image_size = (self.histology_image.shape[0], self.histology_image.shape[1])
        elif image_size is None:
            image_size = (500, 500)
            print(f"未指定图像尺寸且无组织学图像，使用默认尺寸: {image_size}")

        # 坐标转换核心逻辑：使用scale_factor
        try:
            # 提取空间转录组数据中的缩放因子（适配10x Genomics等标准格式）
            coord_type = self.coord_type
            scale_factor = self.scale_factor

            # 物理坐标（通常为微米）转换为像素坐标：像素 = 物理坐标 × 缩放因子
            x_scaled = self.coordinates[:, 0] * scale_factor
            y_scaled = self.coordinates[:, 1] * scale_factor
            print(f"使用{coord_type}缩放因子 {scale_factor} 转换坐标")

        except (KeyError, StopIteration) as e:
            # 无缩放因子时使用MinMaxScaler作为降级方案
            print(f"警告：{str(e)}，使用MinMaxScaler进行坐标映射（可能存在偏差）")
            scaler = MinMaxScaler(feature_range=(0, image_size[1] - 1))
            x_scaled = scaler.fit_transform(self.coordinates[:, 0].reshape(-1, 1)).flatten()
            scaler = MinMaxScaler(feature_range=(0, image_size[0] - 1))
            y_scaled = scaler.fit_transform(self.coordinates[:, 1].reshape(-1, 1)).flatten()

        # 为每个基因创建表达热图
        for gene in tqdm(genes, desc="创建基因表达热图"):
            if gene not in self.expression_data.columns:
                print(f"警告: 基因 {gene} 不在表达数据中，跳过")
                continue

            # 获取基因表达值
            expression = self.expression_data[gene].values

            # 创建空白画布
            gene_map = np.zeros(image_size, dtype=np.float32)

            # 将表达值分配到对应坐标
            x_int = np.round(x_scaled).astype(int)
            y_int = np.round(y_scaled).astype(int)

            # 处理可能的边界溢出（确保坐标在图像范围内）
            x_int = np.clip(x_int, 0, image_size[1] - 1)
            y_int = np.clip(y_int, 0, image_size[0] - 1)

            # 累加表达值（处理重叠点）
            for x, y, val in zip(x_int, y_int, expression):
                gene_map[y, x] += val

            # 插值平滑
            if interpolation_method == 'gaussian':
                gene_map = gaussian(gene_map, sigma=sigma, preserve_range=True)
            elif interpolation_method == 'idw':
                gene_map = self._inverse_distance_weighting(
                    x_scaled, y_scaled, expression, image_size, power=2)
            else:
                raise ValueError(f"不支持的插值方法: {interpolation_method}")

            # 归一化到0-1范围
            gene_map = (gene_map - gene_map.min()) / (gene_map.max() - gene_map.min() + 1e-10)

            # 保存热图
            self.results[gene] = {'expression_map': gene_map}
            output_path = os.path.join(self.output_dir, "gene_maps", f"{gene}_expression.png")

            plt.figure(figsize=(8, 8))
            plt.imshow(gene_map, cmap='viridis')
            plt.colorbar(label='标准化表达值')
            plt.title(f"{gene} 表达热图")
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"基因表达热图已保存至: {os.path.join(self.output_dir, 'gene_maps')}")
        return self

    def _inverse_distance_weighting(self, x, y, values, image_size, power=2):
        """反距离加权插值"""
        # 创建网格
        grid_x, grid_y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        # 构建KDTree加速近邻搜索
        tree = KDTree(np.vstack([x, y]).T)

        # 搜索最近邻点并计算IDW
        distances, indices = tree.query(grid_points, k=10)  # 取10个最近邻
        weights = 1 / (distances ** power + 1e-10)  # 避免除零

        # 计算加权平均
        weighted_values = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)

        # 重塑为图像尺寸
        return weighted_values.reshape(image_size)

    def calculate_colocalization(self, genes, threshold_method='quantile', quantile=0.9):
        """计算基因高表达区域与ROI的覆盖关系（以ROI覆盖率为核心指标）

        Args:
            genes: 基因列表
            threshold_method: 高表达区域阈值方法，'quantile'或'otsu'
            quantile: 分位数阈值（仅当threshold_method='quantile'时有效）
        """
        if self.roi_mask is None:
            raise ValueError("请先加载ROI掩码（调用load_manual_roi）")

        # 调整ROI掩码尺寸以匹配基因表达热图
        for gene in tqdm(genes, desc="计算覆盖指标"):
            if gene not in self.results:
                print(f"警告: 未找到 {gene} 的表达热图，跳过")
                continue

            expr_map = self.results[gene]['expression_map']
            roi_resized = cv2.resize(
                self.roi_mask,
                (expr_map.shape[1], expr_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # 确定高表达区域阈值
            if threshold_method == 'quantile':
                threshold = np.quantile(expr_map, quantile)
            elif threshold_method == 'otsu':
                _, threshold = cv2.threshold(
                    (expr_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
                threshold /= 255  # 归一化回0-1范围
            else:
                raise ValueError(f"不支持的阈值方法: {threshold_method}")

            # 创建高表达区域掩码
            high_expr_mask = (expr_map >= threshold).astype(np.uint8)

            # 计算核心指标
            intersection = np.logical_and(high_expr_mask, roi_resized).sum()
            high_expr_total = high_expr_mask.sum()
            roi_total = roi_resized.sum()

            # 主要指标：ROI覆盖率（重点关注）
            roi_coverage = intersection / roi_total if roi_total > 0 else 0

            # 辅助指标：基因表达特异性
            expression_specificity = intersection / high_expr_total if high_expr_total > 0 else 0

            # 综合质量评分
            roi_quality_score = roi_coverage * expression_specificity

            # 保留原有指标作为参考
            union = np.logical_or(high_expr_mask, roi_resized).sum()
            iou = intersection / union if union > 0 else 0
            dice = (2 * intersection) / (high_expr_total + roi_total) if (high_expr_total + roi_total) > 0 else 0

            # 保存结果
            self.results[gene].update({
                'high_expr_mask': high_expr_mask,
                'roi_mask_resized': roi_resized,
                'threshold': threshold,
                # 主要指标
                'roi_coverage': roi_coverage,  # 重点关注
                'expression_specificity': expression_specificity,
                'roi_quality_score': roi_quality_score,
                # 参考指标
                'iou': iou,
                'dice': dice
            })

        print("覆盖指标计算完成")
        return self

    def interpret_results(self, gene):
        """根据新指标解释结果"""
        if gene not in self.results:
            print(f"警告: 未找到 {gene} 的分析结果")
            return

        res = self.results[gene]

        coverage = res['roi_coverage']
        specificity = res['expression_specificity']
        quality = res['roi_quality_score']

        print(f"\n=== {gene} 共定位分析结果 ===")
        print(f"ROI覆盖率: {coverage:.2%}")
        print(f"基因表达特异性: {specificity:.2%}")
        print(f"ROI标注质量分数: {quality:.4f}")

        # 结果解读
        if coverage >= 0.7:
            print("✅ ROI标注质量良好：基因高表达充分覆盖ROI区域")
        elif coverage >= 0.3:
            print("⚠️ ROI标注质量一般：ROI区域仅部分被覆盖")
        else:
            print("❌ ROI标注可能不合理：基因高表达与ROI重叠度低")

        if specificity >= 0.7:
            print("✅ 基因在ROI内特异性表达")
        elif specificity >= 0.3:
            print("⚠️ 基因表达部分集中于ROI内")
        else:
            print("❌ 基因表达分散，不局限于ROI")

        return self

    def permutation_test(self, genes, num_permutations=1000):
        """通过置换检验评估ROI覆盖率的显著性

        Args:
            genes: 基因列表
            num_permutations: 置换次数
        """
        for gene in tqdm(genes, desc="进行置换检验"):
            if gene not in self.results or 'high_expr_mask' not in self.results[gene]:
                print(f"警告: 未找到 {gene} 的高表达掩码，跳过置换检验")
                continue

            high_expr_mask = self.results[gene]['high_expr_mask']
            roi_mask = self.results[gene]['roi_mask_resized']
            roi_total = roi_mask.sum()
            actual_coverage = self.results[gene]['roi_coverage']

            # 进行置换检验（基于ROI覆盖率）
            permutation_coverages = []
            for _ in range(num_permutations):
                # 随机旋转和平移高表达掩码
                shuffled_mask = self._random_rotate_translate(high_expr_mask)
                # 计算与ROI的覆盖率
                intersection = np.logical_and(shuffled_mask, roi_mask).sum()
                perm_coverage = intersection / roi_total if roi_total > 0 else 0
                permutation_coverages.append(perm_coverage)

            # 计算p值（实际覆盖率高于随机分布的比例）
            p_value = np.mean(np.array(permutation_coverages) >= actual_coverage)

            # 保存结果
            self.results[gene].update({
                'permutation_coverages': permutation_coverages,
                'p_value': p_value
            })

            # 绘制置换检验结果
            self._plot_permutation_results(gene, actual_coverage, permutation_coverages)

        print("置换检验完成")
        return self

    def _random_rotate_translate(self, mask):
        """随机旋转和平移掩码，用于置换检验"""
        # 随机旋转（-180到180度）
        angle = np.random.uniform(-180, 180)
        rotated = rotate(mask, angle, reshape=False, mode='constant', cval=0)

        # 随机平移（最多10%图像尺寸）
        max_shift_x = int(mask.shape[1] * 0.1)
        max_shift_y = int(mask.shape[0] * 0.1)
        shift_x = np.random.randint(-max_shift_x, max_shift_x)
        shift_y = np.random.randint(-max_shift_y, max_shift_y)

        # 执行平移
        translated = np.roll(rotated, shift_y, axis=0)
        translated = np.roll(translated, shift_x, axis=1)

        # 平移后边缘设为0
        if shift_y > 0:
            translated[:shift_y, :] = 0
        elif shift_y < 0:
            translated[shift_y:, :] = 0

        if shift_x > 0:
            translated[:, :shift_x] = 0
        elif shift_x < 0:
            translated[:, shift_x:] = 0

        return (translated > 0.5).astype(np.uint8)

    def _plot_permutation_results(self, gene, actual_coverage, permutation_coverages):
        """绘制置换检验结果直方图"""
        plt.figure(figsize=(8, 6))
        plt.hist(permutation_coverages, bins=30, alpha=0.7, label='置换样本覆盖率分布')
        plt.axvline(actual_coverage, color='red', linestyle='--',
                    label=f'实际覆盖率: {actual_coverage:.4f}')
        plt.xlabel('ROI覆盖率')
        plt.ylabel('频率')
        plt.title(f'{gene} ROI覆盖置换检验 (p值: {self.results[gene]["p_value"]:.4f})')
        plt.legend()

        output_path = os.path.join(self.output_dir, "colocalization",
                                   f"{gene}_permutation_test.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_colocalization(self, genes):
        """可视化基因高表达区域与ROI的覆盖情况"""
        # 创建专门存放结果的目录
        spot_roi_dir = os.path.join(self.output_dir, "colocalization")
        os.makedirs(spot_roi_dir, exist_ok=True)

        for gene in genes:
            if gene not in self.results or 'high_expr_mask' not in self.results[gene]:
                print(f"警告: 未找到 {gene} 的覆盖数据，跳过可视化")
                continue

            expr_map = self.results[gene]['expression_map']
            high_expr_mask = self.results[gene]['high_expr_mask']
            roi_mask = self.results[gene]['roi_mask_resized']
            coverage = self.results[gene]['roi_coverage']
            expr_values = self.expression_data[gene].values
            x_coords = self.coordinates[:, 0]
            y_coords = self.coordinates[:, 1]

            # ----------------------
            # 1. 生成改进的4合1覆盖图
            # ----------------------
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))

            # 1.1 基因表达热图
            axes[0, 0].imshow(expr_map, cmap='viridis')
            axes[0, 0].set_title(f'{gene} 表达热图')
            axes[0, 0].axis('off')

            # 1.2 ROI掩码与高表达区域叠加（突出覆盖率）
            overlay = np.zeros((*expr_map.shape, 3), dtype=np.uint8)
            overlay[roi_mask == 1] = [0, 100, 0]  # 深绿色：ROI区域
            overlay[high_expr_mask == 1] = [255, 0, 0]  # 红色：高表达区域
            overlay[np.logical_and(high_expr_mask == 1, roi_mask == 1)] = [255, 255, 0]  # 黄色：重叠区域

            axes[0, 1].imshow(overlay)
            axes[0, 1].set_title(f'ROI覆盖可视化 (覆盖率: {coverage:.2%})')
            axes[0, 1].axis('off')

            # 1.3 覆盖率热图（用颜色深度表示覆盖程度）
            coverage_map = np.zeros((*expr_map.shape, 3), dtype=np.uint8)
            for i in range(3):
                coverage_map[:, :, i] = (expr_map * 255).astype(np.uint8)

            # 在ROI区域内用红色渐变表示表达强度
            roi_indices = roi_mask == 1
            coverage_map[roi_indices, 0] = np.clip(expr_map[roi_indices] * 255, 0, 255)
            coverage_map[roi_indices, 1] = 0
            coverage_map[roi_indices, 2] = 0

            axes[1, 0].imshow(coverage_map)
            axes[1, 0].set_title('ROI内基因表达强度')
            axes[1, 0].axis('off')

            # 1.4 表达热图与ROI轮廓叠加
            if self.histology_image is not None:
                axes[1, 1].imshow(cv2.resize(
                    self.histology_image,
                    (expr_map.shape[1], expr_map.shape[0])
                ))
            else:
                axes[1, 1].imshow(expr_map, cmap='viridis')

            contours = measure.find_contours(roi_mask, level=0.5)
            for contour in contours:
                axes[1, 1].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)

            axes[1, 1].set_title('表达热图与ROI轮廓叠加')
            axes[1, 1].axis('off')

            # 保存4合1图像
            colocalization_path = os.path.join(self.output_dir, "colocalization",
                                               f"{gene}_coverage_analysis.png")
            plt.savefig(colocalization_path, dpi=300, bbox_inches='tight')
            plt.close()

            # ----------------------
            # 2. 生成单独的spot分布图
            # ----------------------
            # 坐标转换
            try:
                scale_factor = self.scale_factor
                x_scaled = x_coords * scale_factor
                y_scaled = y_coords * scale_factor
            except:
                scaler = MinMaxScaler(feature_range=(0, expr_map.shape[1] - 1))
                x_scaled = scaler.fit_transform(x_coords.reshape(-1, 1)).flatten()
                scaler = MinMaxScaler(feature_range=(0, expr_map.shape[0] - 1))
                y_scaled = scaler.fit_transform(y_coords.reshape(-1, 1)).flatten()

            # 准备背景图
            if self.histology_image is not None:
                background = cv2.resize(
                    self.histology_image,
                    (expr_map.shape[1], expr_map.shape[0])
                )
            else:
                background = np.zeros((*expr_map.shape, 3), dtype=np.uint8)

            # 绘制spot图
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            ax.imshow(background)

            # 绘制spot（颜色对应表达值）
            scatter = ax.scatter(
                x_scaled, y_scaled,
                c=expr_values,
                cmap='viridis',
                s=((self.spot_diameter*self.scale_factor)/2)**2,
                alpha=0.8
            )

            # 叠加ROI轮廓
            contours = measure.find_contours(roi_mask, level=0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)

            ax.set_title(f'{gene} 表达spot在ROI上的分布 (覆盖率: {coverage:.2%})')
            ax.axis('off')
            plt.colorbar(scatter, ax=ax, shrink=0.8, label='表达值')

            # 保存spot图
            spot_path = os.path.join(spot_roi_dir, f"{gene}_spot_on_roi.png")
            plt.savefig(spot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存 {gene} 的spot分布图: {spot_path}")

            # 输出结果解释
            self.interpret_results(gene)

        print(f"覆盖分析可视化结果已保存至: {os.path.join(self.output_dir, 'colocalization')}")
        return self

    def generate_summary_table(self, genes, output_name="summary_table.png"):
        """生成覆盖分析结果汇总表格（以ROI覆盖率为核心指标）"""
        # 准备数据
        data = []
        for gene in genes:
            if gene not in self.results:
                continue

            res = self.results[gene]
            # 显著性标记
            p_val = res.get('p_value', np.nan)
            if p_val <= 0.001:
                sig = '***'
            elif p_val <= 0.01:
                sig = '**'
            elif p_val <= 0.05:
                sig = '*'
            else:
                sig = ''

            data.append([
                gene,
                f"{res.get('roi_coverage', np.nan):.2%}",
                f"{res.get('expression_specificity', np.nan):.2%}",
                f"{res.get('roi_quality_score', np.nan):.4f}",
                f"{res.get('iou', np.nan):.4f}",
                f"{res.get('dice', np.nan):.4f}",
                f"{p_val:.4f}{sig}"
            ])

        # 创建表格
        fig, ax = plt.subplots(figsize=(14, len(data) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        # 表格数据和列名
        table_data = np.array(data)
        col_labels = ['基因', 'ROI覆盖率', '表达特异性', '质量评分', 'IoU', 'Dice系数', 'p值']

        # 创建表格
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # 保存表格
        output_path = os.path.join(self.output_dir, "colocalization", output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"分析结果汇总表格已保存至: {output_path}")
        return self

    def run_complete_analysis(self, h5ad_path, roi_path, genes,
                              black_line_threshold=50,
                              min_contour_area=100,
                              min_area=50,
                              interpolation_method='gaussian',
                              sigma=2,
                              threshold_method='quantile',
                              quantile=0.9,
                              num_permutations=1000):
        """运行完整的覆盖分析流程

        Args:
            h5ad_path: h5ad文件路径
            roi_path: ROI图像文件路径
            genes: 要分析的基因列表
            black_line_threshold: 黑色线条的阈值
            min_contour_area: 轮廓的最小面积
            min_area: 最终ROI的最小面积
            interpolation_method: 插值方法
            sigma: 高斯平滑参数
            threshold_method: 高表达区域阈值方法
            quantile: 分位数阈值
            num_permutations: 置换检验次数
        """
        print("开始完整覆盖分析流程...")

        # 执行完整分析步骤
        (self.load_h5ad_data(h5ad_path)
         .load_manual_roi(roi_path,
                          black_line_threshold=black_line_threshold,
                          min_contour_area=min_contour_area,
                          min_area=min_area)
         .visualize_manual_roi()
         .create_gene_expression_maps(genes,
                                      interpolation_method=interpolation_method,
                                      sigma=sigma)
         .calculate_colocalization(genes,
                                   threshold_method=threshold_method,
                                   quantile=quantile)
         .permutation_test(genes,
                           num_permutations=num_permutations)
         .visualize_colocalization(genes)
         .generate_summary_table(genes))

        print("完整覆盖分析流程完成！")
        print(f"所有结果已保存至: {self.output_dir}")
        return self


# 主函数示例
if __name__ == "__main__":
    species = "Homo sapiens"
    cancer_type = "COAD"
    sample_id = "TENX154"
    data_dir = "/Users/wuyang/Documents/MyPaper/3/dataset/HEST-data/"

    # 1. h5ad文件路径（包含空间转录组数据和组织学图像）
    h5ad_path = os.path.join(data_dir, species, cancer_type,
                             f"{sample_id}_adata.h5ad")

    # 2. ROI文件路径（支持图像格式或坐标文件）
    roi_path = f"/Users/wuyang/Documents/MyPaper/3/gsVis/src/DownStream/roi_files/{sample_id}_downscaled_fullres.jpeg"

    gene_list = ['ACTG2', 'CSRP1', 'DES', 'FXYD3']
    output_dir = "/Users/wuyang/Documents/MyPaper/3/gsVis/src/DownStream/gene_roi_colocalization"  # 结果输出目录

    # 初始化分析器并运行完整分析
    analyzer = SpatialColocalizationAnalyzer(output_dir=output_dir)
    analyzer.run_complete_analysis(
        h5ad_path=h5ad_path,
        roi_path=roi_path,
        genes=gene_list,
        black_line_threshold=20,  # 阈值越高可以识别更黑的线条
        min_contour_area=100,  # 最小轮廓面积
        min_area=1000,  # 最终ROI最小面积
        quantile=0.99,  # 取表达值前n%作为高表达区域
        num_permutations=1000  # 置换检验次数
    )