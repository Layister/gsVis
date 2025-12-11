"""多模态相似度计算（细胞组成 + 功能轴）"""

import numpy as np


class MultiModalSimilarity:
    """多模态相似度计算类：仅使用细胞组成 + 功能轴"""

    def __init__(self,
                 csm_weights=None,   # CSM 模式权重 [cell, func]
                 mp_weights=None,    # MP 模式权重 [cell, func]
                 decimal_places=4):

        self.csm_weights = csm_weights or [0.4, 0.6]
        self.mp_weights = mp_weights or [0.4, 0.6]
        self.decimal_places = decimal_places

    # -------------------------------------------------------
    # 通用标准化函数（非负 + L1 标准化）
    # -------------------------------------------------------
    def _normalize_vector(self, values):
        """把任意实数向量变成非负、和为1的概率向量"""
        v = np.asarray(values, dtype=float)

        if v.size == 0:
            return v

        # 允许有负数：整体平移到非负
        min_v = v.min()
        if min_v < 0:
            v = v - min_v

        s = v.sum()
        if s <= 0:
            # 全 0 或者数值太小，直接返回原值
            return v

        return v / s

    # -------------------------------------------------------
    # Bray-Curtis 相似度（适用于比例数据 & 功能轴向量）
    # -------------------------------------------------------
    def _bray_curtis_similarity(self, vec_a, vec_b):
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)

        numerator = np.sum(np.abs(vec_a - vec_b))
        denominator = np.sum(vec_a + vec_b)

        if denominator == 0:
            return round(0.0, self.decimal_places)

        return round(1 - numerator / denominator, self.decimal_places)

    # -------------------------------------------------------
    # 1. 细胞组成相似度
    # -------------------------------------------------------
    def calc_cell_context_similarity(self, feat_a, feat_b):
        """细胞组成相似度（Bray-Curtis）"""

        ct_a = feat_a["cell_context"]["cell_type_proportions"]
        ct_b = feat_b["cell_context"]["cell_type_proportions"]

        all_keys = set(ct_a.keys()).union(ct_b.keys())
        if not all_keys:
            return round(0.0, self.decimal_places)

        vec_a = [ct_a.get(k, 0.0) for k in all_keys]
        vec_b = [ct_b.get(k, 0.0) for k in all_keys]

        # 进行标准化
        vec_a = self._normalize_vector(vec_a)
        vec_b = self._normalize_vector(vec_b)

        return self._bray_curtis_similarity(vec_a, vec_b)

    # -------------------------------------------------------
    # 2. 功能相似度
    # -------------------------------------------------------
    def calc_functional_similarity(self, feat_a, feat_b):
        """功能轴向量相似度（Bray-Curtis）"""

        func_a = feat_a.get("functional", {})
        func_b = feat_b.get("functional", {})

        axes_a = func_a.get("axis_scores", {})
        axes_b = func_b.get("axis_scores", {})

        if axes_a and axes_b:
            all_axes = set(axes_a.keys()).union(axes_b.keys())
            vec_a = [axes_a.get(ax, 0.0) for ax in all_axes]
            vec_b = [axes_b.get(ax, 0.0) for ax in all_axes]

            # 功能轴向量归一化到“贡献比例”
            vec_a = self._normalize_vector(vec_a)
            vec_b = self._normalize_vector(vec_b)

            return self._bray_curtis_similarity(vec_a, vec_b)

        # 如果功能轴都没有，则认为功能不可比，返回 0
        return round(0.0, self.decimal_places)

    # -------------------------------------------------------
    # 3. 综合相似度（细胞组成 + 功能轴）
    # -------------------------------------------------------
    def calc_comprehensive_similarity(self, feat_a, feat_b, mode="CSMs"):
        """综合 2D 相似度"""

        sim_cell = self.calc_cell_context_similarity(feat_a, feat_b)
        sim_func = self.calc_functional_similarity(feat_a, feat_b)

        # 权重选择
        if mode == "CSMs":
            wc, wf = self.csm_weights
        elif mode == "MPs":
            wc, wf = self.mp_weights
        else:
            wc = wf = 1.0

        denom = wc + wf
        if denom == 0:
            return round(0.0, self.decimal_places)

        sim = (wc * sim_cell + wf * sim_func) / denom
        return round(sim, self.decimal_places)
