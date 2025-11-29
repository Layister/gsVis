"""多模态相似度计算"""

import numpy as np

class MultiModalSimilarity:
    """多模态相似度计算类"""

    def __init__(self,
                 csm_weights=None,  # CSMs模式的加权比例
                 mp_weights=None,  # MPs模式的加权比例
                 decimal_places=4):  # 结果保留小数位数
        """
        初始化相似度计算器
        :param csm_weights: list/tuple，CSMs模式权重 [转录, 微环境, 功能]，默认[0.3,0.5,0.2]
        :param mp_weights: list/tuple，MPs模式权重 [转录, 微环境, 功能]，默认[0.2,0.2,0.6]
        :param decimal_places: int，结果保留小数位数，默认4
        """
        # 初始化加权配置
        self.csm_weights = csm_weights or [0.3, 0.2, 0.5]
        self.mp_weights = mp_weights or [0.2, 0.2, 0.6]
        self.decimal_places = decimal_places

    def calc_transcript_similarity(self, feat_a, feat_b):
        """转录特征相似度：加权Jaccard系数（按SpecScore加权）"""
        genes_a = {g: s for g, s in feat_a["transcript"]["specific_genes"]}
        genes_b = {g: s for g, s in feat_b["transcript"]["specific_genes"]}
        all_genes = set(genes_a.keys()).union(set(genes_b.keys()))

        if not all_genes:
            return round(0.0, self.decimal_places)

        # 加权Jaccard计算（极简版）
        numerator = 0.0  # 交集加权和
        denominator = 0.0  # 并集加权和
        for g in all_genes:
            s_a = genes_a.get(g, 0.0)
            s_b = genes_b.get(g, 0.0)
            numerator += min(s_a, s_b)
            denominator += max(s_a, s_b)

        return round(numerator / (denominator + 1e-8), self.decimal_places)  # +1e-8避免除0

    def _bray_curtis_similarity(self, vec_a, vec_b):
        """私有方法：计算Bray-Curtis相似度（1 - 距离）"""
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        numerator = np.sum(np.abs(vec_a - vec_b))
        denominator = np.sum(vec_a + vec_b)
        if denominator == 0:
            return round(0.0, self.decimal_places)
        return round(1 - (numerator / denominator), self.decimal_places)

    def calc_cell_context_similarity(self, feat_a, feat_b):
        """细胞微环境相似度：Bray-Curtis系数（适配比例数据）"""
        # 核心修正：字段名从 cell_context_feature → cell_context
        ct_a = feat_a["cell_context"]["cell_type_proportions"]
        ct_b = feat_b["cell_context"]["cell_type_proportions"]
        all_ct = set(ct_a.keys()).union(set(ct_b.keys()))

        if not all_ct:
            return round(0.0, self.decimal_places)

        # 提取比例向量
        vec_a = [ct_a.get(ct, 0.0) for ct in all_ct]
        vec_b = [ct_b.get(ct, 0.0) for ct in all_ct]

        # 计算Bray-Curtis相似度
        return self._bray_curtis_similarity(vec_a, vec_b)

    def calc_functional_similarity(self, feat_a, feat_b):
        """功能通路相似度：纯Jaccard系数（全量保留通路）"""
        # 核心修正：字段名从 functional_feature → functional
        pw_a = {p["term"] for p in feat_a["functional"]["enriched_pathways"]}
        pw_b = {p["term"] for p in feat_b["functional"]["enriched_pathways"]}

        if not pw_a and not pw_b:
            return round(0.0, self.decimal_places)

        # 纯Jaccard计算
        intersection = len(pw_a.intersection(pw_b))
        union = len(pw_a.union(pw_b))
        return round(intersection / union, self.decimal_places) if union > 0 else round(0.0, self.decimal_places)

    def calc_comprehensive_similarity(self, feat_a, feat_b, mode="CSMs"):
        """综合相似度：动态加权（CSMs/MPs）"""
        sim_trans = self.calc_transcript_similarity(feat_a, feat_b)
        sim_cell = self.calc_cell_context_similarity(feat_a, feat_b)
        sim_func = self.calc_functional_similarity(feat_a, feat_b)

        # 加权策略（使用初始化的权重配置）
        if mode == "CSMs":  # 肿瘤内聚类：微环境权重最高
            w1, w2, w3 = self.csm_weights
            total = w1 * sim_trans + w2 * sim_cell + w3 * sim_func
        elif mode == "MPs":  # 泛癌聚类：功能通路权重最高
            w1, w2, w3 = self.mp_weights
            total = w1 * sim_trans + w2 * sim_cell + w3 * sim_func
        else:  # 默认平均
            total = (sim_trans + sim_cell + sim_func) / 3

        return round(total, self.decimal_places)