"""功能轴 + 细胞环境 的结构注释系统"""

import numpy as np
from analysis_config import AnalysisConfig


# =============================
# 预定义功能轴注释映射
# =============================
AXIS_NAME_MAP = {
    axis_name: (names[0] if names else axis_name.replace("axis_", "").replace("_", " ").title())
    for axis_name, names in AnalysisConfig.functional_axes.items()
}


class HybridAnnotator:
    """注释 CSM / MP：基于聚合特征 3d_features（cell_context + functional）"""

    # ====================================================
    # 🔵 主函数：注释一个结构
    # ====================================================
    def annotate_structure(self, structure_info, structure_type="CSM"):
        """
        structure_info: 来自 consensus.py 的单个结构字典
                        必须包含聚合好的 3d_features 字段
        """
        feat = structure_info.get("3d_features", {})
        cell = feat.get("cell_context", {})
        func = feat.get("functional", {})

        # 1. 功能轴注释
        func_annot = self._annotate_functional_axis(func.get("axis_scores", {}))

        # 2. 细胞语境注释
        context_annot = self._annotate_cell_context(cell.get("cell_type_proportions", {}))

        # 3. 合并两个注释
        hybrid = f"{func_annot} [{context_annot}]"

        return {
            "hybrid_name": hybrid,
            "functional": func_annot,
            "contextual": context_annot,
            "structure_type": structure_type,
            "dominant_axes": sorted(func.get("axis_scores", {}).items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:3],
            "dominant_cell_types": sorted(cell.get("cell_type_proportions", {}).items(),
                                          key=lambda x: x[1],
                                          reverse=True)[:3],
        }

    # ====================================================
    # 🔵 1) 功能轴注释
    # ====================================================
    def _annotate_functional_axis(self, axis_scores):
        if not axis_scores:
            return "No dominant functional feature"

        # 找到主导功能轴
        top_axis, top_score = max(axis_scores.items(), key=lambda x: x[1])

        # 如果映射表里有，就用功能轴的英文名字
        if top_axis in AXIS_NAME_MAP:
            return AXIS_NAME_MAP[top_axis]

        # fallback：形如 "TNFA SIGNALING"
        return top_axis.replace("axis_", "").replace("_", " ").title()

    # ====================================================
    # 🔵 2) 细胞组成语境注释
    # ====================================================
    def _annotate_cell_context(self, cell_props):
        if not cell_props:
            return "Unknown"

        total = sum(cell_props.values())
        if total == 0:
            return "Unknown"

        # 找到主导细胞类型
        top_cell, p = max(cell_props.items(), key=lambda x: x[1])

        # 规则：>0.5 = 强主导; >0.3 = 主导; 否则 = 混合
        if p > 0.55:
            return f"{top_cell}-Dominant"
        elif p > 0.30:
            return f"{top_cell}-Primary"
        else:
            # 选 top3 做混合注释
            sorted_cells = sorted(cell_props.items(), key=lambda x: x[1], reverse=True)
            names = [c for c, _ in sorted_cells[:3]]
            return f"{'/'.join(names)}-Mixed"

    # ====================================================
    # 🔵 批量注释结构
    # ====================================================
    def batch_annotate_structures(self, structures_dict, structure_type="CSM"):
        annotated = {}
        for sid, sinfo in structures_dict.items():
            ann = self.annotate_structure(sinfo, structure_type)
            sinfo["annotation"] = ann
            annotated[sid] = sinfo
        return annotated
