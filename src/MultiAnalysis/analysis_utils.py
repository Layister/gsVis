"""工具函数与基因集管理"""

import os
import json
import numpy as np
import pandas as pd
import traceback

from analysis_config import AnalysisConfig


# -----------------------------------------------------------
# Numpy → Python 类型转换（用于 JSON 序列化）
# -----------------------------------------------------------
def convert_numpy_types(obj):
    """将numpy数据类型转换为Python原生类型以便JSON序列化"""
    import numpy as np

    if isinstance(obj, (np.integer, np.int64, np.int32, np.int8, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16, np.float_)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, set):
        return [convert_numpy_types(i) for i in obj]   # JSON-safe
    else:
        return obj


# -----------------------------------------------------------
# 基因集管理器（仅保留 GMT 读取 + 功能轴基因集）
# -----------------------------------------------------------
class GeneSetManager:
    """基因集管理器：直接读取本地GMT文件"""
    _gene_sets = None
    _axis_genesets = None  # axis_name → set(genes)

    @classmethod
    def get_gene_sets(cls):
        """返回所有加载的 gene sets（按库名组织）"""
        if cls._gene_sets is None:
            cls._gene_sets = cls._load_gene_sets()
        return cls._gene_sets

    @classmethod
    def _load_gene_sets(cls):
        """从 analysis_config.local_gmt_files 加载所有GMT文件"""
        gene_sets = {}

        for lib_name, gmt_path in AnalysisConfig.local_gmt_files.items():
            if not os.path.exists(gmt_path):
                print(f"⚠️ GMT文件不存在: {gmt_path}")
                continue

            try:
                gmt_data = cls._read_gmt_file(gmt_path)
                gene_sets[lib_name] = gmt_data
                print(f"✅ 成功加载基因集 {lib_name}, 包含 {len(gmt_data)} 个条目")
            except Exception as e:
                print(f"❌ 加载基因集 {lib_name} 失败: {str(e)}")
                traceback.print_exc()

        return gene_sets

    @classmethod
    def get_axis_genesets(cls):
        """根据 functional_axes 拼出每个功能轴的基因集合"""
        if cls._axis_genesets is not None:
            return cls._axis_genesets

        gene_sets_by_lib = cls.get_gene_sets()
        axis = {}

        for axis_name, gs_names in AnalysisConfig.functional_axes.items():
            collected = set()

            for gs_name in gs_names:
                # 遍历所有库，找到名称匹配的基因集
                for lib_name, lib_sets in gene_sets_by_lib.items():
                    if gs_name in lib_sets:
                        collected.update(lib_sets[gs_name])

            axis[axis_name] = collected

        cls._axis_genesets = axis
        return axis

    @staticmethod
    def _read_gmt_file(gmt_file_path):
        """从GMT文件读取 gene set"""
        gene_sets = {}
        with open(gmt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                set_name = parts[0]
                genes = [g.strip() for g in parts[2:] if g.strip()]
                if genes:
                    gene_sets[set_name] = genes
        return gene_sets


# -----------------------------------------------------------
# （可选）富集分析模块
# -----------------------------------------------------------
def perform_local_enrichment(gene_list):
    """使用本地 GMT 文件做富集分析（可选功能）"""
    if len(gene_list) < 5:
        return {"error": "基因数量不足", "total_genes": len(gene_list)}

    gene_sets = GeneSetManager.get_gene_sets()
    if not gene_sets:
        return {"error": "没有可用的基因集"}

    filtered = {k: v for k, v in gene_sets.items()
                if k in AnalysisConfig.enrichment_gene_sets}

    if not filtered:
        return {"error": "选定的基因集不可用"}

    # 简单统计：哪些 gene set 含有输入基因
    matched_terms = []
    gene_set_cutoff = AnalysisConfig.enrichment_cutoff

    for lib_name, gmt_sets in filtered.items():
        for term_name, genes in gmt_sets.items():
            overlap = set(gene_list) & set(genes)
            if len(overlap) >= 3:
                matched_terms.append({
                    "term": term_name,
                    "library": lib_name,
                    "overlap": list(overlap),
                    "overlap_count": len(overlap)
                })

    return {
        "matched_terms": matched_terms,
        "matched_count": len(matched_terms),
        "total_genes": len(gene_list)
    }


# -----------------------------------------------------------
# 报告生成模块
# -----------------------------------------------------------
def generate_enhanced_report(all_csms, mp_structures, sample_data):
    """生成增强版分析报告（适配新架构）"""

    total_csms = sum(len(info["structures"]) for info in all_csms.values())
    total_samples = len(sample_data)

    significant_csms = sum(
        1 for cancer_info in all_csms.values()
        for s in cancer_info["structures"].values()
        if s.get("sample_coverage", 0) >= 2
    )

    # 转换 MP 键为字符串
    mp_str = {str(k): convert_numpy_types(v) for k, v in mp_structures.items()}

    # 转换 CSM 键
    csm_str = {}
    for cancer, info in all_csms.items():
        info_copy = info.copy()
        info_copy["structures"] = {
            str(cid): convert_numpy_types(cinfo)
            for cid, cinfo in info["structures"].items()
        }
        csm_str[cancer] = info_copy

    report = {
        "analysis_framework": "分层多模态泛癌分析",
        "summary": {
            "total_cancer_types": len(all_csms),
            "total_csms": total_csms,
            "significant_csms": significant_csms,
            "total_mps": len(mp_structures),
            "sample_count": total_samples,
        },
        "pan_cancer_meta_programs": mp_str,
        "cancer_specific_modules": csm_str,
        "analysis_parameters": {
            "consensus_thresholds": AnalysisConfig.consensus_params,
            "enrichment_cutoff": AnalysisConfig.enrichment_cutoff,
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }

    os.makedirs(AnalysisConfig.output_dir, exist_ok=True)
    report_path = os.path.join(AnalysisConfig.output_dir, "enhanced_analysis_report.json")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(report), f, indent=2, ensure_ascii=False)
        print(f"✅ 增强分析报告已保存: {report_path}")
    except Exception as e:
        print(f"❌ 生成报告失败: {str(e)}")

    return report
