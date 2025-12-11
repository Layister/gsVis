import os
import sys
import json
from collections import Counter

# 添加路径以确保模块可导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis_config import AnalysisConfig
from data_loader import load_multi_sample_data, group_samples_by_cancer
from consensus import HierarchicalConsensusBuilder
from annotator import HybridAnnotator
from visualization import enhanced_visualization
from analysis_utils import GeneSetManager, generate_enhanced_report
from feature_engine import build_clusters_for_samples



# ====================================================================================
# ⭐ 主分析流程：分癌种 CSM → 泛癌 MP → 注释 → 报告 → 可视化
# ====================================================================================

def enhanced_main_analysis():
    print("=" * 70)
    print("      🌈 增强版肿瘤结构分析框架（2D 模型：功能轴 + 微环境）")
    print("=" * 70)

    # ------------------ 预加载基因集（用于功能轴） ------------------
    print("\n📚 加载基因集...")
    GeneSetManager.get_gene_sets()

    # ------------------ 加载样本数据 ------------------
    print("\n📂 加载样本数据...")
    sample_data = load_multi_sample_data(AnalysisConfig.sample_paths)
    if not sample_data:
        print("❌ 无有效样本数据")
        return {}, {}, {}

    # print("\n=== Check cluster domains in first sample ===")
    # for i, c in enumerate(sample_data[0]["clusters"][:5]):
    #     print(f"Cluster {i}: domain_count = {len(c['domains'])}")
    #     print("First 5 domains:", c["domains"][:5])
    #     print()


    # ------------------ 构建 3D（2D核心）特征 ------------------
    print("\n🔬 构建聚类特征（3d_features：transcript + cell_context + functional）...")
    all_cluster_features = build_clusters_for_samples(sample_data)
    print(f"   → 共 {len(all_cluster_features)} 个聚类")

    # print("\n=== Check cluster features (cell_context + functional) ===")
    # fe = FeatureEngine(sample_data[0])
    # features = fe.build_all_clusters_3d_features()
    # for i, f in enumerate(features[:5]):
    #     print(f"\n--- Cluster {i} ---")
    #     #  细胞组成
    #     cc = f["cell_context"]
    #     print("Cell Context:")
    #     print("  cell_type_count:", cc["cell_type_count"])
    #     print("  dominance:", cc["dominance"])
    #     print("  top cell types:",
    #           sorted(cc["cell_type_proportions"].items(),
    #                  key=lambda x: x[1],
    #                  reverse=True)[:10])
    #     #  功能轴
    #     func = f["functional"]
    #     axis_scores = func.get("axis_scores", {})
    #     print("\nFunctional Axes:")
    #     if axis_scores:
    #         print("  axis_scores:",
    #               sorted(axis_scores.items(), key=lambda x: x[1], reverse=True))
    #     else:
    #         print("  axis_scores: (none)")
    #     #  富集通路（注释）
    #     enriched = func.get("enriched_pathways", [])
    #     print("\nTop enriched pathways:")
    #     if enriched:
    #         for p in enriched[:5]:  # 只看前 5 个
    #             print(f"  • {p['term']} (adj_p={p['adj_pvalue']}, source={p['source']})")
    #     else:
    #         print("  (none)")
    #     print()


    # ------------------ 分癌种样本分组 ------------------
    cancer_groups = group_samples_by_cancer(sample_data)
    print("\n🎗 识别到以下癌症类型：")
    for c, samples in cancer_groups.items():
        print(f"   • {c}: {len(samples)} 个样本")

    # ------------------ CSM 分析 ------------------
    print("\n" + "=" * 50)
    print("          🔵 第一阶段：癌症内部共识结构 (CSMs)")
    print("=" * 50)

    builder = HierarchicalConsensusBuilder()
    annotator = HybridAnnotator()

    all_csms = {}
    intra_cancer_networks = {}

    for cancer, samples in cancer_groups.items():
        if len(samples) < 2:
            print(f"⏭️ 跳过 {cancer}（样本不足）")
            continue

        print(f"\n🚀 构建 {cancer} 内部 CSM...")
        csms, G = builder.build_intra_cancer_consensus(samples)

        annotated = annotator.batch_annotate_structures(csms, structure_type="CSM")

        all_csms[cancer] = {
            "structures": annotated,
            "cancer_name": samples[0]["cancer_name"],
            "sample_count": len(samples)
        }
        intra_cancer_networks[cancer] = G

    if not all_csms:
        print("❌ 未识别到任何 CSM")
        return {}, {}, {}

    # ------------------ MP 泛癌结构 ------------------
    print("\n" + "=" * 50)
    print("          🔴 第二阶段：泛癌结构 (MPs)")
    print("=" * 50)

    mp_structures, pan_network = builder.build_pan_cancer_consensus(all_csms)

    mp_structures = annotator.batch_annotate_structures(
        mp_structures,
        structure_type="MP"
    )

    # ------------------ 报告 ------------------
    print("\n📊 生成报告...")
    report = generate_enhanced_report(all_csms, mp_structures, sample_data)

    # ------------------ 可视化 ------------------
    print("\n🎨 生成可视化图表...")
    enhanced_visualization(all_csms, mp_structures, intra_cancer_networks, pan_network, report)

    print("\n✨ 全流程完成！")
    return all_csms, mp_structures, report


# ====================================================================================
# 简短输出：关键结果摘要
# ====================================================================================

def print_key_findings(all_csms, mp_structures):
    print("\n🔬 关键发现：")
    print("-" * 50)

    # 泛癌结构
    if mp_structures:
        print(f"🎯 跨癌种元程序 MPs: {len(mp_structures)} 个")
        for mp_id, info in list(mp_structures.items())[:5]:
            print(f"   • {mp_id}: {info['annotation']['hybrid_name']}")

    # CSMs
    total_csms = sum(len(x["structures"]) for x in all_csms.values())
    print(f"\n🎯 癌症特异结构 CSMs: {total_csms} 个")
    for c, info in all_csms.items():
        print(f"   • {c} ({info['cancer_name']}): {len(info['structures'])} 个")


# ====================================================================================

def print_methodology_summary():
    print("\n💡 方法学摘要：")
    print("-" * 30)
    print("• 特征：2D 指纹（功能轴 + 细胞组成）")
    print("• 相似度：功能轴 + 微环境加权组合")
    print("• 分层共识：按癌症 → 泛癌")
    print("• 注释：功能轴解释 + 细胞语境解释")
    print("• 可视化：PCA、雷达图、网络图")


# ====================================================================================

if __name__ == "__main__":
    all_csms, mp_structures, report = enhanced_main_analysis()

    print_key_findings(all_csms, mp_structures)
    print_methodology_summary()

    print(f"\n📁 结果已保存至: {AnalysisConfig.output_dir}/")
    print("   包含：")
    print("   • hierarchical_structure.png")
    print("   • functional_pca.png")
    print("   • radar/  (包含所有结构的功能轴 + 细胞组成雷达图)")
    print("   • network_*.png")
    print("   • pan_cancer_network.png")
    print("   • annotation_summary.png")
    print("   • enhanced_analysis_report.json")
    print("✨ 完成！")
