import os
import sys
from collections import Counter

# 添加当前目录到Python路径，确保可以导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis_config import AnalysisConfig
from data_loader import load_multi_sample_data, group_samples_by_cancer
from consensus import HierarchicalConsensusBuilder
from annotator import HybridAnnotator
from visualization import enhanced_visualization
from analysis_utils import GeneSetManager, generate_enhanced_report, batch_correct_expression
from feature_engine import ThreeDFeatureEngine
from similarity import MultiModalSimilarity


def build_cluster_3d_features(sample_data):
    """
    为所有样本的所有聚类构建三维特征（跨样本可比）
    :param sample_data: 加载后的样本数据
    :return: 所有聚类的三维特征列表 + 样本-特征映射字典
    """
    print("\n🔬 正在构建聚类三维特征（转录+微环境+功能）...")
    all_cluster_features = []  # 存储所有聚类的三维特征
    sample_feature_map = {}  # 样本ID -> 该样本的聚类特征列表

    for sample in sample_data:
        sample_id = sample["sample_id"]
        # 初始化特征工程（适配当前样本的标准化数据）
        feature_engine = ThreeDFeatureEngine(sample)
        # 构建该样本所有聚类的三维特征
        sample_cluster_features = feature_engine.build_all_clusters_3d_features()
        # 保存到全局列表和映射字典
        all_cluster_features.extend(sample_cluster_features)
        sample_feature_map[sample_id] = sample_cluster_features

    print(f"✅ 成功构建 {len(all_cluster_features)} 个聚类的三维特征")
    return all_cluster_features, sample_feature_map


def calc_cluster_similarity_matrix(all_cluster_features):
    """
    计算聚类间相似度矩阵（适配CSMs/MPs构建）
    :param all_cluster_features: 所有聚类的三维特征列表
    :return: 相似度矩阵结果（dict）
    """
    print("\n📏 正在计算聚类间多模态相似度矩阵...")

    # 兼容AnalysisConfig（如果是类，用 getattr 而非 get）
    csm_weights = getattr(AnalysisConfig, "csm_weights", [0.3, 0.5, 0.2])
    mp_weights = getattr(AnalysisConfig, "mp_weights", [0.2, 0.2, 0.6])

    similarity_calc = MultiModalSimilarity(
        csm_weights=csm_weights,
        mp_weights=mp_weights,
        decimal_places=4
    )

    # 构建聚类ID到特征的映射
    cluster_id_to_feat = {feat["cluster_id"]: feat for feat in all_cluster_features}
    cluster_ids = list(cluster_id_to_feat.keys())
    n_clusters = len(cluster_ids)

    # 初始化相似度矩阵
    sim_matrix = {}
    for i in range(n_clusters):
        cluster_i = cluster_ids[i]
        sim_matrix[cluster_i] = {}
        feat_i = cluster_id_to_feat[cluster_i]
        for j in range(n_clusters):
            cluster_j = cluster_ids[j]
            feat_j = cluster_id_to_feat[cluster_j]

            # 区分CSMs/MPs模式（同癌症=CSMs，跨癌症=MPs）
            if feat_i["cancer_type"] == feat_j["cancer_type"]:
                sim = similarity_calc.calc_comprehensive_similarity(feat_i, feat_j, mode="CSMs")
            else:
                sim = similarity_calc.calc_comprehensive_similarity(feat_i, feat_j, mode="MPs")

            sim_matrix[cluster_i][cluster_j] = sim

    print(f"✅ 完成 {n_clusters} × {n_clusters} 相似度矩阵计算")

    return {
        "matrix": sim_matrix,
        "cluster_ids": cluster_ids,
        "cluster_feats": cluster_id_to_feat
    }


def enhanced_main_analysis():
    """增强版主分析函数 - 实现分层多模态框架"""
    print("=" * 60)
    print("增强版肿瘤结构分析 - 分层多模态框架")
    print("=" * 60)

    # 预加载基因集
    print("\n📚 正在加载本地基因集...")
    gene_sets = GeneSetManager.get_gene_sets()
    if not gene_sets:
        print("❌ 错误: 没有可用的基因集，分析终止")
        return {}, {}, {}
    else:
        print(f"✅ 成功加载 {len(gene_sets)} 个基因集")

    # 加载数据
    print("\n📂 正在加载样本数据...")
    sample_data = load_multi_sample_data(AnalysisConfig.sample_paths)
    if not sample_data:
        print("❌ 没有可用样本数据，分析终止")
        return {}, {}, {}

    # 批次效应校正
    if AnalysisConfig.batch_correction:
        print("\n🔧 正在进行批次效应校正...")
        sample_data = batch_correct_expression(sample_data)

    # -------------------------- 构建三维特征+相似度矩阵 --------------------------
    print("\n🔬 正在构建聚类三维特征（转录+微环境+功能）...")
    all_cluster_features = []  # 存储所有聚类的三维特征
    for sample in sample_data:
        sample_id = sample["sample_id"]
        # 初始化特征工程
        feature_engine = ThreeDFeatureEngine(sample)
        # 构建该样本所有聚类的三维特征
        sample_cluster_features = feature_engine.build_all_clusters_3d_features()

        # 核心：统一整合3d_features字段（和consensus.py结构一致）
        for cluster_feat in sample_cluster_features:
            cluster_feat['3d_features'] = {
                'transcript': cluster_feat['transcript_feature'],
                'cell_context': cluster_feat['cell_context_feature'],
                'functional': cluster_feat['functional_feature']
            }
            all_cluster_features.append(cluster_feat)
    print(f"✅ 成功构建 {len(all_cluster_features)} 个聚类的三维特征")

    # 计算相似度矩阵（统一传入3d_features）
    print("\n📏 正在计算聚类间多模态相似度矩阵...")
    similarity_calc = MultiModalSimilarity()
    # 构建聚类ID到特征的映射
    cluster_id_to_feat = {feat["cluster_id"]: feat for feat in all_cluster_features}
    cluster_ids = list(cluster_id_to_feat.keys())
    n_clusters = len(cluster_ids)
    # 初始化相似度矩阵
    sim_matrix = {}
    for i in range(n_clusters):
        cluster_i = cluster_ids[i]
        sim_matrix[cluster_i] = {}
        # 传入统一的3d_features结构
        feat_i = cluster_id_to_feat[cluster_i]['3d_features']
        for j in range(n_clusters):
            cluster_j = cluster_ids[j]
            # 传入统一的3d_features结构
            feat_j = cluster_id_to_feat[cluster_j]['3d_features']
            # 区分CSMs/MPs模式（同癌症=CSMs，跨癌症=MPs）
            if cluster_id_to_feat[cluster_i]["cancer_type"] == cluster_id_to_feat[cluster_j]["cancer_type"]:
                sim = similarity_calc.calc_comprehensive_similarity(feat_i, feat_j, mode="CSMs")
            else:
                sim = similarity_calc.calc_comprehensive_similarity(feat_i, feat_j, mode="MPs")
            sim_matrix[cluster_i][cluster_j] = sim
    print(f"✅ 完成 {n_clusters} × {n_clusters} 相似度矩阵计算")
    # ------------------------------------------------------------------------------------------

    # 按癌症类型分组
    cancer_groups = group_samples_by_cancer(sample_data)

    print(f"\n🎯 识别到 {len(cancer_groups)} 种癌症类型: {list(cancer_groups.keys())}")
    for cancer_type, samples in cancer_groups.items():
        print(f"   - {cancer_type}: {len(samples)} 个样本")

    # 第一阶段：癌症内部共识 (CSMs)
    print("\n" + "=" * 50)
    print("第一阶段: 癌症内部共识分析 (CSMs)")
    print("=" * 50)

    consensus_builder = HierarchicalConsensusBuilder()
    annotator = HybridAnnotator()

    all_csms = {}
    intra_cancer_networks = {}

    for cancer_type, samples in cancer_groups.items():
        if len(samples) >= 2:  # 至少需要2个样本
            print(f"\n🔍 分析 {cancer_type} 内部共识结构...")
            # 原有调用逻辑（无新增参数，避免不匹配）
            csms, network = consensus_builder.build_intra_cancer_consensus(samples)

            if csms:
                # 为CSMs添加注释
                print(f"   📝 为 {len(csms)} 个CSMs添加注释...")
                annotated_csms = annotator.batch_annotate_structures(csms, samples, "CSM")

                all_csms[cancer_type] = {
                    'structures': annotated_csms,
                    'cancer_name': samples[0]['cancer_name'],
                    'sample_count': len(samples),
                    # 存储当前癌症的聚类特征（仅备份，不影响原有逻辑）
                    'cluster_features': [f for f in all_cluster_features if f["cancer_type"] == cancer_type]
                }
                intra_cancer_networks[cancer_type] = network
                print(f"   ✅ {cancer_type}: 发现 {len(csms)} 个稳健的CSMs")
            else:
                print(f"   ⚠️ {cancer_type}: 未发现稳健的CSMs")
        else:
            print(f"   ⏭️ {cancer_type}: 样本数不足 ({len(samples)}), 跳过")

    if not all_csms:
        print("\n❌ 警告: 未发现任何癌症特异性结构")
        return {}, {}, {}

    # 第二阶段：泛癌共识 (MPs)
    print("\n" + "=" * 50)
    print("第二阶段: 泛癌共识分析 (MPs)")
    print("=" * 50)

    print("🔍 构建泛癌共识网络...")

    mp_structures, pan_cancer_network = consensus_builder.build_pan_cancer_consensus(all_csms)

    # 为MPs添加注释
    if mp_structures:
        print(f"   📝 为 {len(mp_structures)} 个MPs添加注释...")
        all_samples_flat = []
        for cancer_group in cancer_groups.values():
            all_samples_flat.extend(cancer_group)

        annotated_mps = annotator.batch_annotate_structures(mp_structures, all_samples_flat, "MP", all_csms)
        mp_structures = annotated_mps
        print(f"   ✅ 发现 {len(mp_structures)} 个泛癌元程序 (MPs)")
    else:
        print("   ⚠️ 未发现泛癌元程序")

    # 生成报告
    print("\n📊 生成分析报告...")
    report = generate_enhanced_report(all_csms, mp_structures, sample_data)

    # 可视化
    print("\n🎨 生成可视化结果...")
    enhanced_visualization(all_csms, mp_structures, intra_cancer_networks, pan_cancer_network, report)

    print("\n" + "=" * 60)
    print("分析完成! 🎉")
    print("=" * 60)

    return all_csms, mp_structures, report


def print_key_findings(all_csms, mp_structures):
    """输出关键发现"""
    print("\n🔬 关键科学发现:")
    print("-" * 50)

    if mp_structures:
        print(f"🎯 发现了 {len(mp_structures)} 个泛癌元程序 (MPs)")
        print("   这些代表跨癌症类型的核心生物学规律:")

        for mp_id, info in list(mp_structures.items())[:5]:  # 显示前5个
            hybrid_name = info.get('annotation', {}).get('hybrid_name', 'Unannotated')
            cancer_types = info.get('cancer_types', [])
            cancer_count = len(cancer_types)

            print(f"   • {mp_id}: {hybrid_name}")
            print(f"     覆盖 {cancer_count} 种癌症: {', '.join(cancer_types[:3])}" +
                  ("..." if len(cancer_types) > 3 else ""))

        if len(mp_structures) > 5:
            print(f"   ... 还有 {len(mp_structures) - 5} 个MPs")

    # CSMs统计
    total_csms = sum(len(info['structures']) for info in all_csms.values())
    print(f"\n🎯 发现了 {total_csms} 个癌症特异性模块 (CSMs)")

    for cancer_type, cancer_info in all_csms.items():
        csms = cancer_info['structures']
        if csms:
            cancer_name = cancer_info['cancer_name']
            print(f"   📍 {cancer_name} ({cancer_type}): {len(csms)} 个CSMs")

            # 统计注释类型
            annotations = [csm['annotation']['hybrid_name'] for csm in csms.values()
                           if 'annotation' in csm]
            if annotations:
                common_annots = Counter(annotations).most_common(2)
                for annot, count in common_annots:
                    print(f"      • {count} 个 {annot}")


def print_methodology_summary():
    """输出方法学总结"""
    print("\n💡 分析方法说明:")
    print("-" * 30)
    print(f"• 框架: 分层多模态分析 (CSMs → MPs)")
    # -------------------------- 更新方法学说明 --------------------------
    print(f"• 特征: 三维指纹 (区域特异性转录 + 细胞微环境组成 + 全量功能通路)")
    print(f"• 相似度: 多模态加权计算 (CSMs:微环境权重0.5 | MPs:功能通路权重0.6)")
    # ------------------------------------------------------------------------------------------
    print(f"• 去噪: 移除MT/RPS/HSP噪声基因")
    print(f"• 富集分析: {AnalysisConfig.enrichment_gene_sets}")
    print(f"• 可视化: 分层结构 + 3D特征空间 + 共识网络")


if __name__ == "__main__":
    # 运行增强分析
    all_csms, mp_structures, report = enhanced_main_analysis()

    # 输出关键发现
    print_key_findings(all_csms, mp_structures)

    # 输出方法学总结
    print_methodology_summary()

    # 输出结果位置
    print(f"\n📁 详细结果保存在: {AnalysisConfig.output_dir}/")
    print("   包含:")
    print("   • enhanced_analysis_report.json - 完整分析报告")
    print("   • hierarchical_structure.png - 分层结构图")
    print("   • 3d_feature_space.png - 3D特征空间图")
    print("   • network_*.png - 各癌症网络图")
    print("   • pan_cancer_network.png - 泛癌网络图")
    print("   • annotation_summary.png - 注释摘要图")
    print("   • comprehensive_report.png - 综合分析报告图")
    # -------------------------- 新增：补充结果说明（增量） --------------------------
    print("   • cluster_3d_features.json - 聚类三维特征数据")
    print("   • cluster_similarity_matrix.json - 聚类相似度矩阵")
    # ------------------------------------------------------------------------------------------

    print("\n✨ 分析流程完成!")