"""可视化功能"""

import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import Counter
import seaborn as sns

from analysis_config import AnalysisConfig



def visualize_hierarchical_structure(all_csms, mp_structures, output_dir):
    """可视化分层结构"""
    try:
        # 癌症类型分布
        cancer_types = list(all_csms.keys())
        csm_counts = [len(info['structures']) for info in all_csms.values()]

        # MPs按癌症类型覆盖
        mp_cancer_coverage = []
        for mp_id, mp_info in mp_structures.items():
            # 检查是否存在 cancer_count，不存在则跳过或设为0
            if 'cancer_count' in mp_info:
                mp_cancer_coverage.append(mp_info['cancer_count'])
            else:
                print(f"警告：MP {mp_id} 缺少 cancer_count，跳过")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # CSMs分布
        bars = ax1.bar(cancer_types, csm_counts, color='skyblue', alpha=0.7)
        ax1.set_title('Cancer-Specific Modules (CSMs) Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of CSMs')
        ax1.tick_params(axis='x', rotation=45)

        # 在柱子上添加数值
        for bar, count in zip(bars, csm_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(count), ha='center', va='bottom')

        # MPs覆盖度
        if mp_cancer_coverage:
            ax2.hist(mp_cancer_coverage, bins=range(2, max(mp_cancer_coverage) + 2),
                     alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title('Pan-Cancer Meta-Programs (MPs) Coverage', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Cancer Types')
            ax2.set_ylabel('Number of MPs')

            # 添加统计信息
            avg_coverage = np.mean(mp_cancer_coverage)
            ax2.axvline(avg_coverage, color='red', linestyle='--',
                        label=f'Average: {avg_coverage:.1f}')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hierarchical_structure.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 分层结构图已生成")

    except Exception as e:
        print(f"❌ 分层结构可视化失败: {str(e)}")


def visualize_3d_features(all_csms, mp_structures, output_dir):
    """可视化三维特征分布"""
    try:
        # 收集所有结构的特征信息
        all_features = []
        labels = []
        colors = []
        sizes = []

        # 收集CSMs特征
        for cancer_type, cancer_info in all_csms.items():
            for csm_id, csm_info in cancer_info['structures'].items():
                if 'annotation' in csm_info:
                    desc = csm_info['annotation']['descriptive']
                    context = csm_info['annotation']['contextual']

                    # 特征向量: [功能显著性, 环境多样性, 基因数量(标准化)]
                    func_score = desc.get('enrichment_score', 1) if desc.get('enrichment_score') else 1
                    context_diversity = context.get('context_diversity', 1)
                    gene_count = min(csm_info['annotation']['gene_count'] / 100, 10)  # 标准化

                    all_features.append([func_score, context_diversity, gene_count])
                    labels.append(f"CSM_{cancer_type}_{csm_id}")
                    colors.append('blue')
                    sizes.append(50)  # CSMs使用较小点

        # 收集MPs特征
        for mp_id, mp_info in mp_structures.items():
            if 'annotation' in mp_info:
                desc = mp_info['annotation']['descriptive']
                context = mp_info['annotation']['contextual']

                func_score = desc.get('enrichment_score', 1) if desc.get('enrichment_score') else 1
                context_diversity = context.get('context_diversity', 1)
                gene_count = min(mp_info['annotation']['gene_count'] / 100, 10)

                all_features.append([func_score, context_diversity, gene_count])
                labels.append(f"MP_{mp_id}")
                colors.append('red')
                sizes.append(100)  # MPs使用较大点

        if all_features:
            # 创建3D散点图
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            features_array = np.array(all_features)

            scatter = ax.scatter(features_array[:, 0], features_array[:, 1], features_array[:, 2],
                                 c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.5)

            ax.set_xlabel('Functional Significance\n(-log10 p-value)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Context Diversity\n(Number of Cell Types)', fontsize=11, fontweight='bold')
            ax.set_zlabel('Gene Count\n(Scaled)', fontsize=11, fontweight='bold')
            ax.set_title('3D Feature Space of Cancer Structures', fontsize=14, fontweight='bold')

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='CSMs (Cancer-Specific)'),
                Patch(facecolor='red', label='MPs (Pan-Cancer)')
            ]
            ax.legend(handles=legend_elements, loc='upper left')

            # 添加网格
            ax.grid(True, alpha=0.3)

            plt.savefig(os.path.join(output_dir, "3d_feature_space.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 3D特征空间图已生成")

            # 修正：直接调用函数而不是self.method
            create_2d_projections(features_array, colors, labels, output_dir)

    except Exception as e:
        print(f"❌ 3D特征可视化失败: {str(e)}")


def create_2d_projections(features_array, colors, labels, output_dir):
    """创建2D投影图 - 独立的辅助函数"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        projections = [
            (0, 1, 'Function vs Diversity'),
            (0, 2, 'Function vs Gene Count'),
            (1, 2, 'Diversity vs Gene Count')
        ]

        for idx, (x_idx, y_idx, title) in enumerate(projections):
            ax = axes[idx // 2, idx % 2]

            for i, (color, label) in enumerate(zip(colors, labels)):
                ax.scatter(features_array[i, x_idx], features_array[i, y_idx],
                           c=color, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

            ax.set_xlabel(['Function', 'Function', 'Diversity'][idx])
            ax.set_ylabel(['Diversity', 'Gene Count', 'Gene Count'][idx])
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # 第四个图例显示图
        axes[1, 1].axis('off')
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=10, label='CSMs'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='MPs')
        ]
        axes[1, 1].legend(handles=legend_elements, loc='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "2d_feature_projections.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 2D特征投影图已生成")

    except Exception as e:
        print(f"2D投影图生成失败: {str(e)}")


def visualize_consensus_networks(intra_networks, pan_network, output_dir):
    """可视化共识网络"""
    try:
        # 癌症内部网络
        for cancer_type, network in intra_networks.items():
            if network.number_of_nodes() > 0:
                plt.figure(figsize=(12, 10))

                # 使用spring布局
                pos = nx.spring_layout(network, k=2, iterations=100, seed=42)

                # 根据节点度设置节点大小
                node_sizes = [50 + 20 * network.degree(node) for node in network.nodes()]

                # 根据社区着色
                try:
                    from community import best_partition
                    partition = best_partition(network)
                    communities = list(set(partition.values()))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
                    node_colors = [colors[partition[node]] for node in network.nodes()]
                except:
                    node_colors = 'lightblue'

                nx.draw_networkx_nodes(network, pos, node_color=node_colors,
                                       node_size=node_sizes, alpha=0.8, edgecolors='black')

                # 根据权重设置边的宽度和透明度
                edge_weights = [network[u][v]['weight'] * 3 for u, v in network.edges()]
                edge_alphas = [network[u][v]['weight'] for u, v in network.edges()]

                nx.draw_networkx_edges(network, pos, edge_color='gray',
                                       width=edge_weights, alpha=0.6)

                plt.title(f'{cancer_type} - Intra-Cancer Consensus Network\n'
                          f'({network.number_of_nodes()} nodes, {network.number_of_edges()} edges)',
                          fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"network_{cancer_type}.png"),
                            dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ {cancer_type} 网络图已生成")

        # 泛癌网络
        if pan_network and pan_network.number_of_nodes() > 0:
            plt.figure(figsize=(14, 12))

            pos = nx.spring_layout(pan_network, k=3, iterations=200, seed=42)

            # 按癌症类型着色
            node_colors = []
            color_map = {}
            cancer_types = sorted(set(node.split('_')[0] for node in pan_network.nodes()))
            colors = plt.cm.Set3(np.linspace(0, 1, len(cancer_types)))

            for i, cancer in enumerate(cancer_types):
                color_map[cancer] = colors[i]

            for node in pan_network.nodes():
                cancer_type = node.split('_')[0]
                node_colors.append(color_map[cancer_type])

            # 节点大小基于连接数
            node_sizes = [100 + 30 * pan_network.degree(node) for node in pan_network.nodes()]

            nx.draw_networkx_nodes(pan_network, pos, node_color=node_colors,
                                   node_size=node_sizes, alpha=0.8, edgecolors='black', linewidths=1)

            # 边设置
            edge_weights = [pan_network[u][v]['weight'] * 4 for u, v in pan_network.edges()]
            nx.draw_networkx_edges(pan_network, pos, edge_color='lightcoral',
                                   width=edge_weights, alpha=0.7)

            # 创建图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color_map[cancer], label=cancer, alpha=0.8)
                               for cancer in cancer_types]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

            plt.title(f'Pan-Cancer Meta-Programs Network\n'
                      f'({pan_network.number_of_nodes()} CSMs, {pan_network.number_of_edges()} cross-cancer connections)',
                      fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pan_cancer_network.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 泛癌网络图已生成")

    except Exception as e:
        print(f"❌ 网络可视化失败: {str(e)}")


def visualize_annotation_summary(all_csms, mp_structures, output_dir):
    """可视化注释摘要"""
    try:
        # 收集所有注释
        all_annotations = []
        csms_annotations = []
        mps_annotations = []

        for cancer_type, cancer_info in all_csms.items():
            for csm_id, csm_info in cancer_info['structures'].items():
                if 'annotation' in csm_info:
                    annotation = csm_info['annotation']['hybrid_name']
                    all_annotations.append(annotation)
                    csms_annotations.append(annotation)

        for mp_id, mp_info in mp_structures.items():
            if 'annotation' in mp_info:
                annotation = mp_info['annotation']['hybrid_name']
                all_annotations.append(annotation)
                mps_annotations.append(annotation)

        if not all_annotations:
            print("⚠️ 没有可用的注释数据")
            return

        # 创建三个子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 总体注释分布
        annotation_counts = Counter(all_annotations).most_common(15)
        if annotation_counts:
            annotations, counts = zip(*annotation_counts)
            y_pos = np.arange(len(annotations))

            axes[0].barh(y_pos, counts, color='lightgreen', alpha=0.7)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(annotations, fontsize=9)
            axes[0].set_xlabel('Frequency')
            axes[0].set_title('Top 15 Structure Annotations\n(Overall)', fontweight='bold')

            # 在柱子上添加数值
            for i, count in enumerate(counts):
                axes[0].text(count + 0.1, i, str(count), va='center', fontsize=8)

        # 2. CSMs注释分布
        if csms_annotations:
            csms_counts = Counter(csms_annotations).most_common(10)
            csms_annots, csms_nums = zip(*csms_counts) if csms_counts else ([], [])

            axes[1].pie(csms_nums, labels=csms_annots, autopct='%1.1f%%', startangle=90,
                        textprops={'fontsize': 8})
            axes[1].set_title('CSMs Annotation Distribution', fontweight='bold')

        # 3. MPs注释分布  
        if mps_annotations:
            mps_counts = Counter(mps_annotations).most_common(10)
            mps_annots, mps_nums = zip(*mps_counts) if mps_counts else ([], [])

            axes[2].pie(mps_nums, labels=mps_annots, autopct='%1.1f%%', startangle=90,
                        textprops={'fontsize': 8})
            axes[2].set_title('MPs Annotation Distribution', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "annotation_summary.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 注释摘要图已生成")

    except Exception as e:
        print(f"❌ 注释摘要可视化失败: {str(e)}")


def create_comprehensive_report_visualization(report, output_dir):
    """创建综合分析报告可视化"""
    try:
        # 创建总结仪表板
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 癌症类型分布
        summary = report.get('summary', {})
        cancer_breakdown = summary.get('cancer_type_breakdown', {})

        if cancer_breakdown:
            cancer_names = list(cancer_breakdown.keys())
            csms_counts = [info['csms_count'] for info in cancer_breakdown.values()]
            sample_counts = [info['sample_count'] for info in cancer_breakdown.values()]

            x = np.arange(len(cancer_names))
            width = 0.35

            bars1 = axes[0, 0].bar(x - width / 2, csms_counts, width, label='CSMs', alpha=0.7)
            bars2 = axes[0, 0].bar(x + width / 2, sample_counts, width, label='Samples', alpha=0.7)

            axes[0, 0].set_xlabel('Cancer Types')
            axes[0, 0].set_ylabel('Counts')
            axes[0, 0].set_title('Cancer Type Distribution', fontweight='bold')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(cancer_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 结构类型饼图
        total_csms = summary.get('total_csms', 0)
        total_mps = summary.get('total_mps', 0)

        if total_csms + total_mps > 0:
            sizes = [total_csms, total_mps]
            labels = [f'CSMs\n({total_csms})', f'MPs\n({total_mps})']
            colors = ['lightblue', 'lightcoral']

            axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=90, textprops={'fontweight': 'bold'})
            axes[0, 1].set_title('Structure Type Distribution', fontweight='bold')

        # 3. 分析框架说明
        framework = report.get('analysis_framework', 'Hierarchical Multi-Modal Analysis')
        axes[1, 0].axis('off')
        framework_text = (
            f"Key Features:\n\n"
            f"• Hierarchical: CSMs → MPs\n"
            f"• Multi-Modal: 3D Feature Fingerprints\n"
            f"• Denoising: MT/RPS/HSP removal\n"
            f"• Statistical Gating: Hypergeometric test\n"
            f"• Hybrid Annotation: Function + Context"
        )
        axes[1, 0].text(0.1, 0.9, framework_text, transform=axes[1, 0].transAxes,
                        fontsize=12, va='top', linespacing=1.5,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

        # 4. 关键发现总结
        key_findings = (
            f"Key Discoveries:\n\n"
            f"• {total_csms} Cancer-Specific Modules\n"
            f"• {total_mps} Pan-Cancer Meta-Programs\n"
            f"• {summary.get('total_cancer_types', 0)} Cancer Types Analyzed\n"
            f"• {summary.get('sample_count', 0)} Total Samples"
        )
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, key_findings, transform=axes[1, 1].transAxes,
                        fontsize=12, va='top', linespacing=1.5, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

        plt.suptitle('Comprehensive Pan-Cancer Analysis Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comprehensive_report.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 综合分析报告图已生成")

    except Exception as e:
        print(f"❌ 综合分析报告可视化失败: {str(e)}")


def enhanced_visualization(all_csms, mp_structures, intra_networks, pan_network, report=None):
    """增强版可视化主函数"""
    output_dir = AnalysisConfig.output_dir
    os.makedirs(output_dir, exist_ok=True)


    print("\n" + "=" * 50)
    print("开始生成增强可视化...")

    try:
        # 1. 分层结构图
        visualize_hierarchical_structure(all_csms, mp_structures, output_dir)

        # 2. 三维特征分布图
        visualize_3d_features(all_csms, mp_structures, output_dir)

        # 3. 网络可视化
        visualize_consensus_networks(intra_networks, pan_network, output_dir)

        # 4. 注释摘要
        visualize_annotation_summary(all_csms, mp_structures, output_dir)

        # 5. 综合分析报告（如果提供了报告）
        if report:
            create_comprehensive_report_visualization(report, output_dir)

        print(f"✅ 所有可视化结果已保存到: {output_dir}/")

    except Exception as e:
        print(f"❌ 可视化过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()