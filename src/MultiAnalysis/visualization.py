"""可视化：基于 2D 特征（功能轴 + 细胞组成）"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from analysis_config import AnalysisConfig


# ===============================================================
# 工具函数：雷达图
# ===============================================================

def radar_plot(ax, scores_dict, title):
    if not scores_dict:
        ax.text(0.5, 0.5, "No Data", ha='center')
        return

    labels = list(scores_dict.keys())
    values = list(scores_dict.values())

    # 闭环
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_title(title, fontsize=12)


# ===============================================================
# 1. 可视化层次结构（CSMs & MPs 分布）
# ===============================================================

def visualize_hierarchical_structure(all_csms, mp_structures, output_dir):
    try:
        cancers = list(all_csms.keys())
        csm_counts = [len(info['structures']) for info in all_csms.values()]
        mp_counts = len(mp_structures)

        plt.figure(figsize=(10, 6))
        plt.bar(cancers, csm_counts, color='skyblue')
        plt.title("Cancer-Specific Modules Count")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hierarchy_summary.png"))
        plt.close()
        print("✔ CSM count figure saved")
    except Exception as e:
        print("❌ Error in visualize_hierarchical_structure:", e)


# ===============================================================
# 2. PCA 可视化所有结构的 axis_scores（功能结构空间）
# ===============================================================

def visualize_functional_pca(all_csms, mp_structures, output_dir):
    try:
        X = []
        colors = []
        labels = []

        axis_names = list(AnalysisConfig.functional_axes.keys())

        # -------- CSM 部分 --------
        for cancer, info in all_csms.items():
            # 有些 cancer 可能没有 "structures" 键，做个防御
            structures = info.get("structures", {})
            for csm_id, csm in structures.items():

                # 兼容两种存储格式：
                # 1) csm["3d_features"]["functional"]["axis_scores"]
                # 2) csm["functional"]["axis_scores"]
                if "3d_features" in csm:
                    func_block = csm["3d_features"].get("functional", {})
                else:
                    func_block = csm.get("functional", {})

                axis_scores = func_block.get("axis_scores", {})
                if not axis_scores:
                    continue

                X.append([axis_scores.get(ax, 0.0) for ax in axis_names])
                colors.append("blue")
                labels.append(f"{cancer}_{csm_id}")

        # -------- MP 部分 --------
        for mp_id, mp in mp_structures.items():
            if "3d_features" in mp:
                func_block = mp["3d_features"].get("functional", {})
            else:
                func_block = mp.get("functional", {})

            axis_scores = func_block.get("axis_scores", {})
            if not axis_scores:
                continue

            X.append([axis_scores.get(ax, 0.0) for ax in axis_names])
            colors.append("red")
            labels.append(f"MP_{mp_id}")

        # 如果有效点太少，没必要做 PCA
        if len(X) < 2:
            print("⚠ PCA skipped (insufficient data)")
            return

        X = np.array(X)

        # PCA 降维到 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.8)

        # 简单标一下点的 label（可以按需关掉）
        for i, lab in enumerate(labels):
            plt.text(X_pca[i, 0], X_pca[i, 1], lab, fontsize=6, alpha=0.7)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Functional axes PCA (CSMs & MPs)")

        legend_handles = [
            Patch(color="blue", label="CSM"),
            Patch(color="red", label="MP"),
        ]
        plt.legend(handles=legend_handles, loc="best")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "functional_pca.png"))
        plt.close()

        print("✔ Functional PCA saved")

    except Exception as e:
        print("❌ Error in visualize_functional_pca:", e)


# ===============================================================
# 3. 雷达图：功能轴与细胞组成（为每个 CSM/MP 绘制）
# ===============================================================

def visualize_feature_radar(structures, name_prefix, output_dir):
    """为每个结构绘制雷达图（功能轴 + 细胞组成）"""
    os.makedirs(os.path.join(output_dir, "radar"), exist_ok=True)

    for sid, sinfo in structures.items():
        feat = sinfo.get("3d_features", {})

        func = feat.get("functional", {}).get("axis_scores", {})
        cell = feat.get("cell_context", {}).get("cell_type_proportions", {})

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(polar=True))
        radar_plot(axs[0], func, "Functional Axes")
        radar_plot(axs[1], cell, "Cell Composition")

        plt.suptitle(f"{name_prefix}_{sid}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radar", f"{name_prefix}_{sid}.png"))
        plt.close()


# ===============================================================
# 4. 网络图（保留原始逻辑）
# ===============================================================

def visualize_consensus_network(network, name, output_dir):
    if network.number_of_nodes() == 0:
        return

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(network, seed=42)

    nx.draw_networkx_nodes(network, pos, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(network, pos, edge_color='gray', alpha=0.6)

    plt.title(f"{name} Consensus Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_network.png"))
    plt.close()


# ===============================================================
# 5. 热图（MP × 癌种）
# ===============================================================

def visualize_mp_cancer_heatmap(mp_structures, output_dir):
    if not mp_structures:
        return

    # 所有癌种
    all_cancers = sorted({
        c
        for mp in mp_structures.values()
        for c in mp.get("cancer_types", [])
    })

    if not all_cancers:
        return

    mp_ids = sorted(mp_structures.keys(), key=str)
    mat = np.zeros((len(mp_ids), len(all_cancers)), dtype=int)

    # 填矩阵
    for i, mid in enumerate(mp_ids):
        mp = mp_structures[mid]
        node_cancers = [n.split("_")[0] for n in mp.get("nodes", [])]
        for j, cancer in enumerate(all_cancers):
            mat[i, j] = node_cancers.count(cancer)

    plt.figure(figsize=(max(6, len(all_cancers) * 0.7),
                        max(6, len(mp_ids) * 0.4)))

    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="CSM count")

    plt.xticks(range(len(all_cancers)), all_cancers, rotation=45, ha="right")
    plt.yticks(range(len(mp_ids)), [f"MP_{mid}" for mid in mp_ids])

    plt.title("MP × CancerType Participation")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "mp_cancer_heatmap.png"))
    plt.close()

    print("✔ MP × Cancer heatmap saved.")


# ===============================================================
# 6. 双部图（左边 CSM，右边 MP）
# ===============================================================

def visualize_csm_mp_bipartite(all_csms, mp_structures, output_dir):
    if not mp_structures:
        return

    G = nx.Graph()

    # 收集 CSM 信息：id -> cancer_type
    csm_info = {}
    for cancer, info in all_csms.items():
        for cid, csm in info.get("structures", {}).items():
            csm_info[cid] = {
                "cancer_type": cancer,
                "label": csm.get("annotation", {}).get("hybrid_name", cid),
            }

    # 添加 MP 节点 + 边
    for mid, mp in mp_structures.items():
        mp_node = f"MP_{mid}"
        G.add_node(mp_node, bipartite=1, kind="MP")

        for n in mp.get("nodes", []):
            # 这里假设 build_pan_cancer_consensus 里节点 id 类似 "COAD_<csm_id>"
            # 如果你的命名不一样，这里需要对应调整。
            parts = n.split("_", 1)
            csm_id = parts[1] if len(parts) > 1 else n

            if csm_id not in csm_info:
                continue

            if csm_id not in G:
                G.add_node(
                    csm_id,
                    bipartite=0,
                    kind="CSM",
                    cancer_type=csm_info[csm_id]["cancer_type"],
                )

            G.add_edge(csm_id, mp_node)

    if G.number_of_edges() == 0:
        return

    csm_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "CSM"]
    mp_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "MP"]

    # 简单手工 layout：CSM 在 x=0，MP 在 x=1
    pos = {}
    for i, n in enumerate(csm_nodes):
        pos[n] = (0.0, i)
    for i, n in enumerate(mp_nodes):
        pos[n] = (1.0, i)

    plt.figure(figsize=(12, max(6, len(mp_nodes) * 0.3)))

    nx.draw_networkx_nodes(G, pos, nodelist=csm_nodes, node_size=30, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=mp_nodes, node_size=200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    # 不画 label，避免太挤

    plt.axis("off")
    plt.title("CSM–MP Bipartite Mapping")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "csm_mp_bipartite.png"))
    plt.close()

    print("✔ CSM–MP bipartite network saved.")


# ===============================================================
# 7. 注释摘要
# ===============================================================

def visualize_annotation_summary(all_csms, mp_structures, output_dir):
    try:
        annotations = []
        for cancer, info in all_csms.items():
            for sid, sinfo in info['structures'].items():
                annotations.append(sinfo["annotation"]["hybrid_name"])
        for sid, sinfo in mp_structures.items():
            annotations.append(sinfo["annotation"]["hybrid_name"])

        from collections import Counter
        counts = Counter(annotations).most_common(20)

        labels, values = zip(*counts)

        plt.figure(figsize=(10, 8))
        plt.barh(labels, values)
        plt.title("Top Structure Annotations")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "annotation_summary.png"))
        plt.close()

        print("✔ Annotation summary saved")
    except Exception as e:
        print("❌ Error in visualize_annotation_summary:", e)


# ===============================================================
# 主接口
# ===============================================================

def enhanced_visualization(all_csms, mp_structures, intra_networks, pan_network, report=None):
    output_dir = AnalysisConfig.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n===== Begin Visualization =====")

    visualize_hierarchical_structure(all_csms, mp_structures, output_dir)

    visualize_functional_pca(all_csms, mp_structures, output_dir)

    for cancer, net in intra_networks.items():
        visualize_consensus_network(net, f"{cancer}_intra", output_dir)

    visualize_consensus_network(pan_network, "pan_cancer", output_dir)

    for cancer, info in all_csms.items():
        visualize_feature_radar(info["structures"], cancer, output_dir)

    visualize_feature_radar(mp_structures, "MP", output_dir)

    visualize_mp_cancer_heatmap(mp_structures, output_dir)

    visualize_csm_mp_bipartite(all_csms, mp_structures, output_dir)

    visualize_annotation_summary(all_csms, mp_structures, output_dir)

    print("===== Visualization Complete =====")
