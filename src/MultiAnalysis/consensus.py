"""分层共识网络构建（基于2D特征：细胞组成 + 功能轴）"""

import networkx as nx
import numpy as np
from collections import defaultdict
from community import community_louvain
import igraph as ig
import leidenalg
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from analysis_config import AnalysisConfig
from similarity import MultiModalSimilarity
from feature_engine import build_clusters_for_samples


class HierarchicalConsensusBuilder:
    """分层共识网络构建器（CSM → MP）"""

    def __init__(self):
        self.consensus_params = AnalysisConfig.consensus_params
        self.similarity = MultiModalSimilarity()



    # ============================================================
    # 🔵 1. 构建癌种内部 CSMs
    # ============================================================
    def build_intra_cancer_consensus(self, cancer_samples):
        print(f"构建癌症内部共识网络 (CSMs)，样本数={len(cancer_samples)}")

        all_clusters = self._extract_clusters_with_features(cancer_samples)
        if not all_clusters:
            return {}, nx.Graph()

        G = nx.Graph()

        # 添加节点
        for c in all_clusters:
            G.add_node(c["cluster_id"], **c)

        ids = list(G.nodes())
        edges_added = 0

        for i in range(len(ids)):
            ci = ids[i]
            fi = G.nodes[ci]["3d_features"]
            si = ci.split("_cluster_")[0]

            for j in range(i + 1, len(ids)):
                cj = ids[j]
                sj = cj.split("_cluster_")[0]

                if si == sj:
                    continue  # 必须跨样本

                fj = G.nodes[cj]["3d_features"]

                sim = self.similarity.calc_comprehensive_similarity(fi, fj, mode="CSMs")

                if sim >= self.consensus_params["intra_cancer_threshold"]:
                    G.add_edge(ci, cj, weight=sim)
                    edges_added += 1

        print(f"癌症内部网络: {G.number_of_nodes()} 个节点, {edges_added} 条边")

        cancer_type = cancer_samples[0]["cancer_type"]
        csm_clusters = self._detect_csm_structures(G, cancer_samples, cancer_type)

        return csm_clusters, G

    # ============================================================
    # 🔵 2. 提取特征（统一字段）
    # ============================================================
    def _extract_clusters_with_features(self, cancer_samples):
        all_clusters = build_clusters_for_samples(cancer_samples)
        print(f"提取到 {len(all_clusters)} 个聚类特征")
        return all_clusters

    # ============================================================
    # 🔵 3. Leiden 社区划分
    # ============================================================
    def _leiden_partition(self, graph: nx.Graph):
        """使用 Leiden 算法做社区划分"""

        # networkx 节点 -> 连续整数 id
        node_list = list(graph.nodes())
        idx_map = {n: i for i, n in enumerate(node_list)}
        inv_map = {i: n for n, i in idx_map.items()}

        # 边列表
        edges = [(idx_map[u], idx_map[v]) for u, v in graph.edges()]

        g_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)

        # 边权（如果有）
        if graph.number_of_edges() > 0:
            first_edge = next(iter(graph.edges(data=True)))
            if "weight" in first_edge[2]:
                weights = [
                    graph[u][v].get("weight", 1.0)
                    for u, v in graph.edges()
                ]
                g_ig.es["weight"] = weights
                part = leidenalg.find_partition(
                    g_ig,
                    leidenalg.RBConfigurationVertexPartition,
                    weights="weight",
                )
            else:
                part = leidenalg.find_partition(
                    g_ig,
                    leidenalg.RBConfigurationVertexPartition,
                )
        else:
            return {n: 0 for n in node_list}

        partition = {}
        for cid, cluster in enumerate(part):
            for vid in cluster:
                node = inv_map[vid]
                partition[node] = cid

        return partition

    # ============================================================
    # 🔵 4. CSM 结构检测
    # ============================================================
    def _detect_csm_structures(self, graph, samples, cancer_type):
        sample_ids = {s["sample_id"] for s in samples}
        min_cov = max(2, int(len(sample_ids) * self.consensus_params["min_sample_coverage"]))

        # 社区检测
        try:
            # partition = community_louvain.best_partition(graph)
            partition = self._leiden_partition(graph)
        except Exception as e:
            print(f"[WARN] 社区检测失败，使用连通组件。错误：{e}")
            partition = {n: i for i, comp in enumerate(nx.connected_components(graph)) for n in comp}

        # 聚合社区
        comms = defaultdict(list)
        for node, cid in partition.items():
            comms[cid].append(node)

        final = {}

        for cid, nodes in comms.items():
            covered = {n.split("_cluster_")[0] for n in nodes}
            if len(covered) < min_cov:
                continue

            csm_feats = [graph.nodes[n]["3d_features"] for n in nodes]
            agg = self._aggregate_features(csm_feats)

            csm_id = f"{cancer_type}_{cid}"
            final[csm_id] = {
                "id": csm_id,
                "nodes": nodes,
                "sample_coverage": len(covered),
                "coverage_ratio": len(covered) / len(sample_ids),
                "3d_features": agg,
                "cancer_type": cancer_type,
            }

        print(f"{cancer_type} 识别到 {len(final)} 个 CSMs")
        return final

    # ============================================================
    # 🔵 5. 特征聚合（CSM & MP 共用）
    # ============================================================
    def _aggregate_features(self, feats):
        """对多个 cluster 的特征做聚合，用于 CSM 和 MP"""

        # -------- 细胞组成聚合 --------
        cell_sum = defaultdict(float)
        cell_cnt = defaultdict(int)

        for f in feats:
            for k, v in f["cell_context"]["cell_type_proportions"].items():
                cell_sum[k] += v
                cell_cnt[k] += 1

        avg_cell = {k: cell_sum[k] / cell_cnt[k] for k in cell_sum}

        # -------- 功能轴聚合（最关键） --------
        axis_sum = defaultdict(float)
        axis_cnt = defaultdict(int)

        for f in feats:
            for ax, v in f["functional"]["axis_scores"].items():
                axis_sum[ax] += v
                axis_cnt[ax] += 1

        avg_axis = {ax: axis_sum[ax] / axis_cnt[ax] for ax in axis_sum}

        # -------- transcript 仅作注释用途 --------
        transcripts = []
        for f in feats:
            transcripts.extend(f["transcript"]["specific_genes"])

        return {
            "cell_context": {
                "cell_type_proportions": avg_cell,
                "dominance": max(avg_cell.values()) if avg_cell else 0.0,
                "evenness": 1 - max(avg_cell.values()) if avg_cell else 0.0,
            },
            "functional": {
                "axis_scores": avg_axis,
            },
            "transcript": {
                "specific_genes": transcripts[:50],  # 可选
            },
        }

    # ============================================================
    # 🔵 6. 泛癌 MP 构建
    # ============================================================
    def build_pan_cancer_consensus(self, all_csms):
        print("构建泛癌共识网络 (MPs)...")

        # 合并所有 CSM 节点
        structs = {}
        for cancer, info in all_csms.items():
            for cid, item in info.get("structures", {}).items():
                structs[f"{cancer}_{cid}"] = item

        G = nx.Graph()

        for sid, s in structs.items():
            G.add_node(sid, **s)

        ids = list(G.nodes())
        edges = 0

        # 计算跨癌种相似度
        for i in range(len(ids)):
            fi = G.nodes[ids[i]]["3d_features"]
            ci = ids[i].split("_")[0]

            for j in range(i + 1, len(ids)):
                cj = ids[j].split("_")[0]
                if ci == cj:
                    continue

                fj = G.nodes[ids[j]]["3d_features"]
                sim = self.similarity.calc_comprehensive_similarity(fi, fj, mode="MPs")

                if sim >= self.consensus_params["pan_cancer_threshold"]:
                    G.add_edge(ids[i], ids[j], weight=sim)
                    edges += 1

        print(f"MP 网络: {G.number_of_nodes()} 节点, {edges} 条边")

        return self._detect_mp_structures(G), G

    # ============================================================
    # 🔵 7. MP 检测（层次聚类 or 连通组件）
    # ============================================================
    def _detect_mp_structures(self, graph):
        if graph.number_of_nodes() == 0:
            return {}

        min_size = self.consensus_params.get("min_mp_size", 2)
        min_ct = self.consensus_params.get("min_mp_cancer_types", 2)

        # ---------- 小边数：直接用连通组件 ----------
        if graph.number_of_edges() < 5:
            print("边过少，使用连通组件作为 MPs")
            mp = {}
            idx = 0
            for comp in nx.connected_components(graph):
                nodes = list(comp)
                cancers = {n.split("_")[0] for n in nodes}

                # 阈值过滤
                if len(nodes) < min_size or len(cancers) < min_ct:
                    continue

                # 聚合 3D 特征（CSM 的 3d_features 已经在 graph.nodes 里）
                feats = [
                    graph.nodes[n].get("3d_features")
                    for n in nodes
                    if "3d_features" in graph.nodes[n]
                ]
                agg = self._aggregate_features(feats) if feats else {}

                mp[idx] = {
                    "nodes": nodes,
                    "cancer_types": list(cancers),
                    "cancer_count": len(cancers),
                    "3d_features": agg,  # ★ 给 MP 自己一份 3D 特征
                }
                idx += 1
            print(f"识别到 {len(mp)} 个 MPs（连通组件模式）")
            return mp

        # ---------- 正常情况：层次聚类 ----------
        nodes = list(graph.nodes())
        n = len(nodes)

        # 转换为距离矩阵：d = 1 - sim
        dist = np.ones((n, n))
        np.fill_diagonal(dist, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                if graph.has_edge(nodes[i], nodes[j]):
                    w = graph[nodes[i]][nodes[j]].get("weight", 0.0)
                    dist[i, j] = dist[j, i] = max(0.0, 1.0 - w)

        # linkage + fcluster
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=0.5, criterion="distance")

        mp_groups = defaultdict(list)
        for idx, lab in enumerate(labels):
            mp_groups[lab].append(nodes[idx])

        result = {}
        idx = 0
        for _, group in mp_groups.items():
            cancers = {n.split("_")[0] for n in group}

            # 阈值过滤
            if len(group) < min_size or len(cancers) < min_ct:
                continue

            feats = [
                graph.nodes[n].get("3d_features")
                for n in group
                if "3d_features" in graph.nodes[n]
            ]
            agg = self._aggregate_features(feats) if feats else {}

            result[idx] = {
                "nodes": group,
                "cancer_types": list(cancers),
                "cancer_count": len(cancers),
                "3d_features": agg,
            }
            idx += 1

        print(f"识别到 {len(result)} 个 MPs")
        return result
