"""分层共识网络构建"""

import networkx as nx
import numpy as np
import json
from collections import defaultdict
from community import best_partition, community_louvain
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from analysis_config import AnalysisConfig
from similarity import MultiModalSimilarity
from feature_engine import ThreeDFeatureEngine


class HierarchicalConsensusBuilder:
    """分层共识网络构建器"""

    def __init__(self):
        self.consensus_params = AnalysisConfig.consensus_params
        self.similarity_calculator = MultiModalSimilarity()

    def build_intra_cancer_consensus(self, cancer_samples):
        """构建癌症内部共识网络 (CSMs)"""
        print(f"构建癌症内部共识网络，样本数: {len(cancer_samples)}")

        # 为所有聚类构建三维特征
        all_clusters = self._extract_clusters_with_features(cancer_samples)

        if not all_clusters:
            return {}, nx.Graph()

        # 把聚类列表转成 {cluster_id: cluster_info} 字典
        cluster_dict = {}
        for cluster in all_clusters:
            cluster_id = cluster.get("cluster_id")
            if cluster_id:
                cluster_dict[cluster_id] = cluster

        # 构建相似度网络
        G = nx.Graph()

        # 添加节点
        for cluster_id, cluster_info in cluster_dict.items():
            G.add_node(cluster_id, **cluster_info)

        # 计算相似度并添加边
        cluster_ids = list(cluster_dict.keys())
        edges_added = 0

        for i in range(len(cluster_ids)):
            cluster_i = cluster_ids[i]
            features_i = cluster_dict[cluster_i]['3d_features']
            sample_i = cluster_i.split('_cluster_')[0]

            for j in range(i + 1, len(cluster_ids)):
                cluster_j = cluster_ids[j]
                features_j = cluster_dict[cluster_j]['3d_features']
                sample_j = cluster_j.split('_cluster_')[0]

                # 只计算跨样本的连接
                if sample_i != sample_j:
                    # 计算综合相似度
                    similarity = self.similarity_calculator.calc_comprehensive_similarity(
                        features_i, features_j, mode="CSMs"
                    )

                    # 应用癌症内部阈值
                    if similarity >= self.consensus_params['intra_cancer_threshold']:
                        G.add_edge(cluster_i, cluster_j, weight=similarity)
                        edges_added += 1

        print(f"癌症内部网络: {G.number_of_nodes()} 节点, {edges_added} 边")

        # 社区检测识别CSMs
        cancer_type = cancer_samples[0].get('cancer_type', 'Unknown') if cancer_samples else 'Unknown'
        csm_clusters = self._detect_robust_structures(G, cancer_samples, cancer_type)

        return csm_clusters, G

    def build_pan_cancer_consensus(self, all_csms):
        """构建泛癌共识网络 (MPs) - 修复特征统一问题"""
        print("构建泛癌共识网络...")

        if not all_csms:
            return {}, nx.Graph()

        # 合并所有CSMs
        all_structures = {}
        for cancer_type, csms_info in all_csms.items():
            structures = csms_info.get('structures', {})
            for csm_id, csm_info in structures.items():
                pan_id = f"{cancer_type}_{csm_id}"
                all_structures[pan_id] = csm_info

        if not all_structures:
            print("警告：没有可合并的CSM结构")
            return {}, nx.Graph()

        # 构建泛癌网络
        G = nx.Graph()
        edges_added = 0

        # 添加节点 - 修复特征统一问题
        for struct_id, struct_info in all_structures.items():
            if not isinstance(struct_info, dict):
                print(f"跳过无效结构 {struct_id}（非字典类型）")
                continue

            # 统一处理3D特征
            node_attributes = struct_info.copy()
            raw_3d_features = struct_info.get('3d_features', {})

            # 修复：统一特征表示，确保是字典
            if isinstance(raw_3d_features, list) and raw_3d_features:
                # 如果是列表，取第一个元素（应该是字典）
                node_attributes['3d_features'] = raw_3d_features[0]
            elif isinstance(raw_3d_features, dict):
                # 已经是字典，直接使用
                node_attributes['3d_features'] = raw_3d_features
            else:
                # 其他情况设为空字典
                print(f"警告：结构 {struct_id} 的3D特征格式异常: {type(raw_3d_features)}")
                node_attributes['3d_features'] = {}

            G.add_node(struct_id, **node_attributes)

        # 计算跨癌症相似度
        struct_ids = list(all_structures.keys())
        total_pairs = len(struct_ids) * (len(struct_ids) - 1) // 2
        print(f"计算 {total_pairs} 对跨癌症CSM的相似度...")

        valid_pairs = 0
        similarity_values = []

        for i in range(len(struct_ids)):
            struct_i = struct_ids[i]
            node_data_i = G.nodes[struct_i]
            features_i = node_data_i.get('3d_features', {})

            if not isinstance(features_i, dict) or not features_i:
                continue

            cancer_i = struct_i.split('_')[0]

            for j in range(i + 1, len(struct_ids)):
                struct_j = struct_ids[j]
                node_data_j = G.nodes[struct_j]
                features_j = node_data_j.get('3d_features', {})

                if not isinstance(features_j, dict) or not features_j:
                    continue

                cancer_j = struct_j.split('_')[0]

                # 只计算不同癌症类型间的相似度
                if cancer_i != cancer_j:
                    valid_pairs += 1
                    try:
                        similarity = self.similarity_calculator.calc_comprehensive_similarity(
                            features_i, features_j, mode="MPs"
                        )
                        similarity_values.append(similarity)

                        # 应用泛癌阈值
                        threshold = self.consensus_params.get('pan_cancer_threshold', 0.01)
                        if similarity >= threshold:
                            G.add_edge(struct_i, struct_j, weight=similarity)
                            edges_added += 1

                    except Exception as e:
                        print(f"计算 {struct_i} 和 {struct_j} 相似度时出错: {str(e)}")
                        continue

        # 分析相似度分布
        if similarity_values:
            similarity_array = np.array(similarity_values)
            print(f"相似度统计: 均值={similarity_array.mean():.3f}, 标准差={similarity_array.std():.3f}")
            print(f"相似度范围: [{similarity_array.min():.3f}, {similarity_array.max():.3f}]")
            print(
                f"阈值={self.consensus_params.get('pan_cancer_threshold', 0.01):.3f}时，{np.sum(similarity_array >= self.consensus_params.get('pan_cancer_threshold', 0.01))}/{len(similarity_array)} 对满足条件")

        print(f"有效特征对: {valid_pairs}, 满足阈值的边: {edges_added}")
        print(f"泛癌网络: {G.number_of_nodes()} 节点, {edges_added} 边")

        # 识别泛癌元程序 (MPs)
        mp_clusters = self._detect_pan_cancer_structures(G)

        return mp_clusters, G

    def _extract_clusters_with_features(self, cancer_samples):
        """提取癌症样本的聚类及特征"""
        all_clusters = []

        for sample in cancer_samples:
            sample_id = sample["sample_id"]
            print(f"   处理样本: {sample_id}")

            # 初始化特征工程
            feature_engine = ThreeDFeatureEngine(sample)
            # 构建该样本所有聚类的三维特征
            sample_cluster_features = feature_engine.build_all_clusters_3d_features()

            # 把分散的特征整合到该字段下
            for cluster_feat in sample_cluster_features:
                # 整合三个维度特征为 '3d_features'
                cluster_feat['3d_features'] = {
                    'transcript': cluster_feat['transcript_feature'],
                    'cell_context': cluster_feat['cell_context_feature'],
                    'functional': cluster_feat['functional_feature']
                }
                all_clusters.append(cluster_feat)

        print(f"✅ 提取到 {len(all_clusters)} 个聚类特征")
        return all_clusters

    def _detect_robust_structures(self, graph, samples, cancer_type):
        """检测稳健的癌症内部共识结构(CSMs) - 使用聚合特征"""
        sample_count = len(samples)
        min_coverage = max(2, int(sample_count * self.consensus_params.get('min_sample_coverage', 0.4)))
        robust_communities = {}

        try:
            # Louvain社区检测
            partition = community_louvain.best_partition(graph, resolution=self.consensus_params.get('resolution', 0.5))
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
        except Exception as e:
            print(f"⚠️ {cancer_type} 社区检测失败，降级为连通组件: {str(e)[:50]}")
            connected_components = list(nx.connected_components(graph))
            communities = {i: list(comp) for i, comp in enumerate(connected_components)}

        # 遍历社区筛选稳健结构
        for comm_id, nodes in communities.items():
            # 过滤过小结构
            if len(nodes) < self.consensus_params.get('min_structure_size', 2):
                continue

            covered_samples = set()
            community_features_3d = []  # 收集所有节点的3D特征

            # 提取样本覆盖+3D特征
            for node in nodes:
                sample_id = node.split('_cluster_')[0] if '_cluster_' in node else node
                covered_samples.add(sample_id)

                # 提取3D特征
                if node in graph.nodes:
                    node_data = graph.nodes[node]
                    node_3d = node_data.get('3d_features', {})
                    if isinstance(node_3d, dict) and node_3d:
                        community_features_3d.append(node_3d)

            # 过滤覆盖率不足的结构
            if len(covered_samples) < min_coverage:
                continue

            coverage_ratio = len(covered_samples) / sample_count
            csm_id = f"{cancer_type}_{comm_id}"

            # 🎯 使用聚合特征而不是第一个特征
            aggregated_3d = self._aggregate_csm_features(community_features_3d)

            # 保存CSM
            robust_communities[csm_id] = {
                'id': csm_id,
                'cancer_type': cancer_type,
                'nodes': nodes,
                'size': len(nodes),
                'sample_coverage': len(covered_samples),
                'coverage_ratio': coverage_ratio,
                'sample_ids': list(covered_samples),
                '3d_features': aggregated_3d,  # 🎯 使用聚合特征
                'node_features': community_features_3d,  # 保留原始节点特征用于调试
                'has_valid_3d': bool(aggregated_3d)
            }

        print(f"✅ {cancer_type}: 发现 {len(robust_communities)} 个稳健CSMs")
        return robust_communities

    def _aggregate_csm_features(self, community_features_3d):
        """聚合CSM内所有聚类的三维特征"""
        if not community_features_3d:
            return {}

        # 1. 聚合转录特征 - 合并所有特异性基因，按平均SpecScore排序
        all_transcript_genes = {}
        gene_counts = {}

        for feat in community_features_3d:
            transcript_feat = feat.get('transcript', {})
            specific_genes = transcript_feat.get('specific_genes', [])

            for gene, score in specific_genes:
                if gene not in all_transcript_genes:
                    all_transcript_genes[gene] = 0.0
                    gene_counts[gene] = 0
                all_transcript_genes[gene] += score
                gene_counts[gene] += 1

        # 计算平均SpecScore
        avg_transcript_genes = [
            (gene, all_transcript_genes[gene] / gene_counts[gene])
            for gene in all_transcript_genes
        ]
        # 按平均SpecScore降序排列
        avg_transcript_genes.sort(key=lambda x: x[1], reverse=True)

        # 2. 聚合细胞微环境特征 - 计算平均细胞类型比例
        all_cell_types = {}
        cell_type_counts = {}

        for feat in community_features_3d:
            cell_context = feat.get('cell_context', {})
            cell_proportions = cell_context.get('cell_type_proportions', {})

            for cell_type, proportion in cell_proportions.items():
                if cell_type not in all_cell_types:
                    all_cell_types[cell_type] = 0.0
                    cell_type_counts[cell_type] = 0
                all_cell_types[cell_type] += proportion
                cell_type_counts[cell_type] += 1

        # 计算平均比例
        avg_cell_proportions = {
            cell_type: all_cell_types[cell_type] / cell_type_counts[cell_type]
            for cell_type in all_cell_types
        }

        # 3. 聚合功能特征 - 合并所有富集通路，按平均p值排序
        all_pathways = {}
        pathway_counts = {}

        for feat in community_features_3d:
            functional_feat = feat.get('functional', {})
            pathways = functional_feat.get('enriched_pathways', [])

            for pathway in pathways:
                term = pathway.get('term')
                adj_p = pathway.get('adj_pvalue', 1.0)

                if term not in all_pathways:
                    all_pathways[term] = {
                        'adj_pvalue_sum': 0.0,
                        'count': 0,
                        'source': pathway.get('source', 'unknown')
                    }
                all_pathways[term]['adj_pvalue_sum'] += adj_p
                all_pathways[term]['count'] += 1

        # 计算平均p值，按显著性排序
        avg_pathways = [
            {
                'term': term,
                'adj_pvalue': all_pathways[term]['adj_pvalue_sum'] / all_pathways[term]['count'],
                'source': all_pathways[term]['source']
            }
            for term in all_pathways
        ]
        avg_pathways.sort(key=lambda x: x['adj_pvalue'])

        return {
            'transcript': {
                'specific_genes': avg_transcript_genes,
                'spec_score_mean': np.mean(
                    [score for _, score in avg_transcript_genes]) if avg_transcript_genes else 0.0,
                'gene_count': len(avg_transcript_genes)
            },
            'cell_context': {
                'cell_type_proportions': avg_cell_proportions,
                'dominance': max(avg_cell_proportions.values()) if avg_cell_proportions else 0.0,
                'evenness': 1 - max(avg_cell_proportions.values()) if avg_cell_proportions else 0.0,
                'cell_type_count': len(avg_cell_proportions)
            },
            'functional': {
                'enriched_pathways': avg_pathways,
                'pathway_count': len(avg_pathways),
                'term_standardized': True
            }
        }

    def _detect_pan_cancer_structures(self, graph):
        """检测泛癌结构 (MPs) - 修复距离矩阵问题"""
        if graph.number_of_nodes() == 0:
            return {}

        # 当边数很少时，使用连通组件而不是层次聚类
        if graph.number_of_edges() < 5:
            print("边数过少，使用连通组件检测MPs")
            mp_clusters = {}
            for i, component in enumerate(nx.connected_components(graph)):
                nodes = list(component)
                cancer_types = [node.split('_')[0] for node in nodes]
                unique_cancer_types = list(set(cancer_types))
                mp_clusters[i] = {
                    'nodes': nodes,
                    'cancer_types': unique_cancer_types,
                    'cancer_count': len(unique_cancer_types)
                }
            return mp_clusters

        # 使用层次聚类识别MPs
        try:
            # 构建距离矩阵
            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_nodes < 2:
                mp_clusters = {
                    0: {
                        'nodes': nodes,
                        'cancer_types': [node.split('_')[0] for node in nodes],
                        'cancer_count': len(set(node.split('_')[0] for node in nodes))
                    }
                }
                return mp_clusters

            # 计算距离矩阵（1 - 相似度）
            dist_matrix = np.zeros((n_nodes, n_nodes))  # 初始化为零
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if graph.has_edge(nodes[i], nodes[j]):
                        similarity = graph[nodes[i]][nodes[j]]['weight']
                        dist_matrix[i, j] = 1 - similarity
                        dist_matrix[j, i] = 1 - similarity
                    else:
                        dist_matrix[i, j] = 1.0  # 没有连接时距离为1
                        dist_matrix[j, i] = 1.0

            # 确保对角线为零
            np.fill_diagonal(dist_matrix, 0.0)

            # 检查距离矩阵是否有效
            if np.any(np.isnan(dist_matrix)) or np.any(np.isinf(dist_matrix)):
                print("距离矩阵包含无效值，使用连通组件")
                raise ValueError("Invalid distance matrix")

            # 层次聚类
            condensed_dist = squareform(dist_matrix)
            linkage_matrix = linkage(condensed_dist, method='ward')

            # 动态确定聚类数量
            if n_nodes <= 5:
                t_criterion = 0.8
            else:
                t_criterion = 0.5

            clusters = fcluster(linkage_matrix, t=t_criterion, criterion='distance')

            # 按聚类分组
            mp_clusters = defaultdict(list)
            for node_idx, cluster_id in enumerate(clusters):
                mp_clusters[cluster_id - 1].append(nodes[node_idx])

            # 转换为最终格式
            final_mp_clusters = {}
            for mp_id, nodes_list in mp_clusters.items():
                cancer_types = [node.split('_')[0] for node in nodes_list]
                unique_cancer_types = list(set(cancer_types))
                cancer_count = len(unique_cancer_types)

                final_mp_clusters[mp_id] = {
                    'nodes': nodes_list,
                    'cancer_types': unique_cancer_types,
                    'cancer_count': cancer_count
                }

            print(f"层次聚类成功: 将 {n_nodes} 个节点分为 {len(final_mp_clusters)} 个MPs")
            return final_mp_clusters

        except Exception as e:
            print(f"层次聚类失败: {str(e)}，使用连通组件")
            # 失败时使用连通组件作为备选
            mp_clusters = {}
            for i, component in enumerate(nx.connected_components(graph)):
                nodes = list(component)
                cancer_types = [node.split('_')[0] for node in nodes]
                unique_cancer_types = list(set(cancer_types))
                mp_clusters[i] = {
                    'nodes': nodes,
                    'cancer_types': unique_cancer_types,
                    'cancer_count': len(unique_cancer_types)
                }
            return mp_clusters