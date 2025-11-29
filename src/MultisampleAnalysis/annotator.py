"""混合注释系统"""

import numpy as np
from collections import Counter, defaultdict

from analysis_utils import GeneSetManager, perform_local_enrichment


class HybridAnnotator:
    """混合注释系统 - 提供功能+语境的组合注释"""

    def __init__(self):
        self.gene_set_manager = None  # 延迟加载


    def annotate_structure(self, structure_nodes, sample_data, structure_type="CSM", all_csms=None):
        """为结构提供混合注释"""
        all_genes = set()
        cellular_contexts = []
        sample_sources = []
        precomputed_enrichments = []  # 收集所有节点的预计算富集信息

        # 处理MPs的节点
        if structure_type == "MP" and all_csms is not None:
            # 从MP节点反向找到原始CSM的节点
            csm_nodes = []
            for mp_node in structure_nodes:
                # 分割 cancer_type 和 csm_id（如 "EPM_5" -> cancer_type="EPM", csm_id="5"）
                if "_" in mp_node:
                    cancer_type, csm_id = mp_node.split("_", 1)  # 只分割一次
                    # 从all_csms中找到对应的CSM结构
                    if cancer_type in all_csms:
                        csm_struct = all_csms[cancer_type]['structures'].get(csm_id)
                        if csm_struct and 'nodes' in csm_struct:
                            csm_nodes.extend(csm_struct['nodes'])  # 收集CSM的原始节点
            # 使用CSM的原始节点进行注释
            structure_nodes = csm_nodes

        for node in structure_nodes:
            sample_id, cluster_idx = node.split("_cluster_")
            found = False

            for sample in sample_data:
                if sample["sample_id"] == sample_id:
                    cluster = sample["clusters"][int(cluster_idx)]
                    all_genes.update(cluster.get("core_genes", []))

                    # 收集细胞环境信息
                    if '3d_features' in cluster:
                        cellular_contexts.append(
                            cluster['3d_features'].get('cellular_context', {})
                        )

                    # 收集预计算的富集信息
                    precomputed_enrichments.extend(cluster.get('core_enrichment', []))

                    sample_sources.append({
                        'sample_id': sample_id,
                        'cancer_type': sample.get('cancer_type', 'unknown'),
                        'cluster_idx': cluster_idx
                    })
                    found = True
                    break

            if not found:
                print(f"警告: 未找到节点 {node} 对应的样本数据")

        # 1. 描述性注释
        descriptive_annotation = self._get_descriptive_annotation(
            list(all_genes),
            precomputed_enrichments  # 传入预计算富集信息
        )

        # 2. 语境注释
        context_annotation = self._get_context_annotation(cellular_contexts)

        # 3. 组合注释
        hybrid_name = self._combine_annotations(descriptive_annotation, context_annotation)

        return {
            'hybrid_name': hybrid_name,
            'descriptive': descriptive_annotation,
            'contextual': context_annotation,
            'gene_count': len(all_genes),
            'structure_type': structure_type,
            'sample_sources': sample_sources,
            'node_count': len(structure_nodes)
        }

    def _get_descriptive_annotation(self, genes, precomputed_enrichments):
        """基于预计算富集信息生成描述性注释 - 修复版"""
        # 优先使用预计算的富集信息
        if precomputed_enrichments:
            # 聚合所有预计算的富集术语并去重
            unique_terms = {}
            for term in precomputed_enrichments:
                if not isinstance(term, dict) or 'term' not in term:
                    continue

                term_key = f"{term['term']}_{term.get('source', 'unknown')}"
                current_pvalue = term.get('adj_pvalue', 1.0)

                if term_key not in unique_terms or current_pvalue < unique_terms[term_key]['adj_pvalue']:
                    unique_terms[term_key] = term

            # 按p值排序，取最显著的术语
            sorted_terms = sorted(unique_terms.values(), key=lambda x: x.get('adj_pvalue', 1.0))

            if sorted_terms:
                best_term = sorted_terms[0]
                simplified_term = self._simplify_term_name(best_term["term"])

                return {
                    "primary_function": simplified_term,
                    "original_term": best_term["term"],
                    "p_value": best_term.get("adj_pvalue", 1.0),
                    "library": best_term.get("source", "unknown"),
                    "enriched_genes": best_term.get("gene_count", len(genes)),
                    "enrichment_score": -np.log10(max(best_term.get("adj_pvalue", 1.0), 1e-10)),
                    "source": "precomputed"
                }

        # 如果预计算信息不可用，基于基因列表生成描述性注释
        if len(genes) >= 5:
            # 简单的基因功能推断（可以根据需要扩展）
            gene_keywords = {
                'immune': ['CD8', 'CD4', 'CD3', 'IL', 'IFN', 'TNF', 'B2M'],
                'cell_cycle': ['CDK', 'CCN', 'PCNA', 'MKI67', 'TOP2A'],
                'apoptosis': ['CASP', 'BCL', 'BAX', 'BAD'],
                'metabolism': ['GAPDH', 'LDH', 'PKM', 'ACLY'],
                'extracellular_matrix': ['COL', 'FN1', 'LAM', 'MMP'],
                'neural': ['NEFL', 'NEFM', 'SNAP', 'SYN']
            }

            detected_functions = []
            for func, markers in gene_keywords.items():
                matches = sum(1 for gene in genes if any(marker in gene.upper() for marker in markers))
                if matches >= 2:  # 至少有2个相关基因
                    detected_functions.append(func)

            if detected_functions:
                primary_func = detected_functions[0].replace('_', ' ').title()
            else:
                primary_func = "Mixed Functionality"

            return {
                "primary_function": primary_func,
                "details": f"Based on {len(genes)} genes",
                "p_value": 1.0,
                "source": "gene_based"
            }

        return {
            "primary_function": "Unknown Function",
            "details": "Insufficient data for annotation",
            "p_value": 1.0,
            "source": "fallback"
        }

    def _get_context_annotation(self, cellular_contexts):
        """获取语境注释 - 修复空值处理"""
        if not cellular_contexts:
            return {
                "dominant_context": "Unknown",
                "cell_composition": {},
                "context_diversity": 0
            }

        # 过滤掉非字典类型的上下文
        valid_contexts = [ctx for ctx in cellular_contexts if isinstance(ctx, dict)]

        if not valid_contexts:
            return {
                "dominant_context": "Unknown",
                "cell_composition": {},
                "context_diversity": 0
            }

        # 聚合细胞类型比例
        aggregated_context = defaultdict(list)
        for context in valid_contexts:
            for cell_type, proportion in context.items():
                aggregated_context[cell_type].append(proportion)

        # 计算平均比例
        avg_context = {}
        for cell_type, proportions in aggregated_context.items():
            avg_context[cell_type] = np.mean(proportions)

        # 确定主导细胞环境
        dominant_context = "Unknown"
        context_diversity = len(avg_context)

        if avg_context:
            # 找到主导细胞类型
            dominant_cell_type, max_proportion = max(avg_context.items(), key=lambda x: x[1])

            # 分类语境
            if max_proportion > 0.6:
                dominant_context = f"{dominant_cell_type}-Dominant"
            elif max_proportion > 0.3:
                sorted_types = sorted(avg_context.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_types) > 1 and sorted_types[1][1] > 0.2:
                    dominant_context = f"{sorted_types[0][0]}/{sorted_types[1][0]}-Mixed"
                else:
                    dominant_context = f"{dominant_cell_type}-Primary"
            else:
                top_types = [t[0] for t in sorted(avg_context.items(), key=lambda x: x[1], reverse=True)[:3]]
                dominant_context = f"{'/'.join(top_types)}-Mixed"

        return {
            "dominant_context": dominant_context,
            "cell_composition": dict(sorted(avg_context.items(), key=lambda x: x[1], reverse=True)),
            "context_diversity": context_diversity,
            "dominant_cell_proportion": max_proportion if avg_context else 0
        }

    def _simplify_term_name(self, term_name):
        """简化富集项名称"""
        # 移除冗余信息
        simplified = term_name

        # 移除GO编号
        if "GO:" in simplified and "~" in simplified:
            simplified = simplified.split("~")[-1]

        # 移除括号内容
        if "(" in simplified and ")" in simplified:
            start = simplified.find("(")
            end = simplified.find(")")
            if end > start:
                simplified = simplified[:start] + simplified[end + 1:]

        # 限制长度
        if len(simplified) > 40:
            simplified = simplified[:37] + "..."

        return simplified.strip()

    def _combine_annotations(self, descriptive, contextual):
        """组合描述性和语境性注释"""
        primary_func = descriptive.get("primary_function", "Unknown")
        dominant_context = contextual.get("dominant_context", "Unknown")

        # 进一步简化功能名称
        if len(primary_func) > 25:
            # 尝试按逗号或分号分割取第一部分
            for separator in [",", ";", "-"]:
                if separator in primary_func:
                    parts = primary_func.split(separator)
                    if len(parts[0]) <= 25:
                        primary_func = parts[0].strip()
                        break

        # 最终长度限制
        if len(primary_func) > 25:
            primary_func = primary_func[:22] + "..."

        return f"{primary_func} [{dominant_context}]"

    def batch_annotate_structures(self, structures_dict, sample_data, structure_type="CSM", all_csms=None):
        """批量注释多个结构"""
        annotated_structures = {}
        for struct_id, structure_info in structures_dict.items():
            nodes = structure_info.get('nodes', [])
            annotation = self.annotate_structure(
                nodes,
                sample_data,
                structure_type,
                all_csms=all_csms  # 传递给annotate_structure
            )
            structure_info['annotation'] = annotation
            annotated_structures[struct_id] = structure_info
        return annotated_structures