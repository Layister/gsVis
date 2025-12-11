"""特征工程：以细胞组成 + 功能轴为核心（转录仅用于注释）"""

import numpy as np
from analysis_utils import GeneSetManager
from analysis_config import AnalysisConfig


class FeatureEngine:
    """特征工程类（相似度基于 2D：细胞组成 + 功能轴）"""

    def __init__(self, sample_data):
        self.sample_id = sample_data["sample_id"]
        self.cancer_type = sample_data["cancer_type"]
        self.domain_features_norm = sample_data["domain_features_norm"]
        self.cell_type_data_norm = sample_data["cell_type_data_norm"]
        self.clusters = sample_data["clusters"]

        # 细胞组成分组 + 功能轴基因集
        self.all_cell_types = sorted({ct for d in self.cell_type_data_norm.values() for ct in d.keys()})
        self.axis_genesets = GeneSetManager.get_axis_genesets()


    # -------------------------------------------------------
    # 1. 转录特征（仅用于注释，不进入相似度）
    # -------------------------------------------------------
    def _get_cluster_domains_expr(self, cluster_domains):
        """汇总聚类内所有 domain 的表达（仅用于注释）"""
        domain_expr = {}

        for domain_id in cluster_domains:
            expr = (
                self.domain_features_norm.get(domain_id, {})
                .get("gene_avg_expr_norm", {})
            )
            for gene, val in expr.items():
                domain_expr.setdefault(gene, []).append(val)

        return {g: float(np.mean(v)) for g, v in domain_expr.items()}

    def get_transcript_feature(self, cluster):
        """计算 SpecScore，用于注释；不参与相似度"""
        cluster_domains = cluster.get("domains", [])
        if not cluster_domains:
            return {"specific_genes": [], "spec_score_mean": 0.0, "gene_count": 0}

        avg_cluster = self._get_cluster_domains_expr(cluster_domains)

        # 外部表达用所有 domain 的平均
        all_domains = list(self.domain_features_norm.keys())
        non_cluster_domains = [d for d in all_domains if d not in cluster_domains]
        avg_non_cluster = self._get_cluster_domains_expr(non_cluster_domains)

        spec = {
            g: avg_cluster.get(g, 0) - avg_non_cluster.get(g, 0)
            for g in avg_cluster
        }

        specific_genes = [(g, s) for g, s in spec.items() if s > 0]
        spec_score_mean = (
            float(np.mean([s for _, s in specific_genes])) if specific_genes else 0.0
        )

        return {
            "specific_genes": specific_genes,
            "spec_score_mean": round(spec_score_mean, 4),
            "gene_count": len(specific_genes),
        }

    # -------------------------------------------------------
    # 2. 细胞组成特征（相似度主通道）
    # -------------------------------------------------------
    def get_cell_context_feature(self, cluster):
        """使用真实细胞类型构建聚类的细胞组成向量"""

        cluster_domains = cluster.get("domains", [])
        if not cluster_domains:
            return {
                "cell_type_proportions": {},
                "dominance": 0.0,
                "evenness": 0.0,
                "cell_type_count": 0,
            }

        # 只在 cluster 内部做统计，不再依赖预先定义的分组
        domain_count = 0  # 有有效 celltype 预测的 spot 数
        cell_count = {}   # ct -> 出现的 spot 数（通过阈值之后）
        cell_conf_sum = {}  # ct -> 置信度和（通过阈值之后）

        for domain_id in cluster_domains:
            props = self.cell_type_data_norm.get(domain_id, {})
            if not props:
                continue

            domain_count += 1

            for ct, score in props.items():
                # 过滤低置信度（或低占比）细胞类型
                if score < AnalysisConfig.conf_threshold:
                    continue

                cell_count[ct] = cell_count.get(ct, 0) + 1
                cell_conf_sum[ct] = cell_conf_sum.get(ct, 0.0) + float(score)

        # 如果这个 cluster 里面没有任何通过阈值的细胞类型
        if domain_count == 0 or not cell_count:
            return {
                "cell_type_proportions": {},
                "dominance": 0.0,
                "evenness": 0.0,
                "cell_type_count": 0,
            }

        # 计算每种细胞类型的 prevalence、mean_conf，并组合成一个 score
        cell_scores = {}
        for ct, cnt in cell_count.items():
            prevalence = cnt / domain_count  # 有该 ct 的 spot 占 cluster 有效 spot 比例
            mean_conf = cell_conf_sum[ct] / cnt
            score = prevalence * mean_conf
            if score > 0:
                cell_scores[ct] = score

        # 归一化成“类似比例”的向量，方便 Bray–Curtis
        total_score = sum(cell_scores.values())
        if total_score > 0:
            cell_type_proportions = {
                ct: round(v / total_score, 6) for ct, v in cell_scores.items()
            }
        else:
            cell_type_proportions = {}

        # 优势度（dominance）和均匀度（evenness = 1 - dominance）
        if cell_type_proportions:
            dominance = max(cell_type_proportions.values())
            evenness = round(1.0 - dominance, 6)
        else:
            dominance = 0.0
            evenness = 0.0

        return {
            "cell_type_proportions": cell_type_proportions,
            "dominance": dominance,
            "evenness": evenness,
            "cell_type_count": len(cell_type_proportions),
        }

    # -------------------------------------------------------
    # 3. 功能轴特征（相似度主通道）
    # -------------------------------------------------------
    def get_functional_feature(self, cluster):
        """功能轴分数 + 精简的富集通路"""
        # 富集 term 去重（用于注释）
        enr = cluster.get("core_enrichment", [])
        term_dict = {}
        for item in enr:
            term = item.get("term")
            adj = item.get("adj_pvalue", 1.0)
            if term not in term_dict or adj < term_dict[term]["adj_pvalue"]:
                term_dict[term] = item

        enriched = [
            {
                "term": it["term"],
                "adj_pvalue": round(it.get("adj_pvalue", 1.0), 6),
                "source": it.get("source", "unknown"),
            }
            for it in term_dict.values()
        ]

        # 功能轴计算
        axis_scores = {}
        domains = cluster.get("domains", [])
        if domains and self.axis_genesets:
            tmp = {axis: [] for axis in self.axis_genesets}

            for did in domains:
                expr = (
                    self.domain_features_norm
                    .get(did, {})
                    .get("gene_avg_expr_norm", {})
                )
                if not expr:
                    continue

                for axis_name, genes in self.axis_genesets.items():
                    matched = [expr[g] for g in genes if g in expr]
                    if matched:
                        tmp[axis_name].append(float(np.mean(matched)))

            axis_scores = {
                axis: float(np.mean(vals))
                for axis, vals in tmp.items()
                if vals
            }

        return {
            "enriched_pathways": enriched,
            "pathway_count": len(enriched),
            "axis_scores": axis_scores,
            "term_standardized": True,
        }

    # -------------------------------------------------------
    # 4. 构建单聚类特征
    # -------------------------------------------------------
    def build_3d_feature(self, cluster):
        raw_id = cluster.get("cluster_id", id(cluster))
        cluster_id = f"{self.sample_id}_cluster_{raw_id}"

        return {
            "cluster_id": cluster_id,
            "sample_id": self.sample_id,
            "cancer_type": self.cancer_type,
            "transcript": self.get_transcript_feature(cluster),
            "cell_context": self.get_cell_context_feature(cluster),
            "functional": self.get_functional_feature(cluster),
        }

    # -------------------------------------------------------
    # 5. 构建所有聚类特征
    # -------------------------------------------------------
    def build_all_clusters_3d_features(self):
        return [self.build_3d_feature(c) for c in self.clusters]


# -------------------------------------------------------
# 6. 工具函数：对多个 sample_data 统一构建 cluster 3D 特征
# -------------------------------------------------------
def build_clusters_for_samples(sample_data_list):
    all_clusters = []
    for sample in sample_data_list:
        eng = FeatureEngine(sample)
        feats = eng.build_all_clusters_3d_features()
        for f in feats:
            f["3d_features"] = {
                "transcript": f["transcript"],
                "cell_context": f["cell_context"],
                "functional": f["functional"],
            }
            all_clusters.append(f)
    return all_clusters

