"""三维特征工程"""

import numpy as np


class ThreeDFeatureEngine:
    """三维特征工程类"""

    def __init__(self, sample_data):
        """
        sample_data: 来自load_multi_sample_data的单个样本字典
        """
        self.sample_id = sample_data["sample_id"]
        self.cancer_type = sample_data["cancer_type"]
        self.domain_features_norm = sample_data["domain_features_norm"]  # 归一化表达数据
        self.cell_type_data_norm = sample_data["cell_type_data_norm"]  # 标准化细胞类型数据
        self.clusters = sample_data["clusters"]  # 样本的所有聚类

    def _get_cluster_domains_expr(self, cluster_domains):
        """汇总聚类内所有domain的归一化表达数据"""
        # cluster_domains: 聚类的domains列表（如["domain_0", "domain_1"]）
        domain_expr = {}
        for domain_id in cluster_domains:
            # 读取该domain的归一化表达数据
            domain_info = self.domain_features_norm.get(domain_id, {})
            expr_norm = domain_info.get("gene_avg_expr_norm", {})
            # 汇总基因表达（后续计算平均值）
            for gene, val in expr_norm.items():
                if gene not in domain_expr:
                    domain_expr[gene] = []
                domain_expr[gene].append(val)
        # 计算聚类内每个基因的平均表达
        avg_expr = {gene: np.mean(vals) for gene, vals in domain_expr.items() if vals}
        return avg_expr

    def _get_non_cluster_domains_expr(self, cluster_domains):
        """汇总聚类外所有domain的归一化表达数据"""
        all_domains = list(self.domain_features_norm.keys())
        non_cluster_domains = [d for d in all_domains if d not in cluster_domains]
        return self._get_cluster_domains_expr(non_cluster_domains)

    def get_transcript_feature(self, cluster):
        """1. 区域特异性转录特征（保留SpecScore>0的基因）"""
        cluster_domains = cluster.get("domains", [])
        if not cluster_domains:
            return {"specific_genes": [], "spec_score_mean": 0.0, "gene_count": 0}

        # 计算聚类内/外平均表达
        avg_cluster = self._get_cluster_domains_expr(cluster_domains)
        avg_non_cluster = self._get_non_cluster_domains_expr(cluster_domains)

        # 计算SpecScore（聚类内-外差值）
        spec_score = {}
        for gene in avg_cluster.keys():
            spec_score[gene] = avg_cluster[gene] - avg_non_cluster.get(gene, 0.0)

        # 保留所有SpecScore>0的基因（无top筛选）
        specific_genes = [(g, s) for g, s in spec_score.items() if s > 0]
        spec_score_mean = np.mean([s for _, s in specific_genes]) if specific_genes else 0.0

        return {
            "specific_genes": specific_genes,
            "spec_score_mean": round(spec_score_mean, 4),
            "gene_count": len(specific_genes)
        }

    def get_cell_context_feature(self, cluster):
        """2. 细胞微环境组成谱（保留所有细胞类型，无比例过滤）"""
        cluster_domains = cluster.get("domains", [])
        if not cluster_domains:
            return {"cell_type_proportions": {}, "dominance": 0.0, "evenness": 0.0, "cell_type_count": 0}

        # 汇总聚类内所有domain的细胞类型比例
        cell_type_sum = {}
        domain_count = 0
        for domain_id in cluster_domains:
            # 读取该domain（spot）的标准化细胞类型比例
            cell_type_prop = self.cell_type_data_norm.get(domain_id, {})
            if not cell_type_prop:
                continue
            domain_count += 1
            # 累加细胞类型比例
            for cell_type, prop in cell_type_prop.items():
                if cell_type not in cell_type_sum:
                    cell_type_sum[cell_type] = 0.0
                cell_type_sum[cell_type] += prop

        # 计算聚类内细胞类型平均比例
        if domain_count == 0:
            cell_type_avg = {}
        else:
            cell_type_avg = {ct: round(sum_prop / domain_count, 4) for ct, sum_prop in cell_type_sum.items()}

        # 计算优势度/均匀度
        if cell_type_avg:
            dominance = max(cell_type_avg.values())
            evenness = round(1 - dominance, 4)
            cell_type_count = len(cell_type_avg)
        else:
            dominance = 0.0
            evenness = 0.0
            cell_type_count = 0

        return {
            "cell_type_proportions": cell_type_avg,
            "dominance": dominance,
            "evenness": evenness,
            "cell_type_count": cell_type_count
        }

    def get_functional_feature(self, cluster):
        """3. 功能通路集合（全量保留功能通路，仅去重复）"""
        # 读取聚类的富集通路（已标准化术语，adj_p<0.01）
        enrichment_list = cluster.get("core_enrichment", [])
        if not enrichment_list:
            return {"enriched_pathways": [], "pathway_count": 0, "term_standardized": True}

        # 去重复（保留adj_p最小的条目）
        term_dict = {}
        for item in enrichment_list:
            term = item.get("term")
            adj_p = item.get("adj_pvalue", 1.0)
            if term not in term_dict or adj_p < term_dict[term]["adj_pvalue"]:
                term_dict[term] = item

        # 转换为列表并保留关键字段
        enriched_pathways = [
            {
                "term": item["term"],
                "adj_pvalue": round(item.get("adj_pvalue", 1.0), 6),
                "source": item.get("source", "unknown")
            }
            for item in term_dict.values()
        ]

        return {
            "enriched_pathways": enriched_pathways,
            "pathway_count": len(enriched_pathways),
            "term_standardized": True
        }

    def build_3d_feature(self, cluster):
        """构建单个聚类的完整三维特征"""
        raw_cluster_id = cluster.get("cluster_id", id(cluster))
        cluster_id = f"{self.sample_id}_cluster_{str(raw_cluster_id)}" # 确保是字符串

        #cluster_id = cluster.get("cluster_id", f"{self.sample_id}_cluster_{id(cluster)}")

        return {
            "cluster_id": cluster_id,
            "sample_id": self.sample_id,
            "cancer_type": self.cancer_type,
            "transcript_feature": self.get_transcript_feature(cluster),
            "cell_context_feature": self.get_cell_context_feature(cluster),
            "functional_feature": self.get_functional_feature(cluster)
        }

    def build_all_clusters_3d_features(self):
        """构建样本内所有聚类的三维特征"""
        all_features = []
        for cluster in self.clusters:
            all_features.append(self.build_3d_feature(cluster))
        return all_features