"""工具函数和基因集管理"""

import os
import json
import numpy as np
import pandas as pd
import ast
import traceback
import gseapy as gp
from collections import defaultdict

from analysis_config import AnalysisConfig


def convert_numpy_types(obj):
    """将numpy数据类型转换为Python原生类型以便JSON序列化 - 增强版"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int8, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16, np.float_)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # 递归处理字典，确保所有键都是字符串
        converted_dict = {}
        for key, value in obj.items():
            # 将键转换为字符串
            str_key = str(key) if not isinstance(key, (str, int, float, bool)) or key is None else key
            converted_dict[str_key] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # 对于其他未知类型，尝试转换为字符串
        try:
            return str(obj)
        except:
            return None


def calculate_gene_similarity(genes1, genes2):
    """计算两个基因集合的相似度（Jaccard相似度）"""
    set1 = set(genes1)
    set2 = set(genes2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def batch_correct_expression(sample_data):
    """简单批次效应校正（标准化）"""
    print("正在进行批次效应校正...")

    all_expressions = []
    sample_indices = []
    gene_lists = []

    # 收集所有样本的表达数据
    for sidx, sample in enumerate(sample_data):
        for cluster_idx, cluster in enumerate(sample["clusters"]):
            try:
                # 对于每个聚类，聚合其所有spots的表达数据
                cluster_expr = {}
                spot_count = 0

                # 遍历聚类中的所有spots
                for spot_id in cluster.get("domains", []):
                    if spot_id in sample["domain_features"]:
                        spot_data = sample["domain_features"][spot_id]
                        gene_expr = spot_data.get("gene_avg_expr_domain", {})

                        # 去噪处理
                        from data_loader import DataDenoiser
                        cleaned_expr = DataDenoiser.denoise_expression_data(gene_expr)

                        # 聚合表达数据（取平均值）
                        for gene, expr in cleaned_expr.items():
                            if gene in cluster_expr:
                                cluster_expr[gene] += expr
                            else:
                                cluster_expr[gene] = expr

                        spot_count += 1

                # 计算平均表达
                if spot_count > 0:
                    for gene in cluster_expr:
                        cluster_expr[gene] /= spot_count

                    all_expressions.append(cluster_expr)
                    sample_indices.append(sidx)
                    gene_lists.append(list(cluster_expr.keys()))
                else:
                    print(f"样本 {sidx} 聚类 {cluster_idx} 没有spots的表达数据")

            except Exception as e:
                print(f"处理样本 {sidx} 聚类 {cluster_idx} 时出错: {str(e)}")
                continue

    if not all_expressions:
        print("警告: 没有可用的表达数据，跳过批次校正")
        return sample_data

    print(f"收集到 {len(all_expressions)} 个聚类的表达数据")

    # 转换为矩阵并标准化
    try:
        # 获取所有基因的并集
        all_genes = set()
        for genes in gene_lists:
            all_genes.update(genes)
        all_genes = sorted(all_genes)

        # 构建表达矩阵
        expr_matrix = np.zeros((len(all_expressions), len(all_genes)))
        for i, expr_dict in enumerate(all_expressions):
            for j, gene in enumerate(all_genes):
                expr_matrix[i, j] = expr_dict.get(gene, 0)

        print(f"表达数据矩阵形状: {expr_matrix.shape}")

        # 检查是否有足够的数据进行标准化
        if len(expr_matrix) > 1 and len(all_genes) > 0:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_expr = scaler.fit_transform(expr_matrix)

            # 回填校正后的数据
            expr_idx = 0
            for sidx, sample in enumerate(sample_data):
                for cluster in sample["clusters"]:
                    if expr_idx < len(scaled_expr):
                        cluster["scaled_expression"] = scaled_expr[expr_idx]
                        cluster["expression_genes"] = all_genes
                        expr_idx += 1
            print("✅ 批次校正完成")
        else:
            print("⚠️ 数据不足，跳过批次校正")

    except Exception as e:
        print(f"❌ 批次校正过程中出错: {str(e)}")
        traceback.print_exc()

    return sample_data


class GeneSetManager:
    """基因集管理器 - 直接读取GMT文件版本"""
    _gene_sets = None

    @classmethod
    def get_gene_sets(cls):
        if cls._gene_sets is None:
            cls._gene_sets = cls._load_gene_sets()
        return cls._gene_sets

    @classmethod
    def _load_gene_sets(cls):
        """直接读取GMT文件，不依赖gseapy的get_library"""
        gene_sets = {}

        for name, file_path in AnalysisConfig.local_gmt_files.items():
            if os.path.exists(file_path):
                try:
                    gmt_data = cls._read_gmt_file(file_path)
                    gene_sets[name] = gmt_data
                    print(f"✅ 成功加载基因集: {name}, 包含 {len(gmt_data)} 个条目")
                except Exception as e:
                    print(f"❌ 加载基因集 {name} 失败: {str(e)}")
                    traceback.print_exc()
            else:
                print(f"⚠️ 警告: GMT文件不存在: {file_path}")

        return gene_sets

    @staticmethod
    def _read_gmt_file(gmt_file_path):
        """直接读取GMT文件格式"""
        gene_sets = {}
        with open(gmt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                # GMT格式: 基因集名称 TAB 描述 TAB 基因1 TAB 基因2 TAB ...
                gene_set_name = parts[0]
                genes = [gene.strip() for gene in parts[2:] if gene.strip()]

                if gene_set_name and genes:
                    gene_sets[gene_set_name] = genes

        return gene_sets


def perform_local_enrichment(gene_list):
    """使用本地GMT文件进行富集分析"""
    if len(gene_list) < 5:
        return {"error": "基因数量不足进行富集分析", "total_genes": len(gene_list)}

    gene_sets = GeneSetManager.get_gene_sets()
    if not gene_sets:
        return {"error": "没有可用的基因集", "total_genes": len(gene_list)}

    try:
        # 只使用配置中选定的基因集
        filtered_gene_sets = {
            k: v for k, v in gene_sets.items()
            if k in AnalysisConfig.enrichment_gene_sets
        }

        if not filtered_gene_sets:
            return {"error": "选定的基因集不可用", "total_genes": len(gene_list)}

        print(f"使用基因集进行富集分析: {list(filtered_gene_sets.keys())}")
        print(f"输入基因数量: {len(gene_list)}")

        # 检查基因是否在基因集中有匹配
        all_genes_in_sets = set()
        for gene_set in filtered_gene_sets.values():
            for genes in gene_set.values():
                all_genes_in_sets.update(genes)

        matched_genes = set(gene_list) & all_genes_in_sets
        print(f"在基因集中匹配到的基因数量: {len(matched_genes)}")

        if len(matched_genes) < 3:
            return {
                "error": f"基因匹配不足 (仅{len(matched_genes)}个基因在基因集中)",
                "total_genes": len(gene_list),
                "matched_genes": len(matched_genes)
            }

        significant_terms = []

        for gset_name in filtered_gene_sets.keys():
            # 获取对应的GMT文件路径
            gmt_file_path = AnalysisConfig.local_gmt_files.get(gset_name)
            if not gmt_file_path or not os.path.exists(gmt_file_path):
                print(f"警告: GMT文件不存在: {gmt_file_path}")
                continue

            try:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=[gmt_file_path],  # 传入GMT文件路径列表
                    organism="human",
                    outdir=None,
                    cutoff=AnalysisConfig.enrichment_cutoff,
                    no_plot=True
                )

                # 处理结果
                if hasattr(enr, 'results') and enr.results is not None:
                    if isinstance(enr.results, dict):
                        for library_name, df in enr.results.items():
                            if df is not None and not df.empty:
                                significant_df = df[df['Adjusted P-value'] < AnalysisConfig.enrichment_cutoff]
                                for _, row in significant_df.iterrows():
                                    significant_terms.append({
                                        "term": row['Term'],
                                        "p_value": float(row['P-value']),
                                        "adjusted_p": float(row['Adjusted P-value']),
                                        "genes": row['Genes'].split(';') if isinstance(row['Genes'], str) else row[
                                            'Genes'],
                                        "library": gset_name
                                    })
                    elif hasattr(enr, 'res2d') and enr.res2d is not None and not enr.res2d.empty:
                        df = enr.res2d
                        significant_df = df[df['Adjusted P-value'] < AnalysisConfig.enrichment_cutoff]
                        for _, row in significant_df.iterrows():
                            significant_terms.append({
                                "term": row['Term'],
                                "p_value": float(row['P-value']),
                                "adjusted_p": float(row['Adjusted P-value']),
                                "genes": row['Genes'].split(';') if isinstance(row['Genes'], str) else row['Genes'],
                                "library": gset_name
                            })

                print(f"基因集 {gset_name} 分析完成")

            except Exception as e:
                print(f"基因集 {gset_name} 富集分析失败: {str(e)}")
                continue

        if not significant_terms:
            return {
                "warning": f"没有找到显著富集项 (p < {AnalysisConfig.enrichment_cutoff})",
                "total_genes": len(gene_list),
                "matched_genes": len(matched_genes),
                "gene_sets_used": list(filtered_gene_sets.keys())
            }

        return {
            "significant_terms": significant_terms,
            "total_genes": len(gene_list),
            "enriched_pathways": len(significant_terms),
            "matched_genes": len(matched_genes),
            "gene_sets_used": list(filtered_gene_sets.keys())
        }

    except Exception as e:
        print(f"富集分析失败: {str(e)}")
        traceback.print_exc()
        return {"error": f"富集分析失败: {str(e)}", "total_genes": len(gene_list)}


def generate_enhanced_report(all_csms, mp_structures, sample_data):
    """生成增强版分析报告 - 修复序列化问题"""

    # 计算统计信息
    total_csms = sum(len(info['structures']) for info in all_csms.values())
    total_samples = len(sample_data)

    # 计算显著性结构数量
    significant_csms = 0
    for cancer_info in all_csms.values():
        for structure in cancer_info['structures'].values():
            if structure.get('sample_coverage', 0) >= 2:
                significant_csms += 1

    # 确保MPs结构中的键是字符串
    mp_structures_str_keys = {}
    for mp_id, mp_info in mp_structures.items():
        mp_structures_str_keys[str(mp_id)] = convert_numpy_types(mp_info)

    # 确保CSMs结构中的键是字符串
    all_csms_str_keys = {}
    for cancer_type, cancer_info in all_csms.items():
        cancer_info_copy = cancer_info.copy()
        structures_str_keys = {}
        for csm_id, csm_info in cancer_info['structures'].items():
            structures_str_keys[str(csm_id)] = convert_numpy_types(csm_info)
        cancer_info_copy['structures'] = structures_str_keys
        all_csms_str_keys[cancer_type] = cancer_info_copy

    report = {
        "analysis_framework": "分层多模态泛癌分析",
        "framework_details": {
            "approach": "两步走分层架构 (CSMs → MPs)",
            "features": "三维特征指纹 (转录排位 + 细胞环境 + 核心基因)",
            "similarity": "多模态相似度 (超几何检验 + RRHO + 余弦相似度 + Simpson)",
            "consensus": "分层共识网络",
            "denoising": "移除线粒体/核糖体/热休克蛋白基因"
        },
        "summary": {
            "total_cancer_types": len(all_csms),
            "total_csms": total_csms,
            "significant_csms": significant_csms,
            "total_mps": len(mp_structures),
            "sample_count": total_samples,
            "cancer_type_breakdown": {
                cancer_type: {
                    "cancer_name": info['cancer_name'],
                    "csms_count": len(info['structures']),
                    "sample_count": info['sample_count'],
                    "significant_structures": sum(1 for s in info['structures'].values()
                                                  if s.get('sample_coverage', 0) >= 2)
                }
                for cancer_type, info in all_csms.items()
            }
        },
        "pan_cancer_meta_programs": mp_structures_str_keys,
        "cancer_specific_modules": all_csms_str_keys,
        "analysis_parameters": {
            "denoising_genes": AnalysisConfig.noise_genes,
            "feature_engineering": AnalysisConfig.feature_params,
            "similarity_weights": AnalysisConfig.similarity_weights,
            "consensus_thresholds": AnalysisConfig.consensus_params,
            "enrichment_cutoff": AnalysisConfig.enrichment_cutoff
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }

    # 保存报告
    os.makedirs(AnalysisConfig.output_dir, exist_ok=True)
    report_path = os.path.join(AnalysisConfig.output_dir, "enhanced_analysis_report.json")

    try:
        # 确保完全转换
        report_serializable = convert_numpy_types(report)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)
        print(f"✅ 增强分析报告已保存: {report_path}")
    except Exception as e:
        print(f"❌ 保存报告失败: {str(e)}")
        # 尝试简化保存
        try:
            simplified_report = {
                "summary": convert_numpy_types(report["summary"]),
                "total_structures": {
                    "CSMs": total_csms,
                    "MPs": len(mp_structures)
                }
            }
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_report, f, indent=2)
            print("✅ 简化版报告已保存")
        except Exception as e2:
            print(f"❌ 无法保存任何版本的报告: {e2}")

    return report