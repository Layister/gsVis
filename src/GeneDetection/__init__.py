# import numpy as np
# import pandas as pd
# from  gss_analysis import run_gss_analysis
#
#
# def analyze_at_spot_domain_level(mk_score_df, adata, gss_threshold=0.7,
#                                  expression_threshold=0.5, top_genes_per_spot=5):
#     """
#     Spot微域特征基因识别（原始spot为中心）
#     """
#     spot_domain_features = {}
#     regional_neighbors = adata.uns.get("regional_neighbors", {})
#
#     print(f"开始处理 {adata.n_obs} 个spots...")
#
#     for spot_idx in range(adata.n_obs):
#         # 步骤1: 确定当前spot的微域范围
#         spot_str = str(spot_idx)
#         if spot_str in regional_neighbors:
#             domain_spots = regional_neighbors[spot_str] # 使用中心spot+邻居作为微域
#         else:
#             domain_spots = [spot_idx] # 没有预定义邻居，微域只包含自身
#
#         # 步骤2: 计算微域级别的GSS和表达特征
#         domain_gss_scores, domain_expression_stats = calculate_domain_features(
#             mk_score_df, adata, domain_spots
#         )
#
#         # 步骤3: 筛选特征基因
#         feature_genes_df = filter_feature_genes(
#             domain_gss_scores, domain_expression_stats,
#             gss_threshold, expression_threshold, top_genes_per_spot
#         )
#
#         # 步骤4: 以原始spot为中心保存结果
#         spot_domain_features[spot_idx] = {
#             # 微域组成信息
#             'center_spot': spot_idx,  # 中心spot索引
#             'all_domain_spots': domain_spots,  # 完整微域spots
#             'domain_size': len(domain_spots),  # 微域大小
#
#             # 特征基因信息
#             'feature_genes': feature_genes_df,  # 特征基因DataFrame
#             'top_gene': feature_genes_df.iloc[0]['gene'] if len(feature_genes_df) > 0 else None,
#             'num_feature_genes': len(feature_genes_df),
#
#             # 统计信息
#             'mean_gss_score': domain_gss_scores.mean() if len(domain_gss_scores) > 0 else 0,
#             'max_gss_score': domain_gss_scores.max() if len(domain_gss_scores) > 0 else 0,
#         }
#
#         # 进度显示
#         if spot_idx % 100 == 0:
#             print(f"已处理 {spot_idx}/{adata.n_obs} spots")
#
#     print(f"完成! 共为 {len(spot_domain_features)} 个spots计算了微域特征基因")
#     return spot_domain_features
#
#
# def calculate_domain_features(mk_score_df, adata, domain_spots):
#     # 检查基因顺序是否一致（mk_score_df行索引 vs adata.var_names）
#     if not mk_score_df.index.equals(adata.var_names):
#         raise ValueError("mk_score_df的基因顺序与adata的基因顺序不一致！")
#
#     # 1. 计算微域GSS分数（按位置选取spot列，行是基因）
#     domain_gss = mk_score_df.iloc[:, domain_spots]  # 正确：按位置选列（domain_spots是整数）
#     domain_gss_scores = domain_gss.mean(axis=1)  # 每个基因在微域内的平均GSS（行方向求平均）
#     # 确保结果是Series，索引为基因名
#     domain_gss_scores = pd.Series(domain_gss_scores, index=mk_score_df.index, name='gss_score')
#
#     # 2. 计算微域表达统计（adata中spot顺序与mk_score_df列顺序一致，直接按位置取）
#     domain_expression = adata[domain_spots, :].X  # adata[spot位置, 所有基因]
#     if hasattr(domain_expression, 'toarray'):
#         domain_expression = domain_expression.toarray()  # 稀疏矩阵转稠密
#
#     # 每个基因的表达统计（axis=0：按列求，列对应基因）
#     mean_expression = np.mean(domain_expression, axis=0)
#     expression_variance = np.var(domain_expression, axis=0)
#
#     expression_stats = {
#         'mean_expression': mean_expression,
#         'expression_variance': expression_variance,
#         'max_expression': np.max(domain_expression, axis=0),
#         'min_expression': np.min(domain_expression, axis=0)
#     }
#
#     return domain_gss_scores, expression_stats
#
#
# def filter_feature_genes(domain_gss_scores, expression_stats,
#                          gss_threshold=0.7, expression_threshold=0.5, top_k=5):
#     # 构建候选基因DataFrame（基因顺序一致，长度相同）
#     candidate_genes = pd.DataFrame({
#         'gene': domain_gss_scores.index,  # 基因名
#         'gss_score': domain_gss_scores.values.astype(float),  # 强制float类型
#         'mean_expression': expression_stats['mean_expression'].astype(float),
#         'expression_variance': expression_stats['expression_variance'].astype(float)
#     })
#
#     # 移除异常值（如inf）
#     candidate_genes = candidate_genes.replace([np.inf, -np.inf], np.nan).dropna()
#
#     # 筛选逻辑
#     high_gss = candidate_genes['gss_score'] > gss_threshold
#     high_expr = candidate_genes['mean_expression'] > expression_threshold
#     filtered_genes = candidate_genes[high_gss & high_expr]
#
#     # 调整保留数量
#     if len(filtered_genes) == 0:
#         filtered_genes = candidate_genes.nlargest(max(3, top_k), 'gss_score')
#     else:
#         if len(filtered_genes) > top_k:
#             filtered_genes = filtered_genes.nlargest(top_k, 'gss_score')
#         elif len(filtered_genes) < 3:
#             filtered_genes = candidate_genes.nlargest(max(3, top_k), 'gss_score')
#
#     # 排序并添加排名
#     filtered_genes = filtered_genes.sort_values('gss_score', ascending=False)
#     filtered_genes['rank'] = range(1, len(filtered_genes) + 1)
#
#     return filtered_genes