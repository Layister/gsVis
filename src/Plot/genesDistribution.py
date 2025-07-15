import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 选择前5个基因进行可视化（可根据需要调整）
path = "/Users/wuyang/Documents/MyPaper/3/gsVis/data/BRCA/Human_Breast_Cancer/latent_to_gene/Human_Breast_Cancer_gene_marker_score.feather"
mk_score_df = pd.read_feather(path)

# 选择n个代表性基因
selected_genes = mk_score_df['HUMAN_GENE_SYM'].head(3).tolist()
subset = mk_score_df[mk_score_df['HUMAN_GENE_SYM'].isin(selected_genes)]
melted_df = subset.melt(id_vars='HUMAN_GENE_SYM', var_name='Sample', value_name='Expression')

# 确保基因名列是字符串类型
melted_df['HUMAN_GENE_SYM'] = melted_df['HUMAN_GENE_SYM'].astype(str)

# 计算表达值为0的细胞比例
zero_ratio_dict = {}
non_zero_df = pd.DataFrame(columns=melted_df.columns)
for gene in selected_genes:
    gene_data = melted_df[melted_df['HUMAN_GENE_SYM'] == gene]
    zero_count = (gene_data['Expression'] == 0).sum()
    total_count = len(gene_data)
    zero_ratio = zero_count / total_count
    zero_ratio_dict[gene] = zero_ratio

    # 筛选出非0表达值的数据
    non_zero_gene_data = gene_data[gene_data['Expression'] != 0]
    non_zero_df = pd.concat([non_zero_df, non_zero_gene_data])

# 创建分布图
plt.figure(figsize=(12, 8))
ax = sns.histplot(
    data=non_zero_df,
    x='Expression',
    hue='HUMAN_GENE_SYM',  # 按基因着色
    hue_order=selected_genes,  # 显式设置基因顺序
    element='step',        # 阶梯状直方图
    stat='density',        # 显示密度而非计数
    common_norm=False,     # 每个基因独立归一化
    kde=True,              # 添加核密度估计
    palette='tab10',       # 使用预定义颜色方案
    alpha=0.4,             # 调整透明度
    edgecolor='none',      # 移除边缘线
    line_kws={'linewidth': 2},  # KDE线宽
    legend=False  # 禁用自动生成的图例
)

# 获取Seaborn实际使用的颜色映射
handles = []
colors = {}
for i, gene in enumerate(selected_genes):
    # 获取直方图的颜色
    patch = plt.Rectangle((0, 0), 1, 1, fc=sns.color_palette('tab10')[i])
    handles.append(patch)
    colors[gene] = sns.color_palette('tab10')[i]

# 手动创建图例
plt.legend(
    handles=handles,
    labels=selected_genes,
    title='Gene Symbol',
    title_fontsize=12,
    fontsize=11,
    loc='best',
    frameon=True,
    framealpha=0.8
)

# 添加标题和标签
plt.title('Expression Distribution Comparison of Selected Genes (Non-zero Values)', fontsize=16, pad=20)
plt.xlabel('Expression Level', fontsize=14)
plt.ylabel('Density', fontsize=14)

# 添加网格
plt.grid(axis='y', alpha=0.2)

# 添加均值线，使用与图例相同的颜色映射
for gene in selected_genes:
    gene_data = non_zero_df[non_zero_df['HUMAN_GENE_SYM'] == gene]['Expression']
    mean_val = gene_data.mean()
    plt.axvline(mean_val, color=colors[gene],  # 使用与图例相同的颜色
                linestyle='--', linewidth=1.5, alpha=0.8)

# 添加统计信息框，包含表达值为0的细胞比例
stats_text = "\n".join([f"{gene}: μ={non_zero_df[non_zero_df['HUMAN_GENE_SYM']==gene]['Expression'].mean():.2f}, "
                       f"σ={non_zero_df[non_zero_df['HUMAN_GENE_SYM']==gene]['Expression'].std():.2f}, "
                       f"Zero Ratio={zero_ratio_dict[gene]:.2%}"
                       for gene in selected_genes])

plt.annotate(stats_text,
             xy=(0.98, 0.95),
             xycoords='axes fraction',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             ha='right', va='top')

# 调整布局
plt.tight_layout()
plt.show()