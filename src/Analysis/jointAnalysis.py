import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

work_dir = '../output/HEST2/'
specie = 'Homo sapiens'

res_dict = {}
gene_keys = defaultdict(list)
cancer_gene_counts = defaultdict(Counter)  # 按癌症类型统计基因频率
root_path = work_dir + f'{specie}'

for cancer in os.scandir(root_path):
    if cancer.is_dir() and not cancer.name.startswith('.'):  # 跳过隐藏文件
        for sample_dir in os.scandir(cancer.path):
            if sample_dir.is_dir() and not sample_dir.name.startswith('.'):  # 跳过隐藏文件
                current_key = f'{cancer.name}_{sample_dir.name}'

                # 构建文件路径
                calc_file = os.path.join(sample_dir.path, 'calculateTopGSS_selected_genes.csv')
                select_file = os.path.join(sample_dir.path, 'selectGenes_selected_genes.csv')

                # 检查文件是否存在
                if not os.path.exists(calc_file) or not os.path.exists(select_file):
                    print(f"跳过 {current_key}，文件不存在")
                    continue

                try:
                    df_calculate = pd.read_csv(calc_file, sep='\t')
                    df_select = pd.read_csv(select_file, sep='\t')

                    # 找出共同基因
                    common_genes = set(df_calculate['gene'][df_calculate['selected']==True]).intersection(set(df_select['gene']))
                    res_dict[current_key] = common_genes

                    # 记录每个基因出现的键
                    for gene in common_genes:
                        gene_keys[gene].append(current_key)

                    # 按癌症类型统计基因频率
                    for gene in common_genes:
                        cancer_gene_counts[cancer.name][gene] += 1

                except Exception as e:
                    print(f"处理 {current_key} 时出错: {e}")

print(res_dict)

# ----------
# 第一部分：基因频率统计（保持不变）
# ----------

# 统计基因出现频率
all_genes = []
for genes in res_dict.values():
    all_genes.extend(genes)  # 将每个样本的共同基因添加到总列表
gene_counts = Counter(all_genes)  # 统计每个基因出现的次数


# 筛选频率>1的基因，并整理信息
gene_info = []
for gene, keys in gene_keys.items():
    freq = len(keys)
    if freq > 1:  # 只保留频率大于1的基因

        # 提取该基因出现的所有癌症类型
        cancer_types = set()
        for key in keys:
            cancer_type = key.split('_')[0]  # 从"癌症类型_样本名称"中提取癌症类型
            cancer_types.add(cancer_type)

        gene_info.append({
            'gene': gene,
            'frequency': freq,
            'cancers': len(cancer_types),  # 出现的癌症类型数量
            'cancer_types': ', '.join(sorted(cancer_types)),  # 具体的癌症类型列表
            'keys': ', '.join(keys)  # 将键列表用逗号连接成字符串
        })

# 转换为DataFrame并按频率降序排序
df = (pd.DataFrame(gene_info)
      .sort_values(by='frequency', ascending=False)
      .to_csv(root_path + '/gene_frequency.csv', index=False, sep='\t'))

print(f"基因频率信息已保存到文件!")

# ----------
# 第二部分：癌症特异基因统计
# ----------

cancer_specific_genes = []

# 遍历每个癌症类型
for cancer_name, gene_counter in cancer_gene_counts.items():
    # 找出在该癌症类型中出现频率大于1的基因
    for gene, count in gene_counter.items():
        if count > 1:  # 只保留在同一个癌症类型中出现多次的基因
            cancer_specific_genes.append({
                'cancer_type': cancer_name,
                'gene': gene,
                'frequency_in_cancer': count,
                'samples': ', '.join([key for key in gene_keys[gene] if key.split('_')[0] == cancer_name])
            })

# 转换为DataFrame并按癌症类型和频率排序
if cancer_specific_genes:
    df_cancer = (pd.DataFrame(cancer_specific_genes)
                 .sort_values(by=['cancer_type', 'frequency_in_cancer'], ascending=[True, False])
                 .to_csv(root_path + '/cancer_specific_genes.csv', index=False, sep='\t'))
    print(f"癌症特异基因信息已保存到文件!")
else:
    print("未找到任何癌症特异基因")

# ----------
# 第三部分：只在某一个癌症中出现的基因（新增部分）
# ----------

cancer_unique_genes = []

# 遍历所有基因
for gene, keys in gene_keys.items():
    # 提取该基因出现的所有癌症类型
    cancer_types = set()
    for key in keys:
        cancer_type = key.split('_')[0]
        cancer_types.add(cancer_type)

    # 检查是否只在某一个癌症类型中出现
    if len(cancer_types) == 1:
        cancer_name = list(cancer_types)[0]
        frequency = len(keys)

        cancer_unique_genes.append({
            'gene': gene,
            'cancer_type': cancer_name,
            'frequency': frequency,
            'samples': ', '.join(keys)
        })

# 按频率降序排序并保存
if cancer_unique_genes:
    df_unique = (pd.DataFrame(cancer_unique_genes)
                 .sort_values(by=['frequency'], ascending=False)
                 .to_csv(root_path + '/cancer_unique_genes.csv', index=False, sep='\t'))
    print(f"单一癌症特有基因信息已保存到文件!")
    print(f"共找到 {len(cancer_unique_genes)} 个只在单一癌症中出现的基因")
else:
    print("未找到任何只在单一癌症中出现的基因")

# ----------
# 可视化部分
# ----------

# 绘制频率分布图
plt.figure(figsize=(12, 6))

# 选择出现频率最高的前30个基因进行展示（避免图过于拥挤）
top_genes = gene_counts.most_common(30)
genes, counts = zip(*top_genes) if top_genes else ([], [])

plt.bar(genes, counts, color='skyblue')
plt.title('Gene Occurrence Frequency in Common Gene Sets', fontsize=14)
plt.xlabel('Gene', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, ha='right')  # 旋转基因名，避免重叠
plt.tight_layout()  # 自动调整布局
plt.savefig(root_path + '/gene_frequency_plot.png')  # 保存图片
plt.show()

# 查看所有基因的频率分布直方图（按频率区间统计）
plt.figure(figsize=(10, 6))
plt.hist(gene_counts.values(), bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Gene Occurrence Frequencies', fontsize=14)
plt.xlabel('Occurrence Frequency', fontsize=12)
plt.ylabel('Number of Genes', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(root_path + '/gene_frequency_distribution.png')  # 保存图片
plt.show()

# 为每个癌症类型绘制特异基因频率图
# for cancer_name, gene_counter in cancer_gene_counts.items():
#     if len(gene_counter) > 0:  # 只处理有数据的癌症类型
#         # 选择该癌症类型中出现频率最高的前10个基因
#         top_genes = gene_counter.most_common(10)
#         genes, counts = zip(*top_genes) if top_genes else ([], [])
#
#         plt.figure(figsize=(10, 6))
#         plt.bar(genes, counts, color='lightcoral')
#         plt.title(f'Top 10 Specific Genes in {cancer_name}', fontsize=14)
#         plt.xlabel('Gene', fontsize=12)
#         plt.ylabel('Frequency in Cancer', fontsize=12)
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.savefig(f"{root_path}/{cancer_name}_specific_genes.png")  # 保存图片
#         plt.show()