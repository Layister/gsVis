import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict


# 原有的数据处理部分
work_dir = '/Users/wuyang/Documents/MyPaper/3/gsVis/output/'
dataset = 'HEST'
specie = 'Homo sapiens'

res_dict = {}
gene_keys = defaultdict(list)
root_path = work_dir + f'{dataset}/{specie}'

for cancer in os.scandir(root_path):
    if cancer.is_dir():
        for id in os.scandir(cancer):
            # 构建当前样本的键
            current_key = f'{cancer.name}_{id.name}'

            # 加载两个基因文件
            df_calculate = pd.read_csv(id.path + '/calculateTopGSS_selected_genes.csv', sep='\t')
            df_select = pd.read_csv(id.path + '/selectGenes_selected_genes.csv', sep='\t')

            # 找出共同基因
            common_genes = set(df_calculate['gene']).intersection(set(df_select['gene']))
            res_dict[current_key] = common_genes

            # 记录每个基因出现的键
            for gene in common_genes:
                gene_keys[gene].append(current_key)

print(res_dict)

# ----------

# 统计基因出现频率
all_genes = []
for genes in res_dict.values():
    all_genes.extend(genes)  # 将每个样本的共同基因添加到总列表
gene_counts = Counter(all_genes)  # 统计每个基因出现的次数

# ----------

# 筛选频率>1的基因，并整理信息
gene_info = []
for gene, keys in gene_keys.items():
    freq = len(keys)
    if freq > 1:  # 只保留频率大于1的基因
        gene_info.append({
            'gene': gene,
            'frequency': freq,
            'keys': ', '.join(keys)  # 将键列表用逗号连接成字符串
        })

# 转换为DataFrame并按频率降序排序
df = (pd.DataFrame(gene_info)
      .sort_values(by='frequency', ascending=False)
      .to_csv(root_path+'/gene_frequency.csv', index=False, sep='\t'))

print(f"基因信息已保存到文件!")

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
plt.show()

# 查看所有基因的频率分布直方图（按频率区间统计）
plt.figure(figsize=(10, 6))
plt.hist(gene_counts.values(), bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Gene Occurrence Frequencies', fontsize=14)
plt.xlabel('Occurrence Frequency', fontsize=12)
plt.ylabel('Number of Genes', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()