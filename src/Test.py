import numpy as np
from esda.moran import Moran
from libpysal.weights import KNN
import matplotlib.pyplot as plt

# 生成空间坐标（n个点）
n = 1000
np.random.seed(42)
coords = np.random.random((n, 2)) * 10

# 生成3类基因表达（调整参数增强空间特异性）
n_genes = 300
genes = np.zeros((n_genes, n))

# A类基因（高聚集，增强特异性）
for i in range(100):
    center = np.random.random(2) * 10
    dist = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    # 增强空间衰减，增加噪声但保持高特异性
    genes[i] = np.exp(-dist * 2) * 15 + np.random.normal(0, 1, n)

# B类基因（随机分布，降低方差）
for i in range(100, 200):
    genes[i] = np.random.normal(5, 1.5, n)  # 降低方差

# C类基因（均匀但有微小波动）
for i in range(200, 300):
    genes[i] = np.ones(n) * 5 + np.random.normal(0, 0.3, n)

# 可视化一个A类基因的空间表达模式
plt.figure(figsize=(6,6))
plt.scatter(coords[:,0], coords[:,1], c=genes[50], cmap='viridis', s=50)
plt.colorbar(label='Expression Level')
plt.title('Spatial Expression Pattern of Gene A')
plt.show()

# 构建空间权重矩阵（使用KNN，确保每个点有8个邻居）
w = KNN.from_array(coords, k=8)
w.transform = 'r'  # 行标准化

# 计算筛选前的Moran's I
morans_i_before = []
valid_genes_before = []

for i in range(n_genes):
    gene = genes[i]
    # 检查方差和Moran's I有效性
    if np.var(gene) > 1e-10:
        mi = Moran(gene, w)
        if not np.isnan(mi.I):
            morans_i_before.append(mi.I)
            valid_genes_before.append((i, mi.I))  # 保存基因索引和对应的Moran's I

# 基于空间自相关性筛选（直接选择Moran's I高的基因）
valid_genes_before.sort(key=lambda x: x[1], reverse=True)  # 按Moran's I降序排列
top_genes_indices = [x[0] for x in valid_genes_before[:100]]  # 选择前100个
filtered_genes = genes[top_genes_indices]

# 计算筛选后的Moran's I
morans_i_after = []
for gene in filtered_genes:
    if np.var(gene) > 1e-10:
        mi = Moran(gene, w)
        if not np.isnan(mi.I):
            morans_i_after.append(mi.I)

# 结果对比
if morans_i_before and morans_i_after:
    avg_before = np.mean(morans_i_before)
    avg_after = np.mean(morans_i_after)
    enhancement = ((avg_after - avg_before) / avg_before) * 100 if avg_before > 0 else 0

    print(f"筛选前基因数: {len(morans_i_before)}")
    print(f"筛选后基因数: {len(morans_i_after)}")
    print(f"筛选前平均Moran's I: {avg_before:.4f}")
    print(f"筛选后平均Moran's I: {avg_after:.4f}")
    print(f"增强度: {enhancement:.2f}%")

    # 可视化分布变化
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(morans_i_before, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=avg_before, color='black', linestyle='--', label=f'avg: {avg_before:.4f}')
    plt.xlabel('Moran\'s I')
    plt.ylabel('Genes Number')
    plt.title('Before Moran\'s I Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(morans_i_after, bins=20, alpha=0.7, color='orange')
    plt.axvline(x=avg_after, color='black', linestyle='--', label=f'avg: {avg_after:.4f}')
    plt.xlabel('Moran\'s I')
    plt.ylabel('Genes Number')
    plt.title('After Moran\'s I Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("警告：筛选后无有效基因或计算结果全部为NaN")