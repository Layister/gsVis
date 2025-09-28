import os
import pandas as pd
import matplotlib.pyplot as plt
import upsetplot as usp
from collections import defaultdict
import argparse
import warnings


# 忽略绘图警告
warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")

# 设置中文字体支持
plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def parse_gene_frequency(file_path):
    """解析gene_frequency.csv文件，提取每个癌症类型包含的基因集合"""
    df = pd.read_csv(file_path, sep='\t')
    cancer_genes = defaultdict(set)

    for _, row in df.iterrows():
        gene = row['gene']
        keys_str = row['keys']
        key_entries = [k.strip() for k in keys_str.split(',')]

        for entry in key_entries:
            if '_' in entry:
                cancer_type = entry.split('_', 1)[0]
                cancer_genes[cancer_type].add(gene)

    return cancer_genes


def get_common_genes(cancer_genes, cancer_types):
    """计算指定癌症类型列表的共同基因"""
    if not cancer_types:
        return set()

    # 从第一个癌症类型开始
    common_genes = cancer_genes[cancer_types[0]].copy()

    # 与其他癌症类型求交集
    for cancer in cancer_types[1:]:
        common_genes.intersection_update(cancer_genes[cancer])

    return common_genes


def plot_upset(cancer_genes, output_dir, min_size=1, max_sets=10, min_cancers=1):
    """绘制Upset图并输出多种癌症的共有基因（至少 min_cancers 种）"""
    os.makedirs(output_dir, exist_ok=True)

    # 限制显示的癌症类型数量
    if len(cancer_genes) > max_sets:
        sorted_cancers = sorted(cancer_genes.items(),
                                key=lambda x: len(x[1]),
                                reverse=True)
        cancer_genes = dict(sorted_cancers[:max_sets])
        print(f"癌症类型过多，仅显示基因数量最多的{max_sets}种")

    cancer_types = list(cancer_genes.keys())
    num_cancers = len(cancer_types)

    if num_cancers < min_cancers:
        print(f"需要至少{min_cancers}种癌症类型才能分析共有基因")
        return

    # 收集所有基因及其在各癌症中的出现情况
    gene_presence = []
    all_genes = set()
    for genes in cancer_genes.values():
        all_genes.update(genes)

    for gene in all_genes:
        presence = {}
        for cancer in cancer_types:
            presence[cancer] = gene in cancer_genes[cancer]
        gene_presence.append(presence)

    # 转换为DataFrame
    df = pd.DataFrame(gene_presence)

    # 计算每个组合的基因数量
    upset_data = df.groupby(cancer_types).size().reset_index(name='count')
    upset_data = upset_data[upset_data['count'] >= min_size]

    if upset_data.empty:
        print(f"没有找到大小≥{min_size}的基因交集")
        return

    # 输出多种癌症组合的共有基因（至少min_cancers种）
    print(f"\n===== 至少{min_cancers}种癌症的共有基因 =====")
    gene_output_path = os.path.join(output_dir, "multi_cancer_common_genes.txt")
    valid_combinations = 0

    with open(gene_output_path, 'w') as f:
        # 遍历所有组合
        for _, row in upset_data.iterrows():
            # 获取该组合包含的癌症类型
            combo_cancers = [cancer for cancer in cancer_types if row[cancer]]
            combo_size = row['count']

            # 只处理包含至少min_cancers种癌症的组合
            if len(combo_cancers) >= min_cancers:
                valid_combinations += 1
                # 计算共有基因
                common_genes = get_common_genes(cancer_genes, combo_cancers)
                common_genes_list = sorted(common_genes)

                # 组合名称
                combo_name = "、".join(combo_cancers)

                # 输出到控制台
                print(f"\n{combo_name} 的共有基因 ({len(common_genes_list)}个):")
                print(", ".join(common_genes_list[:10]) + ("..." if len(common_genes_list) > 10 else ""))

                # 写入文件
                f.write(f"===== {combo_name} 的共有基因 ({len(common_genes_list)}个) =====\n")
                f.write(", ".join(common_genes_list) + "\n\n")

    if valid_combinations == 0:
        print(f"\n没有找到包含至少{min_cancers}种癌症的基因交集")
        os.remove(gene_output_path)  # 删除空文件
    else:
        print(f"\n包含至少{min_cancers}种癌症的共有基因已保存至: {gene_output_path}")

    # 绘制Upset图
    upset_data = upset_data.set_index(cancer_types)['count']
    plt.figure(figsize=(12, 8))

    usp.plot(
        upset_data,
        sort_by='cardinality',
        sort_categories_by='input',
        show_counts=True,
        element_size=40,
    )

    plt.title("不同癌症类型间的基因交集分布", fontsize=16)

    # 保存图片
    output_path = os.path.join(output_dir, "cancer_gene_upset.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Upset图已保存至: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='绘制Upset图并输出多种癌症的共有基因')
    parser.add_argument('--gene-freq',
                        default='/Users/wuyang/Documents/MyPaper/3/gsVis/output/HEST/Homo sapiens/gene_frequency.csv',
                        help='gene_frequency.csv文件路径')
    parser.add_argument('--output-dir', default='./upset_plots', help='输出目录')
    parser.add_argument('--min-size', type=int, default=1, help='最小交集大小，默认1')
    parser.add_argument('--max-sets', type=int, default=10, help='最多显示的癌症类型数量，默认8')
    parser.add_argument('--min-cancers', type=int, default=2, help='最少癌症种类，默认1种')

    args = parser.parse_args()

    print("解析基因频率数据...")
    cancer_genes = parse_gene_frequency(args.gene_freq)

    print("绘制Upset图并计算共有基因...")
    plot_upset(
        cancer_genes,
        args.output_dir,
        min_size=args.min_size,
        max_sets=args.max_sets,
        min_cancers=args.min_cancers
    )


if __name__ == "__main__":
    main()
