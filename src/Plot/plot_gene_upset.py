import os
import pandas as pd
import matplotlib.pyplot as plt
import upsetplot as usp
from collections import defaultdict
import argparse
import warnings

# 忽略绘图警告
warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
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


def plot_upset(cancer_genes, output_dir, min_size=1, selected_cancers=None, min_cancers=1):
    """绘制Upset图并输出多种癌症的共有基因"""
    os.makedirs(output_dir, exist_ok=True)

    # 如果指定了要分析的癌症类型，则只分析这些类型
    if selected_cancers is not None and len(selected_cancers) > 0:
        # 只保留数据中存在的癌症类型
        available_cancers = [cancer for cancer in selected_cancers if cancer in cancer_genes]
        missing_cancers = set(selected_cancers) - set(available_cancers)

        if missing_cancers:
            print(f"警告：以下癌症类型在数据中不存在: {', '.join(missing_cancers)}")

        if not available_cancers:
            print("错误：没有找到指定的癌症类型数据")
            return

        cancer_genes = {cancer: cancer_genes[cancer] for cancer in available_cancers}
        print(f"分析指定的癌症类型: {', '.join(available_cancers)}")
    else:
        # 如果没有指定癌症类型，使用所有癌症类型
        print(f"分析所有癌症类型: {', '.join(cancer_genes.keys())}")

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
        if os.path.exists(gene_output_path):
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

    plt.title("Distribution of Gene Overlap Among Different Cancer Types", fontsize=16)

    # 保存图片
    output_path = os.path.join(output_dir, "cancer_gene_upset.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Upset图已保存至: {output_path}")
    return output_path


if __name__ == "__main__":

    cancers = [
        #'COAD',
        'COADREAD',
        #'EPM',
        'GBM',
        'HCC',
        'HGSOC',
        #'IDC',
        'ILC',
        'LUAD',
        #'PAAD',
        #'PRAD',
        'READ',
        #'SCCRCC',
        'SKCM',
    ]

    # COADREAD_TENX70, IDC_NCBI684, IDC_NCBI681, PRAD_INT28, PRAD_INT27, READ_ZEN40, READ_ZEN36, SCCRCC_INT21, SCCRCC_INT20, SCCRCC_INT18, SCCRCC_INT17, SCCRCC_INT11, COAD_TENX156, COAD_TENX155, COAD_TENX154, COAD_TENX152, COAD_TENX149, COAD_TENX148, COAD_TENX147, COAD_MISC73, COAD_MISC67, COAD_MISC64, COAD_MISC57, COAD_MISC49, COAD_MISC48, COAD_MISC41, COAD_MISC39, COAD_MISC37, COAD_MISC36, COAD_MISC35, COAD_MISC34, COAD_MISC33, COAD_TENX128, COAD_TENX92, COAD_TENX91, COAD_TENX90, COAD_TENX89, COAD_TENX49, COAD_ZEN42

    parser = argparse.ArgumentParser(description='绘制Upset图并输出多种癌症的共有基因')
    parser.add_argument('--gene-freq',
                        default='../output/HEST2/Homo sapiens/gene_frequency.csv',
                        help='gene_frequency.csv文件路径')
    parser.add_argument('--output-dir', default='./Plot/upset_plots', help='输出目录')
    parser.add_argument('--min-size', type=int, default=2, help='最小交集大小，默认1')
    parser.add_argument('--selected-cancers', default=cancers, help='指定要分析的癌症类型')
    parser.add_argument('--min-cancers', type=int, default=2, help='最少癌症种类，默认2种')

    args = parser.parse_args()

    print("解析基因频率数据...")
    cancer_genes = parse_gene_frequency(args.gene_freq)

    # 处理选定的癌症类型
    print("绘制Upset图并计算共有基因...")
    plot_upset(
        cancer_genes,
        args.output_dir,
        min_size=args.min_size,
        selected_cancers=args.selected_cancers,
        min_cancers=args.min_cancers
    )