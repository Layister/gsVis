import warnings
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import os
import re
from matplotlib.patches import Rectangle

# 忽略绘图警告
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


def gene_cooccurrence_analysis(
        file_path,
        vis_type = 'ConvexHull', #'ConvexHull', 'Rect'
        max_visulize = 10,
        cancer_colors=None,  # 改为可选参数
        save_image=True,
        image_path="./Plot/gene_cooccurrence_plots/gene_networks.png",
):
    # 创建输出目录
    output_dir = os.path.dirname(image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 读取CSV文件
        data_df = pd.read_csv(
            file_path,
            sep='\t',
            header=0,
            names=['cancer_type', 'gene', 'frequency_in_cancer', 'samples'],
        )

        print(f"成功加载数据：{len(data_df)}条记录")

        # 根据 cancer_colors 筛选要分析的癌症类型
        if cancer_colors is not None and len(cancer_colors) > 0:
            # 只分析 cancer_colors 中包含的癌症类型
            data_df = data_df[data_df['cancer_type'].isin(cancer_colors.keys())]
            print(f"根据 cancer_colors 筛选癌症类型：{list(cancer_colors.keys())}")
        else:
            # 如果没有提供 cancer_colors，为所有癌症类型生成明显不同的颜色
            all_cancer_types = data_df['cancer_type'].unique()

            # 使用更加鲜明且易于区分的颜色
            distinct_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
                '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
                '#5254a3', '#6b6ecf', '#9c9ede', '#3182bd', '#e6550d'
            ]

            cancer_colors = {}
            for i, cancer_type in enumerate(all_cancer_types):
                # 循环使用颜色列表，确保每种癌症都有明显不同的颜色
                color_index = i % len(distinct_colors)
                cancer_colors[cancer_type] = distinct_colors[color_index]

            print(f"未提供 cancer_colors，分析所有癌症类型：{list(all_cancer_types)}")
            print(f"为 {len(all_cancer_types)} 种癌症类型分配了明显不同的颜色")

        # 显示涉及的癌症类型
        cancer_types = data_df['cancer_type'].unique()
        if len(cancer_types) == 0:
            print("错误：没有找到可分析的癌症类型数据")
            return None, None

        print(f"分析的癌症类型：{', '.join(cancer_types)}")
        print(f"共 {len(cancer_types)} 种癌症类型")

        # 为每种癌症提取Top10高频基因
        cancer_top_genes = {}
        for cancer in cancer_types:
            cancer_data = data_df[data_df['cancer_type'] == cancer]
            top_genes = cancer_data.sort_values(
                'frequency_in_cancer',
                ascending=False
            )['gene'].head(10).tolist()
            cancer_top_genes[cancer] = top_genes
            print(f"{cancer}的Top10高频基因：{', '.join(top_genes[:5])}...")

        # 收集所有癌症的Top10基因
        all_top_genes = set()
        for genes in cancer_top_genes.values():
            all_top_genes.update(genes)
        all_top_genes = list(all_top_genes)
        print(f"\n所有癌症的Top10高频基因共 {len(all_top_genes)} 个")

        # 构建全网络（包含所有节点）
        G = nx.Graph()

        # 添加节点（包含所有基因）
        for _, row in data_df.iterrows():
            gene = row['gene']
            cancer_type = row['cancer_type']

            # 解析样本
            samples = set()
            if pd.notna(row['samples']):
                samples = set(str(row['samples']).replace(' ', '').split(','))

            # 添加或更新节点
            if G.has_node(gene):
                G.nodes[gene]['cancers'].add(cancer_type)
                G.nodes[gene]['samples'].update(samples)
            else:
                G.add_node(
                    gene,
                    frequency=row['frequency_in_cancer'],
                    cancers={cancer_type},
                    samples=samples
                )

        # 添加边（计算所有基因对的共现关系）
        genes = list(G.nodes)
        for i in range(len(genes)):
            gene_i = genes[i]
            samples_i = G.nodes[gene_i]['samples']

            for j in range(i + 1, len(genes)):
                gene_j = genes[j]
                samples_j = G.nodes[gene_j]['samples']

                common_samples = samples_i & samples_j
                if common_samples:
                    if G.has_edge(gene_i, gene_j):
                        G[gene_i][gene_j]['weight'] += len(common_samples)
                    else:
                        G.add_edge(gene_i, gene_j, weight=len(common_samples))

        # 网络统计
        print(f"\n网络基本统计：")
        print(f"节点数量：{G.number_of_nodes()}")
        print(f"总边数量（原始）：{G.number_of_edges()}")
        if G.number_of_nodes() > 0:
            print(f"平均连接度：{sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

        # 保存基因度信息
        gene_degree_data = []
        for gene in G.nodes:
            degree = G.degree(gene)
            cancers = ', '.join(G.nodes[gene]['cancers'])
            frequency = G.nodes[gene]['frequency']
            gene_degree_data.append({
                'gene': gene,
                'degree': degree,
                'cancers': cancers,
                'frequency': frequency
            })

        # 按度降序排序并保存
        gene_degree_df = pd.DataFrame(gene_degree_data)
        gene_degree_df = gene_degree_df.sort_values('degree', ascending=False)
        degree_file = "./Plot/gene_cooccurrence_plots/gene_degree.csv"
        gene_degree_df.to_csv(degree_file, index=False)


        # 筛选要显示的边：仅保留两个节点都是top 10基因的边
        edges_to_draw = []
        edge_weights = []
        for u, v, d in G.edges(data=True):
            u_is_top = u in all_top_genes
            v_is_top = v in all_top_genes

            # 只有两个节点都是top 10基因才保留边
            if u_is_top and v_is_top:
                edges_to_draw.append((u, v))
                edge_weights.append(d['weight'] / 2)  # 权重缩放

        print(f"筛选后显示的边数量：{len(edges_to_draw)}")

        # 保存基因对信息（仅包含要显示的边）
        edges_data = []
        for u, v in edges_to_draw:
            edges_data.append([u, v, G[u][v]['weight']])

        edges_df = pd.DataFrame(edges_data, columns=['Gene1', 'Gene2', 'Common_Samples'])
        edges_df = edges_df.sort_values('Common_Samples', ascending=False)

        edges_file = "./Plot/gene_cooccurrence_plots/gene_pairs.csv"
        edges_df.to_csv(edges_file, index=False)
        print(f"\n基因对信息已保存至：{edges_file}")
        print(f"共保存了 {len(edges_df)} 对基因关系")

        # 如果没有节点，提前返回
        if G.number_of_nodes() == 0:
            print("警告：没有有效基因节点可绘制网络")
            return G, edges_df

        # 绘制网络（使用全网络节点）
        plt.figure(figsize=(14, 12))
        ax = plt.gca()

        # 布局（基于全网络计算）
        # pos = nx.spring_layout(G, k=0.5, seed=42)
        pos = nx.spring_layout(G, k=0.3, seed=42, iterations=50)

        # 节点样式（大小由频率决定，颜色由连接度决定）
        node_sizes = [G.nodes[gene]['frequency'] * 10 for gene in G.nodes]
        node_colors = [G.degree(gene) for gene in G.nodes]

        # 绘制所有节点（包括非top 10基因）
        scatter = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            alpha=0.8,
            ax=ax
        )

        # 仅绘制筛选后的边（两个都是top 10基因）
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges_to_draw,
            width=edge_weights,
            alpha=0.3,
            edge_color='gray',
            ax=ax
        )

        # 标签：标记各癌症的Top10基因
        labels = {gene: gene for gene in all_top_genes if gene in G.nodes}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=3,
            font_weight='bold',
            ax=ax
        )

        # 癌症类型框选 - 只框选在 cancer_colors 中的癌症类型
        for cancer in cancer_colors.keys():
            if cancer not in cancer_types:  # 跳过数据中不存在的癌症类型
                continue

            cancer_genes = [
                gene for gene in G.nodes
                if cancer in G.nodes[gene]['cancers']
            ]

            if not cancer_genes:
                continue

            # 使用指定颜色
            color = cancer_colors.get(cancer, '#AAAAAA')

            #可选不同的方法框圈基因
            if vis_type == 'ConvexHull': # 使用凸包框圈
                positions = [pos[gene] for gene in cancer_genes]
                if len(positions) >= 3:  # 凸包需要至少3个点
                    points = np.array(positions)
                    hull = ConvexHull(points)

                    # 绘制凸包
                    polygon = plt.Polygon(points[hull.vertices],
                                          fill=False,
                                          edgecolor=color,
                                          linewidth=1,
                                          linestyle='--',
                                          label=cancer)
                    ax.add_patch(polygon)
                else:
                    # 对于少于3个点的情况，使用小矩形
                    min_x = min(p[0] for p in positions) - 0.03
                    max_x = max(p[0] for p in positions) + 0.03
                    min_y = min(p[1] for p in positions) - 0.03
                    max_y = max(p[1] for p in positions) + 0.03

                    rect = Rectangle(
                        (min_x, min_y), max_x - min_x, max_y - min_y,
                        fill=False, edgecolor=color, linewidth=1, linestyle='--', label=cancer
                    )
                    ax.add_patch(rect)
            else: # 使用矩形框圈
                positions = [pos[gene] for gene in cancer_genes]
                min_x = min(p[0] for p in positions) - 0.1
                max_x = max(p[0] for p in positions) + 0.1
                min_y = min(p[1] for p in positions) - 0.1
                max_y = max(p[1] for p in positions) + 0.1

                rect = Rectangle(
                    (min_x, min_y),
                    max_x - min_x,
                    max_y - min_y,
                    fill=False,
                    edgecolor=color,
                    linewidth=1,
                    linestyle='--',
                    label=cancer
                )
                ax.add_patch(rect)

        # 颜色条和图例
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('genetic connectivity')

        # 限制图例数量
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > max_visulize:
            handles = handles[:max_visulize]
            labels = labels[:max_visulize] + ['...']
        plt.legend(handles, labels, title="cancer type", loc='upper right', bbox_to_anchor=(1.25, 1))

        plt.title('Gene Co-occurrence Network (top 10 co-occurring genes)', fontsize=14)
        plt.axis('off')

        # 保存/显示
        if save_image:
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            print(f"\n网络图像已保存至：{image_path}")
        else:
            plt.show()

        return G, edges_df

    except Exception as e:
        print(f"分析过程出错: {str(e)}")
        return None, None


if __name__ == "__main__":
    file_path = "../output/HEST2/Homo sapiens/cancer_specific_genes.csv"

    """
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
    '#5254a3', '#6b6ecf', '#9c9ede', '#3182bd', '#e6550d'
    """

    custom_colors = {
        #'COAD':'#393B79',
        'COADREAD': '#843C39',
        #'EPM': '#98D8C8',
        'GBM': '#FF6B6B',
        'HCC': '#AEC7E8',
        'HGSOC':'#8C6D31',
        #'IDC': '#4ECDC4',
        'ILC': '#1F77B4',
        'LUAD': '#E377C2',
        #'PAAD': '#FFA07A',
        #'PRAD': '#45B7D1',
        'READ': '#FF7F0E',
        #'SCCRCC': '#2CA02C',
        'SKCM': '#E6550D',
    }

    gene_network, gene_pairs = gene_cooccurrence_analysis(
        file_path=file_path,
        cancer_colors=custom_colors
    )

    if gene_pairs is not None and not gene_pairs.empty:
        print("\n前10对最常共现的基因：")
        print(gene_pairs.head(10))

