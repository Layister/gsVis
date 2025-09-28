import warnings
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
from matplotlib.patches import Rectangle

# 忽略绘图警告
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

# 设置中文字体支持
plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def gene_cooccurrence_analysis(
        file_path,
        cancer_colors,
        save_image=True,
        image_path="./gene_cooccurrence_plots/gene_networks.png",
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

        # 显示涉及的癌症类型
        cancer_types = data_df['cancer_type'].unique()
        print(f"涉及的癌症类型：{', '.join(cancer_types[:5])}{'...' if len(cancer_types) > 5 else ''}")
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

        # 筛选要显示的边：仅保留两个节点都是top 10基因的边
        # 包括同一癌症内部和不同癌症之间的top 10基因
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

        edges_file = "./gene_cooccurrence_plots/gene_pairs.csv"
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
        pos = nx.spring_layout(G, k=0.5, seed=42)

        # 节点样式（大小由频率决定，颜色由连接度决定）
        node_sizes = [G.nodes[gene]['frequency'] * 80 for gene in G.nodes]
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
            font_size=8,
            font_weight='bold',
            ax=ax
        )

        # 癌症类型框选
        all_cancers = set()
        for gene in G.nodes:
            all_cancers.update(G.nodes[gene]['cancers'])

        for cancer in all_cancers:
            cancer_genes = [
                gene for gene in G.nodes
                if cancer in G.nodes[gene]['cancers']
            ]

            if not cancer_genes:
                continue

            positions = [pos[gene] for gene in cancer_genes]
            min_x = min(p[0] for p in positions) - 0.1
            max_x = max(p[0] for p in positions) + 0.1
            min_y = min(p[1] for p in positions) - 0.1
            max_y = max(p[1] for p in positions) + 0.1

            # 使用指定颜色
            color = cancer_colors.get(cancer, '#AAAAAA')
            rect = Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                fill=False,
                edgecolor=color,
                linewidth=2,
                linestyle='--',
                label=cancer
            )
            ax.add_patch(rect)

        # 颜色条和图例
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('基因连接度')

        # 限制图例数量
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 10:
            handles = handles[:10]
            labels = labels[:10] + ['...']
        plt.legend(handles, labels, title="癌症类型", loc='upper right', bbox_to_anchor=(1.25, 1))

        plt.title('基因共现网络（仅显示Top10高频基因之间的边）', fontsize=14)
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
    file_path = "/Users/wuyang/Documents/MyPaper/3/gsVis/output/HEST/Homo sapiens/cancer_specific_genes.csv"

    custom_colors = {
        'GBM': '#FFA07A',
        'COAD': '#FF6B6B',
        'COADREAD': '#4ECDC4',
        'CSCC': '#45B7D1',
        'EPM': '#98D8C8'
    }

    gene_network, gene_pairs = gene_cooccurrence_analysis(
        file_path=file_path,
        cancer_colors=custom_colors
    )

    if gene_pairs is not None and not gene_pairs.empty:
        print("\n前10对最常共现的基因：")
        print(gene_pairs.head(10))