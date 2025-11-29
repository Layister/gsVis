"""配置参数文件"""


class AnalysisConfig:
    """分析参数配置"""

    # 数据路径
    root_dir = "../../output/HEST/Homo sapiens/"
    sample_paths = [
        # root_dir + "COAD/TENX89",
        # root_dir + "COAD/TENX90",
        # root_dir + "COAD/TENX91",
        # root_dir + "COAD/TENX92",

        root_dir + "EPM/NCBI629",
        root_dir + "EPM/NCBI630",
        root_dir + "EPM/NCBI631",
        root_dir + "EPM/NCBI632",
        root_dir + "EPM/NCBI633",

        root_dir + "IDC/NCBI681",
        root_dir + "IDC/NCBI682",
        root_dir + "IDC/NCBI683",
        root_dir + "IDC/NCBI684",
        root_dir + "IDC/TENX13",
        root_dir + "IDC/TENX14",

        root_dir + "PAAD/NCBI569",
        root_dir + "PAAD/NCBI570",
        root_dir + "PAAD/NCBI571",
        root_dir + "PAAD/NCBI572",

        root_dir + "PRAD/INT25",
        root_dir + "PRAD/INT26",
        root_dir + "PRAD/INT27",
        root_dir + "PRAD/INT28",
        root_dir + "PRAD/TENX40",
        root_dir + "PRAD/TENX46",

        # root_dir + "READ/ZEN36",
        # root_dir + "READ/ZEN40",
        # root_dir + "READ/ZEN48",
        # root_dir + "READ/ZEN49",
    ]

    # 去噪基因集
    noise_genes = {
        'mitochondrial': ['MT-'] + [f'MT-{i}' for i in range(1, 20)],
        'ribosomal': [f'RPS{i}' for i in range(1, 30)] + [f'RPL{i}' for i in range(1, 40)],
        'heat_shock': ['HSPA', 'HSPB', 'HSPC', 'HSPD', 'HSPE', 'HSPH'] +
                      [f'HSP{i}' for i in range(1, 15)]
    }

    # 三维特征参数
    feature_params = {
        'top_genes_ranked': 2000,
        'top_core_genes': 50,
        'min_cell_types': 3
    }

    # 多模态相似度权重
    similarity_weights = {
        "gene_overlap": 0.2,
        "functional": 0.3,
        "expression_rank": 0.25,
        "cellular_context": 0.25
    }

    # 分层共识参数
    consensus_params = {
        'intra_cancer_threshold': 0.1,
        'pan_cancer_threshold': 0.1,
        'min_sample_coverage': 0.3,
        'hypergeom_cutoff': 0.01,
        'min_structure_size': 2
    }

    # 其他参数
    batch_correction = True
    enrichment_cutoff = 0.05
    output_dir = "enhanced_pancancer_analysis"
    parallel_workers = 40

    # 癌症类型映射
    cancer_type_mapping = {
        'COAD': '结肠癌', 'IDC': '浸润性导管癌', 'BRCA': '乳腺癌',
        'LUAD': '肺腺癌', 'LUSC': '肺鳞癌', 'PRAD': '前列腺癌',
        'SKCM': '皮肤黑色素瘤', 'PAAD': '胰腺癌', 'LIHC': '肝细胞癌',
        'EPM': '恶性外周神经鞘瘤', 'BLCA': '膀胱癌', 'ESCA': '食管癌',
        'GBM': '胶质母细胞瘤', 'HNSC': '头颈鳞癌', 'READ': '直肠癌'
    }

    # 本地GMT文件路径
    local_gmt_files = {
        "GO_BP": "/home/wuyang/hest-data/gseapy/GO_Biological_Process_2025.gmt",
        "GO_CC": "/home/wuyang/hest-data/gseapy/GO_Cellular_Component_2025.gmt",
        "GO_MF": "/home/wuyang/hest-data/gseapy/GO_Molecular_Function_2025.gmt",
        "KEGG": "/home/wuyang/hest-data/gseapy/KEGG_2021_Human.gmt",
        "Hallmark": "/home/wuyang/hest-data/gseapy/MSigDB_Hallmark_2020.gmt"
    }

    enrichment_gene_sets = ["GO_BP", "GO_MF", "KEGG", "Hallmark"]