"""配置参数文件"""

class AnalysisConfig:
    """分析参数配置"""

    # --------------------
    # 数据路径
    # --------------------
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

        root_dir + "READ/ZEN36",
        root_dir + "READ/ZEN40",
        root_dir + "READ/ZEN48",
        root_dir + "READ/ZEN49",
    ]

    # --------------------
    # spot 内细胞类型推测置信度阈值
    # --------------------
    conf_threshold = 0.3

    # --------------------
    # 分层共识参数
    # --------------------
    consensus_params = {
        'intra_cancer_threshold': 0.7,
        'pan_cancer_threshold': 0.7,

        'min_mp_size': 3,
        'min_mp_cancer_types': 2,

        'min_sample_coverage': 0.3,

        'hypergeom_cutoff': 0.01,
        'min_structure_size': 2
    }

    # --------------------
    # 运行 & 输出参数
    # --------------------
    enrichment_cutoff = 0.05
    output_dir = "enhanced_pancancer_analysis"
    parallel_workers = 40
    batch_correction = True

    # --------------------
    # 癌症类型映射
    # --------------------
    cancer_type_mapping = {
        'COAD': '结肠癌', 'IDC': '浸润性导管癌', 'BRCA': '乳腺癌',
        'LUAD': '肺腺癌', 'LUSC': '肺鳞癌', 'PRAD': '前列腺癌',
        'SKCM': '皮肤黑色素瘤', 'PAAD': '胰腺癌', 'LIHC': '肝细胞癌',
        'EPM': '恶性外周神经鞘瘤', 'BLCA': '膀胱癌', 'ESCA': '食管癌',
        'GBM': '胶质母细胞瘤', 'HNSC': '头颈鳞癌', 'READ': '直肠癌'
    }

    # --------------------
    # 本地 GMT 基因集
    # --------------------
    local_gmt_files = {
        "Hallmark": "/home/wuyang/hest-data/gseapy/MSigDB_Hallmark_2020.gmt"
    }
    enrichment_gene_sets = ["Hallmark"]


    # --------------------
    # 功能轴（Functional Axes）
    # --------------------
    functional_axes = {

        # Inflammation & Immune
        "axis_TNFA_SIGNALING": ["TNF-alpha Signaling via NF-kB"],
        "axis_IL6_JAK_STAT3": ["IL-6/JAK/STAT3 Signaling"],
        "axis_INTERFERON_ALPHA": ["Interferon Alpha Response"],
        "axis_INTERFERON_GAMMA": ["Interferon Gamma Response"],
        "axis_INFLAMMATORY_RESPONSE": ["Inflammatory Response"],

        # Hypoxia & Metabolism
        "axis_HYPOXIA": ["Hypoxia"],
        "axis_CHOLESTEROL_HOMEOSTASIS": ["Cholesterol Homeostasis"],
        "axis_FATTY_ACID_METABOLISM": ["Fatty Acid Metabolism"],
        "axis_GLYCOLYSIS": ["Glycolysis"],
        "axis_OXIDATIVE_PHOSPHORYLATION": ["Oxidative Phosphorylation"],
        "axis_ADIPOGENESIS": ["Adipogenesis"],
        "axis_PEROXISOME": ["Peroxisome"],
        "axis_BILE_ACID_METABOLISM": ["Bile Acid Metabolism"],
        "axis_HEME_METABOLISM": ["Heme Metabolism"],
        "axis_MTORC1": ["MTORC1 Signaling"],
        "axis_PI3K_AKT_MTOR": ["PI3K-AKT-MTOR Signaling"],

        # Proliferation
        "axis_G2M_CHECKPOINT": ["G2-M Checkpoint"],
        "axis_MITOTIC_SPINDLE": ["Mitotic Spindle"],
        "axis_DNA_REPAIR": ["DNA Repair"],
        "axis_E2F_TARGETS": ["E2F Targets"],
        "axis_MYC_TARGETS_V1": ["MYC Targets V1"],
        "axis_MYC_TARGETS_V2": ["MYC Targets V2"],

        # Stress responses
        "axis_APOPTOSIS": ["Apoptosis"],
        "axis_UV_RESPONSE_UP": ["UV Response UP"],
        "axis_UV_RESPONSE_DOWN": ["UV Response DN"],
        "axis_REACTIVE_OXYGEN_SPECIES": ["Reactive Oxygen Species Pathway"],
        "axis_P53_PATHWAY": ["P53 Pathway"],
        "axis_UNFOLDED_PROTEIN_RESPONSE": ["Unfolded Protein Response"],
        "axis_PROTEIN_SECRETION": ["Protein Secretion"],
        "axis_COMPLEMENT": ["Complement"],
        "axis_COAGULATION": ["Coagulation"],

        # Development & differentiation
        "axis_EPITHELIAL_MESENCHYMAL_TRANSITION": ["Epithelial Mesenchymal Transition"],
        "axis_NOTCH_SIGNALING": ["Notch Signaling"],
        "axis_HEDGEHOG_SIGNALING": ["Hedgehog Signaling"],
        "axis_WNT_BETA_CATENIN_SIGNALING": ["Wnt-beta Catenin Signaling"],
        "axis_TGF_BETA_SIGNALING": ["TGF-beta Signaling"],
        "axis_KRAS_SIGNALING_UP": ["KRAS Signaling UP"],
        "axis_KRAS_SIGNALING_DOWN": ["KRAS Signaling DOWN"],
        "axis_ESTROGEN_RESPONSE_EARLY": ["Estrogen Response Early"],
        "axis_ESTROGEN_RESPONSE_LATE": ["Estrogen Response Late"],
        "axis_ANDROGEN_RESPONSE": ["Androgen Response"],
        "axis_MYOGENESIS": ["Myogenesis"],

        # Angiogenesis & ECM
        "axis_ANGIOGENESIS": ["Angiogenesis"],
        "axis_APICAL_JUNCTION": ["Apical Junction"],
        "axis_EPITHELIAL_JUNCTION": ["Apical Surface"],
        "axis_ECM_RECEPTOR": ["Epithelial Mesenchymal Transition"],  # EMT 涉及 ECM 重塑

        # Others
        "axis_SPERMATOGENESIS": ["Spermatogenesis"],
        "axis_XENOBIOTIC_METABOLISM": ["Xenobiotic Metabolism"],
        "axis_ALLOGRAFT_REJECTION": ["Allograft Rejection"],
        "axis_HUMORAL_IMMUNE_RESPONSE": ["IL2 STAT5 Signaling"],  # 与 B 细胞活化一致
        "axis_INTERFERON_STAT": ["Interferon Gamma Response"],  # STAT1 hallmark

    }

    # --------------------
    # 新相似度权重（仅 cell_context + functional）
    # --------------------
    csm_weights = [0.4, 0.6]
    mp_weights = [0.4, 0.6]
