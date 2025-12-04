"""

# data_dir = "/home/wuyang/hest-data/process/"
    file_root = "../../output/HEST"

    species = "Homo sapiens"
    cancer_type = "COAD"
    id = "TENX89"

    # 配置路径
    # feather_path = os.path.join(data_dir, species, cancer_type, id, f"latent_to_gene/{id}_gene_marker_score.feather")
    # h5ad_path = os.path.join(data_dir, species, cancer_type, id, f"find_latent_representations/{id}_add_latent.h5ad")
    genes_path = os.path.join(file_root, species, cancer_type, id)
    cellmarker_path = "/home/wuyang/hest-data/CellMarker_Cell_marker.xlsx"  # CellMarker参考数据
    domain_feature_path = genes_path + "/spot_domain_features.json"
    output_dir = genes_path + "/cell_type_inference/"
    os.makedirs(output_dir, exist_ok=True)

    run_r_sct(
        spot_domain_json_path = domain_feature_path,
        cellmarker_xlsx_path = cellmarker_path,
        out_prefix = output_dir
    )


"""