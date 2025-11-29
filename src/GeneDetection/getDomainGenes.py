import os
import json
import warnings
from src.Utils.DataProcess import read_data
from marker_genes import analyze_at_spot_domain_level

warnings.filterwarnings('ignore')

data_dir = "/home/wuyang/hest-data/process/"
output_root = "../../output/HEST"

species = "Homo sapiens"
cancer_type = "PRAD"
id = "INT28"

output_dir = os.path.join(output_root, species, cancer_type, id)
feather_path = os.path.join(data_dir, species, cancer_type, id, f"latent_to_gene/{id}_gene_marker_score.feather")
h5ad_path = os.path.join(data_dir, species, cancer_type, id, f"find_latent_representations/{id}_add_latent.h5ad")

# 读取数据
mk_score_df, adata = read_data(feather_path, h5ad_path)

# 筛选微域特征基因
result = analyze_at_spot_domain_level(mk_score_df=mk_score_df, adata=adata, output_dir=output_dir)

# 转换为可序列化格式
serializable_result = {}
for spot_name, spot_data in result.items():
    serializable_result[spot_name] = {
        'domain_spots': spot_data['domain_spots'],
        'domain_size': spot_data['domain_size'],
        'coordinates': spot_data['coordinates'],
        'feature_genes': spot_data['feature_genes'],
        'num_feature_genes': spot_data['num_feature_genes'],
        'gene_avg_expr_domain': spot_data['gene_avg_expr_domain'],
        'gene_avg_expr_global': spot_data['gene_avg_expr_global'],
        'fold_changes': spot_data['fold_changes']
    }

# 保存结果
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/spot_domain_features.json", 'w') as f:
    json.dump(serializable_result, f, indent=2)