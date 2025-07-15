from config import FindLatentRepresentationsConfig, LatentToGeneConfig
from find_latent_representation import run_find_latent_representation
from latent_to_gene import run_latent_to_gene
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

# 地址
work_dir = "/Users/wuyang/Documents/MyPaper/3/gsVis/data/"
sample_name = "Human_Breast_Cancer"
sample_id = "BRCA/"


latent_config = FindLatentRepresentationsConfig(
    input_hdf5_path = work_dir + sample_id + "/spatial_transcriptomics.h5ad",
    workdir = work_dir + sample_id,
    sample_name = sample_name,
    data_layer="X",
    n_comps=20,  # 潜在空间维度
    #annotation = "annotation_type" #注释类型
    # 其他参数...
)
run_find_latent_representation(latent_config)


# 2. 计算GSS
gss_config = LatentToGeneConfig(
    #input_hdf5_path="your_data_with_latent.h5ad",
    workdir = work_dir + sample_id,
    sample_name = sample_name,
    #annotation = "annotation_type" #注释类型
    # 其他参数...
)
run_latent_to_gene(gss_config)