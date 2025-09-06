from config import FindLatentRepresentationsConfig, LatentToGeneConfig
from find_latent_representation import run_find_latent_representation
from latent_to_gene import run_latent_to_gene
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

# Mus musculus
# LIHB:NCBI627, MEL:ZEN81, PRAD:NCBI793, SKCM:NCBI689

# Homo sapiens
# ACYC:NCBI771, ALL:TENX134, BLCA:NCBI855, CESC:TENX50, COAD:TENX156,
# COADREAD:TENX139, CSCC:NCBI770, EPM:NCBI641, GBM:TENX138, HCC:TENX120,
# HGSOC:TENX142, IDC:TENX99, ILC :TENX96, LNET:TENX72, LUAD:TENX141,
# LUSC:TENX62, PAAD:TENX140, PRAD:TENX157, PRCC:TENX105, READ:ZEN49,
# SCCRCC:INT24, SKCM:TENX158,

# 自定义地址
work_dir = "/Users/wuyang/Documents/MyPaper/3/gsVis/data/"
sample_name = "Human_Breast_Cancer"
sample_id = "BRCA"

# HEST数据地址
tools = 'HEST'
species = 'Homo sapiens' # 'Mus musculus'
cancer = 'SKCM' # 'LIHB'
id = 'TENX158' # 'NCBI627'


latent_config = FindLatentRepresentationsConfig(
    input_hdf5_path = work_dir + f"{tools}/{species}/{cancer}/{id}_adata.h5ad", # work_dir + sample_id + f"/spatial_transcriptomics.h5ad",
    workdir = work_dir + f"{tools}/{species}/{cancer}", # work_dir + sample_id,
    sample_name = id , # sample_name,
    data_layer="X",
    n_comps=20,  # 潜在空间维度
    #annotation = "annotation_type" #注释类型
    # 其他参数...
)
run_find_latent_representation(latent_config)


# 2. 计算GSS
gss_config = LatentToGeneConfig(
    #input_hdf5_path="your_data_with_latent.h5ad",
    workdir = work_dir + f"{tools}/{species}/{cancer}", # work_dir + sample_id,
    sample_name = id , # sample_name,
    #annotation = "annotation_type" #注释类型
    # 其他参数...
)
run_latent_to_gene(gss_config)