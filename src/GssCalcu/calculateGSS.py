from config import FindLatentRepresentationsConfig, LatentToGeneConfig
from find_latent_representation import run_find_latent_representation
from latent_to_gene import run_latent_to_gene
import warnings
import json

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")


# 原始数据地址
# /home/wuyang/hest-data/process/ (服务器地址)
# /Users/wuyang/Documents/MyPaper/3/dataset/HEST-data/ (本地地址)
work_dir = "/home/wuyang/hest-data/process/"

# 数据索引地址
# /home/wuyang/hest-data/cancer_samples.json (服务器地址)
# /Users/wuyang/Documents/MyPaper/3/dataset/cancer_samples.json (本地地址)
with open('/home/wuyang/hest-data/cancer_samples.json', 'r', encoding='utf-8') as f:
    hestData = json.load(f)

n = 5 # 每种癌症下载的样本数
for specie, data in hestData.items():
    if specie == 'Homo sapiens':
        for cancer, sample_ids in data['cancer_types'].items():
            ids_to_query = sample_ids[:] # min(n, len(sample_ids))
            for id in ids_to_query:
                print(f"正在处理{specie}物种{cancer}癌症的{id}样本数据...")

                # 计算潜在表示
                latent_config = FindLatentRepresentationsConfig(
                    input_hdf5_path = work_dir + f"{specie}/{cancer}/{id}_adata.h5ad", # work_dir + sample_id + f"/spatial_transcriptomics.h5ad",
                    workdir = work_dir + f"{specie}/{cancer}", # work_dir + sample_id,
                    sample_name = id , # sample_name,
                    data_layer="X",
                    n_comps=20,  # 潜在空间维度
                    #annotation = "annotation_type" #注释类型
                    # 其他参数...
                )
                run_find_latent_representation(latent_config)


                # 计算GSS
                gss_config = LatentToGeneConfig(
                    #input_hdf5_path="your_data_with_latent.h5ad",
                    workdir = work_dir + f"{specie}/{cancer}", # work_dir + sample_id,
                    sample_name = id , # sample_name,
                    #annotation = "annotation_type" #注释类型
                    # 其他参数...
                )
                run_latent_to_gene(gss_config)